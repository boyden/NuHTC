import warnings
import logging
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import colorsys
import cv2
import sys

proj_path = '/home/sul084/immune-decoder/segmentation/NuHTC'
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)

from collections import OrderedDict
from PIL import Image
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval

import mmcv
import numpy as np
from mmcv.utils import print_log
import torch
from terminaltables import AsciiTable
from itertools import compress

from nuhtc.utils.logger import log_image
from nuhtc.utils.hooks.mask_vis_hook import imshow_det_bboxes
from nuhtc.utils.stats_utils import get_fast_aji, get_fast_aji_plus, get_fast_pq, get_fast_dice, pair_coordinates, \
    get_pairwise_iou
from pycocotools import coco
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment

from matplotlib.ticker import MultipleLocator
try:
    import wandb
except:
    wandb = None

@DATASETS.register_module()
class WSICocoDataset(CocoDataset):
    CLASSES = ('nucleus',)
    PALETTE = None

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def random_colors(self, N, bright=True):
        """Generate random colors.

        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['ismask'] = results['ann_info']['ismask']

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @property
    def min_size(self):
        return 10

    @property
    def min_area(self):
        return 10

    def _filter_imgs(self, min_size=10):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            # TODO these may influence the performance
            # check hover net config
            # if min(img_info['width'], img_info['height']) >= min_size:
            if img_info['width'] * img_info['height'] > min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ismask = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            # original if ann.get('iscrowd', False)
            # if ann.get('iscrowd', False) or 0 in [x1, y1] or x1+w == img_info['width'] or y1+h==img_info['height']:
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # default ismask unless specify it
                gt_masks_ann.append(ann.get('segmentation'))
                gt_ismask.append(ann.get('ismask', 1))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            ismask=np.array(gt_ismask, dtype=np.uint8),
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _mask_post_process(self, pred_mask_li):
        if len(pred_mask_li) == 0:
            return pred_mask_li, None
        pred_mask_area = pred_mask_li.sum(axis=(1, 2))
        select_id = pred_mask_area >= self.min_area
        if np.sum(select_id) == 0:
            return pred_mask_li[select_id], select_id
        else:
            pred_mask_li = pred_mask_li[select_id]
            pred_mask_area = pred_mask_area[select_id]
            # descending sort
            # mask_area_idx = pred_mask_area.argsort()
            # pred_mask_li = pred_mask_li[mask_area_idx]
            # pred_mask_area = pred_mask_area[mask_area_idx]

        mask_overlap = (pred_mask_li[:, None, :, :] * pred_mask_li[None, :, :, :]).sum(axis=(2, 3))
        mask_area = np.diag(pred_mask_area)
        mask_overlap = mask_overlap - mask_area
        # mask_area_min = np.minimum(pred_mask_area[:, None], pred_mask_area[None, :])
        # mask_ratio_min = mask_overlap/mask_area_min
        # the ratio between the maximum overlap and own area
        mask_ratio = mask_overlap / pred_mask_area
        mask_ratio = mask_ratio.max(axis=1)
        select_id = mask_ratio < 0.999

        pred_mask_li = pred_mask_li[select_id]
        return pred_mask_li, select_id

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 save=False,
                 save_path='infer',
                 overlay=False,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        data_format = kwargs.get('format', 'conic')
        seg_mask = [mmcv.concat_list(tmp_mask[1]) for tmp_mask in results]
        fg_thr = 0.1
        stat_res = {}
        mpq_info_list = []
        VIS_CLASSES = list(self.CLASSES + ('Background',))
        num_classes = len(self.CLASSES)
        confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
        assert len(self) == len(results)
        if save:
            pred_array = []
            # true_array = []
            save_path = f'{save_path}/{self.__class__.__name__}'
            os.makedirs(save_path, exist_ok=True)
        if seg_mask != []:
            bbox_results = [np.concatenate(tmp_res[0]) for tmp_res in results]
            labels = [np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(tmp_res[0])])
                      for tmp_res in results]
            fg_scores = [tmp_box[:, 4] for tmp_box in bbox_results]
            random_idx = np.random.randint(len(seg_mask))
            for idx, tmp_mask in enumerate(seg_mask):
                img_id = self.img_ids[idx]
                img_info = self.coco.imgs[img_id]
                fg_score = fg_scores[idx]
                select_id = fg_score >= fg_thr

                ann = self.get_ann_info(idx)
                gt_bboxes = ann['bboxes']
                gt_labels = ann['labels']
                true_mask_li = ann['masks']
                if len(true_mask_li) != 0 and type(true_mask_li[0]) == list:
                    true_mask_li = [coco.maskUtils.encode(coco.annToMask(gt_seg)) for gt_seg in true_mask_li]

                if tmp_mask != []:
                    pred_mask_li = [tmp_mask[i] for i in range(len(tmp_mask)) if select_id[i]]
                    pred_score_li = fg_score[select_id]
                    labels[idx] = labels[idx][select_id]
                else:
                    pred_mask_li = []
                    pred_score_li = []
                if wandb is None:
                    raise ImportError("wandb is not installed")

                if idx == random_idx and len(pred_mask_li) != 0 and len(true_mask_li) != 0 and wandb.run is not None:
                    img_ori = np.array(Image.open(f'{self.img_prefix}/{self.coco.load_imgs(img_id)[0]["file_name"]}'))
                    gt_masks = coco.maskUtils.decode(true_mask_li).transpose((2, 0, 1))
                    gt_bboxes = np.array(gt_bboxes)
                    gt_labels = np.array(gt_labels)

                    pred_masks = coco.maskUtils.decode(pred_mask_li).transpose((2, 0, 1))
                    pred_bboxes = bbox_results[idx][select_id][:, :4]
                    pred_labels = labels[idx]

                    pairwise_inter, pairwise_union = get_pairwise_iou(gt_masks, pred_masks)
                    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

                    true_id_list = range(len(gt_masks))
                    pred_id_list = range(len(pred_masks))

                    match_iou = 0.5
                    pairwise_iou[pairwise_iou <= match_iou] = 0.0
                    paired_true, paired_pred = np.nonzero(pairwise_iou)
                    #### Munkres pairing with scipy library
                    # the algorithm return (row indices, matched column indices)
                    # if there is multiple same cost in a row, index of first occurence
                    # is return, thus the unique pairing is ensure
                    # inverse pair to get high IoU as minimum
                    # paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)

                    # get the actual FP and FN
                    unpaired_true = [idx for idx in true_id_list if idx not in paired_true]
                    unpaired_pred = [idx for idx in pred_id_list if idx not in paired_pred]

                    if len(unpaired_true) != 0:
                        img_fn_seg = imshow_det_bboxes(
                            img_ori,
                            gt_bboxes[unpaired_true],
                            gt_labels[unpaired_true],
                            gt_masks[unpaired_true],
                            class_names=VIS_CLASSES,
                            show=False)
                        log_image(f"evaluate/fn seg", img_fn_seg, interval=1)

                        img_gt_seg = imshow_det_bboxes(
                            img_ori,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            class_names=VIS_CLASSES,
                            show=False)
                        log_image(f"evaluate/gt seg", img_gt_seg, interval=1)

                    if len(unpaired_pred) != 0:
                        img_fp_seg = imshow_det_bboxes(
                            img_ori,
                            pred_bboxes[unpaired_pred],
                            pred_labels[unpaired_pred],
                            pred_masks[unpaired_pred],
                            class_names=VIS_CLASSES,
                            show=False)
                        log_image(f"evaluate/fp seg", img_fp_seg, interval=1)

                if len(pred_mask_li) != 0:
                    pred_mask_li, nms_idx = self.mask_nms(pred_mask_li, pred_score_li, thr=0.05)
                    labels[idx] = labels[idx][nms_idx]

                # TODO hard to swap to multi process due to large image numbers
                tmp_res = self.stat_calc(true_mask_li, pred_mask_li, match_iou=0.5)
                tmp_multi_res = self.mutlti_stat_calc(true_mask_li, pred_mask_li, gt_labels, labels[idx])
                mpq_info = []
                # aggregate the stat info per class
                for single_class_pq in tmp_multi_res:
                    tp = single_class_pq[0]
                    fp = single_class_pq[1]
                    fn = single_class_pq[2]
                    sum_iou = single_class_pq[3]
                    mpq_info.append([tp, fp, fn, sum_iou])
                mpq_info_list.append(mpq_info)

                if len(pred_mask_li) != 0:
                    self.calculate_confusion_matrix(confusion_matrix, true_mask_li, pred_mask_li, gt_labels, labels[idx])
                else:
                    self.calculate_confusion_matrix(confusion_matrix, true_mask_li, pred_mask_li, gt_labels, [])

                if save:
                    # true_mask = self.convert_format(true_mask_li, gt_labels, img_info, data_format=data_format)
                    pred_mask = self.convert_format(pred_mask_li, labels[idx], img_info, data_format=data_format)

                    # np.save(f'{save_path}/{os.path.splitext(img_info["filename"])[0]}.npy', true_masks)
                    if data_format == 'consep':
                        sio.savemat(f'{save_path}/{os.path.splitext(img_info["filename"])[0]}.mat', pred_mask)
                    else:
                        np.save(f'{save_path}/{os.path.splitext(img_info["filename"])[0]}.npy', pred_mask)

                    # true_array.append(true_mask)
                    pred_array.append(pred_mask)

                if overlay:
                    img_overlay = np.array(Image.open(f'{self.img_prefix}/{img_info["filename"]}').convert('RGB'))
                    line_thickness=2
                    for i in range(len(pred_mask_li)):
                        inst_colour = self.PALETTE[int(labels[idx][i])]
                        inst_map = maskUtils.decode(pred_mask_li[i])
                        inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if len(inst_contour[0]) == 0:
                            continue
                        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
                        if inst_contour.shape[0] < 3:
                            continue
                        if len(inst_contour.shape) != 2:
                            continue # ! check for trickery shape
                        cv2.drawContours(img_overlay, [inst_contour], -1, inst_colour, line_thickness)
                    os.makedirs(f'{save_path}/overlay', exist_ok=True)
                    Image.fromarray(img_overlay).convert('RGB').save(f'{save_path}/overlay/{os.path.splitext(img_info["filename"])[0]}_overlay.png')
                    del img_overlay

                if tmp_res:
                    for k, v in tmp_res.items():
                        if k not in stat_res.keys():
                            stat_res[k] = [v]
                        else:
                            stat_res[k].append(v)

                del pred_mask_li
                del true_mask_li

        eval_results = OrderedDict()
        if stat_res:
            for k, v in stat_res.items():
                if k not in ['tp', 'fp', 'fn', 'iou']:
                    print(f"\n{k}:{np.mean(v)}")
                    eval_results[k] = np.mean(v)
        del stat_res
        if len(mpq_info_list) != 0:
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # sum over all the images
            total_mpq_info_metrics = np.nansum(mpq_info_metrics, axis=0)

            mpq_list = []
            # for each class, get the multiclass PQ
            for cat_idx in range(total_mpq_info_metrics.shape[0]):
                total_tp = total_mpq_info_metrics[cat_idx][0]
                total_fp = total_mpq_info_metrics[cat_idx][1]
                total_fn = total_mpq_info_metrics[cat_idx][2]
                total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                # get the F1-score i.e DQ
                dq = total_tp / (
                    (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                )
                # get the SQ, when not paired, it has 0 IoU so does not impact
                sq = total_sum_iou / (total_tp + 1.0e-6)
                print(f"\nmulti_pq+_{cat_idx}:{dq * sq}")
                mpq_list.append(dq * sq)
                eval_results[f'multi_pq+_{cat_idx}'] = dq * sq
            mpq_metrics = np.mean(mpq_list)
            print(f'\nmulti_pq+:{mpq_metrics}')
            eval_results['multi_pq+'] = mpq_metrics

            dq = mpq_info_metrics[:, :, 0]/(mpq_info_metrics[:, :, 0]+0.5*mpq_info_metrics[:, :, 1] + 0.5*mpq_info_metrics[:, :, 2]+1.0e-6)
            sq = mpq_info_metrics[:, :, 3]/(mpq_info_metrics[:, :, 0]+1.0e-6)
            mpq_metrics = dq*sq
            mpq_metrics = np.nanmean(mpq_metrics, axis=0)
            # for each class, get the multiclass PQ
            for cat_idx in range(len(self.CLASSES)):
                print(f"\nmulti_pq_{cat_idx}:{mpq_metrics[cat_idx]}")
                eval_results[f'multi_pq_{cat_idx}'] = mpq_metrics[cat_idx]
            print(f'\nmulti_pq:{np.mean(mpq_metrics)}')
            eval_results['multi_pq'] = np.mean(mpq_metrics)

        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = confusion_matrix.astype(np.float32) * 100 / (per_label_sums + 1e-5)

        if wandb.run is not None:
            cm_table = wandb.Table(columns=VIS_CLASSES, rows=VIS_CLASSES, data=confusion_matrix)
            wandb.log({"evaluate/confusion matrix": cm_table}, commit=False)
            self.plot_confusion_matrix(confusion_matrix, VIS_CLASSES, wandb_log=True)
        if save:
            if data_format != 'consep':
                # np.save(f'{save_path}/trues.npy', np.array(true_array))
                np.save(f'{save_path}/preds_{data_format}.npy', np.array(pred_array))
            self.plot_confusion_matrix(confusion_matrix, VIS_CLASSES, save_dir=save_path)

        return eval_results

    def stat_calc(self, true_masks, pred_masks, match_iou=0.5):
        if len(true_masks) == 0 and len(pred_masks) == 0:
            return None
        if len(true_masks) == 0 and len(pred_masks) != 0:
            stat_res = {
                'aji': 0,
                'aji_plus': 0,
                'dq': 0,
                'sq': 0,
                'pq': 0,
                'dice': 0,
                'precision': 0,
                'recall': 0,
                'tp': 0,
                'fp': len(pred_masks),
                'fn': 0,
                'iou': 0
            }
            return stat_res
        if len(true_masks) != 0 and len(pred_masks) == 0:
            stat_res = {
                'aji': 0,
                'aji_plus': 0,
                'dq': 0,
                'sq': 0,
                'pq': 0,
                'dice': 0,
                'precision': 0,
                'recall': 0,
                'tp': 0,
                'fp': 0,
                'fn': len(true_masks),
                'iou': 0
            }
            return stat_res
        if isinstance(true_masks, list):
            pairwise_iou = maskUtils.iou(true_masks, pred_masks, [0] * len(pred_masks))
            pairwise_dice = 2 * pairwise_iou / (1 + pairwise_iou)
            true_area = maskUtils.area(true_masks).reshape((-1, 1))
            pred_area = maskUtils.area(pred_masks).reshape((-1, 1))
            inst_area = true_area + pred_area.T
            pairwise_inter = inst_area * pairwise_iou / (1 + pairwise_iou)
            pairwise_union = inst_area / (1 + pairwise_iou)
            true_masks = maskUtils.decode(true_masks).transpose((2, 0, 1))
            pred_masks = maskUtils.decode(pred_masks).transpose((2, 0, 1))
        else:
            pairwise_inter, pairwise_union = self.mask_iou(true_masks, pred_masks, device='cpu')

        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        aji = get_fast_aji(true_masks, pred_masks,
                           pairwise_inter=pairwise_inter,
                           pairwise_union=pairwise_union)
        aji_plus = get_fast_aji_plus(true_masks, pred_masks,
                                     pairwise_inter=pairwise_inter,
                                     pairwise_union=pairwise_union,
                                     paired_true=paired_true,
                                     paired_pred=paired_pred)
        pq = get_fast_pq(true_masks, pred_masks,
                         pairwise_inter=pairwise_inter,
                         pairwise_union=pairwise_union,
                         paired_true=paired_true,
                         paired_pred=paired_pred,
                         match_iou=match_iou)
        tp = len(pq[1][0])
        fp = len(pq[1][3])
        fn = len(pq[1][2])
        iou_sum = pq[0][1]*(tp+1e-6)
        dice = get_fast_dice(true_masks, pred_masks,
                             pairwise_inter=pairwise_inter,
                             pairwise_union=pairwise_union,
                             paired_true=paired_true,
                             paired_pred=paired_pred, )
        stat_res = {
            'aji': aji,
            'aji_plus': aji_plus,
            'dq': pq[0][0],
            'sq': pq[0][1],
            'pq': pq[0][2],
            'dice': dice,
            'precision': tp / (tp + fp + 1e-9),
            'recall': tp / (tp + fn + 1e-9),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'iou': iou_sum
        }
        return stat_res

    def mutlti_stat_calc(self, true_mask_li, pred_mask_li, gt_labels, pred_labels, match_iou=0.5):
        nr_classes = len(self.CLASSES)
        pq = []
        for idx in range(nr_classes):
            pred_inst_oneclass = list(compress(pred_mask_li, pred_labels==idx))
            true_inst_oneclass = list(compress(true_mask_li, gt_labels==idx))
            pq_oneclass_info = self.stat_calc(true_inst_oneclass, pred_inst_oneclass)
            # add (in this order) tp, fp, fn iou_sum
            if pq_oneclass_info:
                pq_oneclass_stats = [
                    pq_oneclass_info['tp'],
                    pq_oneclass_info['fp'],
                    pq_oneclass_info['fn'],
                    pq_oneclass_info['iou'],
                ]
                pq.append(pq_oneclass_stats)
            else:
                pq.append([float('nan'), float('nan'), float('nan'), float('nan')])
        return pq

    @staticmethod
    @torch.no_grad()
    def mask_iou(mask1, mask2, device=None):
        """
        mask1: [m1,n] m1 means number of predicted objects
        mask2: [m2,n] m2 means number of gt objects
        Note: n means image_w x image_h
        """
        # inconsistency with the res of maskUtils.iou, which get the correct res if the dtype of mask1 is uint8
        # float type is with fast speed, other types with very low speed
        if type(mask1) is np.ndarray:
            mask1 = torch.from_numpy(mask1).to(torch.float)
        if type(mask2) is np.ndarray:
            mask2 = torch.from_numpy(mask2).to(torch.float)
        if len(mask1.shape) > 2 or len(mask2.shape) > 2:
            mask1 = mask1.contiguous().view(mask1.shape[0], -1)
            mask2 = mask2.contiguous().view(mask2.shape[0], -1)

        if device is not None and device != 'cpu':
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
        intersection = torch.matmul(mask1, mask2.t())
        area1 = torch.sum(mask1, dim=1).view(-1, 1)
        area2 = torch.sum(mask2, dim=1).view(-1, 1)
        union = (area1 + area2.t()) - intersection
        if device == 'cpu' or device == None:
            return intersection.numpy(), union.numpy()
        else:
            return intersection.detach().cpu().numpy(), union.detach().cpu().numpy()

    @staticmethod
    def mask_iou_np(true_masks, pred_masks):
        """
        mask1: [m1,n] m1 means number of predicted objects
        mask2: [m2,n] m2 means number of gt objects
        Note: n means image_w x image_h
        """
        true_masks = true_masks.astype(float)
        pred_masks = pred_masks.astype(float)
        if len(true_masks.shape) > 2 or len(pred_masks.shape) > 2:
            true_masks = true_masks.reshape((true_masks.shape[0], -1))
            pred_masks = pred_masks.reshape((pred_masks.shape[0], -1))
        pairwise_inter = np.matmul(true_masks, pred_masks.T)
        area1 = np.sum(true_masks.reshape((len(true_masks), -1)), axis=1).reshape(len(true_masks), -1)
        area2 = np.sum(pred_masks.reshape((len(pred_masks), -1)), axis=1).reshape(len(pred_masks), -1)
        pairwise_union = (area1 + area2.T) - pairwise_inter

        return pairwise_inter, pairwise_union

    @staticmethod
    def mask_nms(masks, pred_scores, thr=0.9, min_area=None):
        """https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L98

        Returns:
            tuple: kept dets and indice.
        """
        sort_idx = np.argsort(pred_scores)[::-1]
        mask_len = len(masks)
        tmp_masks = np.array(masks)[sort_idx]
        mask_iou = maskUtils.iou(tmp_masks.tolist(), tmp_masks.tolist(), [0] * mask_len)

        keep_idx = np.ones(mask_len, dtype=np.uint8)
        for i in range(mask_len):
            if not keep_idx[i]:
                continue
            for j in range(i + 1, mask_len):
                if not keep_idx[j]:
                    continue
                tmp_iou = mask_iou[i, j]
                # tmp_iou = maskUtils.iou([tmp_masks[i]], [tmp_masks[j]], [0])[0][0]
                if tmp_iou > thr:
                    keep_idx[j] = 0
        return tmp_masks[keep_idx==1].tolist(), sort_idx[keep_idx==1]

    def calculate_confusion_matrix(self, confusion_matrix, true_masks, pred_masks, gt_labels, pred_labels, tp_iou_thr=0.5):
        mask_ious = maskUtils.iou(true_masks, pred_masks, [0] * len(pred_masks))
        true_positives = np.zeros_like(gt_labels)
        for i, det_label in enumerate(pred_labels):
            det_match = 0
            for j, gt_label in enumerate(gt_labels):
                if mask_ious[j, i] >= tp_iou_thr:
                    det_match += 1
                    # if gt_label == det_label:
                    true_positives[j] += 1  # TP
                    confusion_matrix[gt_label, det_label] += 1
            if det_match == 0:  # BG FP
                confusion_matrix[-1, det_label] += 1
        for num_tp, gt_label in zip(true_positives, gt_labels):
            if num_tp == 0:  # FN
                confusion_matrix[gt_label, -1] += 1

    def _log_confusion_matrix(self, confmatrix, label_x, label_y):
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': label_x, 'y': label_y, 'z': confmatrix,
                                 'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': label_x, 'y': label_y, 'z': confdiag,
                               'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
        fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
        yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}

        fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
        wandb.log({'confusion_matrix': wandb.data_types.Plotly(fig)}, commit=False)
        return {'confusion_matrix': wandb.data_types.Plotly(fig)}

    def plot_confusion_matrix(self,
                              confusion_matrix,
                              labels,
                              save_dir=None,
                              show=False,
                              title='Normalized Confusion Matrix',
                              color_theme='plasma',
                              wandb_log=False):
        """Draw confusion matrix with matplotlib.

        Args:
            confusion_matrix (ndarray): The confusion matrix.
            labels (list[str]): List of class names.
            save_dir (str|optional): If set, save the confusion matrix plot to the
                given path. Default: None.
            show (bool): Whether to show the plot. Default: True.
            title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
            color_theme (str): Theme of the matrix color map. Default: `plasma`.
        """
        # normalize the confusion matrix

        num_classes = len(labels)
        fig, ax = plt.subplots(
            figsize=(num_classes, num_classes * 0.9), dpi=300)
        canvas = fig.canvas
        cmap = plt.get_cmap(color_theme)
        im = ax.imshow(confusion_matrix, cmap=cmap)
        plt.colorbar(mappable=im, ax=ax)

        title_font = {'weight': 'bold', 'size': 16}
        ax.set_title(title, fontdict=title_font)
        label_font = {'size': 12}
        plt.ylabel('Ground Truth Label', fontdict=label_font)
        plt.xlabel('Prediction Label', fontdict=label_font)

        # draw locator
        xmajor_locator = MultipleLocator(1)
        xminor_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(xmajor_locator)
        ax.xaxis.set_minor_locator(xminor_locator)
        ymajor_locator = MultipleLocator(1)
        yminor_locator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(ymajor_locator)
        ax.yaxis.set_minor_locator(yminor_locator)

        # draw grid
        ax.grid(True, which='minor', linestyle='-')

        # draw label
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.tick_params(
            axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
        plt.setp(
            ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

        # draw confution matrix value
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    '{}%'.format(int(confusion_matrix[i, j])),
                    ha='center',
                    va='center',
                    color='w',
                    size=12)

        ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

        fig.tight_layout()
        stream, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        cm_img = buffer.reshape((height, width, -1))
        if wandb_log:
            log_image(f"evaluate/confusion_matrix", cm_img, interval=1)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), format='png')
        if show:
            plt.show()
        plt.close()

    def convert_format(self, mask_li, label_li, img_info, data_format='conic'):
        if data_format == 'pannuke':
            mask = np.zeros((img_info['height'], img_info['width'], len(self.CLASSES)+1), dtype=int)
            if len(mask_li) == 0:
                return mask
            masks = maskUtils.decode(mask_li).transpose((2, 0, 1))
            for i in range(len(self.CLASSES)):
                tmp_masks = masks[label_li==i]
                if len(tmp_masks) == 0:
                    continue
                mask_id_list = np.array(range(1, len(tmp_masks)+1))
                mask[:, :, i] = np.max(tmp_masks*mask_id_list.reshape((-1, 1, 1)), axis=0)
            mask[:, :, -1] = 1 - np.max(masks, axis=0)
        elif data_format == 'consep':
            mask = np.zeros((img_info['height'], img_info['width'], 2), dtype=int)
            if len(mask_li) != 0:
                masks = maskUtils.decode(mask_li).transpose((2, 0, 1))
                mask_id_list = np.array(range(1, len(masks)+1))
                mask[:, :, 0] = np.max(masks*mask_id_list.reshape((-1, 1, 1)), axis=0)
                mask[:, :, 1] = np.max(masks*(label_li + 1).reshape((-1, 1, 1)), axis=0)
                pred_bboxs = maskUtils.toBbox(mask_li)
                pred_centroids = np.zeros((pred_bboxs.shape[0], 2))
                pred_centroids[:, 0] = pred_bboxs[:, 0] + pred_bboxs[:, 2]/2
                pred_centroids[:, 1] = pred_bboxs[:, 1] + pred_bboxs[:, 3]/2
                pred_mat = {
                    'inst_map': mask[:, :, 0],
                    'inst_type': mask[:, :, 1],
                    'inst_centroid': pred_centroids,
                    'inst_uid': np.array(range(1, len(label_li))),
                }
                return pred_mat
            else:
                pred_mat = {
                    'inst_map': mask[:, :, 0],
                    'inst_type': mask[:, :, 1],
                }
                return pred_mat
        else:
            mask = np.zeros((img_info['height'], img_info['width'], 2), dtype=int)
            if len(mask_li) != 0:
                masks = maskUtils.decode(mask_li).transpose((2, 0, 1))
                mask_id_list = np.array(range(1, len(masks)+1))
                mask[:, :, 0] = np.max(masks*mask_id_list.reshape((-1, 1, 1)), axis=0)
                mask[:, :, 1] = np.max(masks*(label_li + 1).reshape((-1, 1, 1)), axis=0)
        return mask