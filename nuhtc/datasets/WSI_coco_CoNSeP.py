import logging
import time
from collections import OrderedDict
from PIL import Image
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO
from pycocotools import mask as maskUtils
import mmcv
import numpy as np
import os
import cv2

from nuhtc.utils.logger import log_image
from nuhtc.utils.hooks.mask_vis_hook import imshow_det_bboxes
from nuhtc.utils.stats_utils import get_fast_aji, get_fast_aji_plus, get_fast_pq, get_fast_dice, pair_coordinates, \
    get_pairwise_iou
from pycocotools import coco
import scipy.io as sio
from . import WSICocoDataset

try:
    import wandb
except:
    wandb = None

@DATASETS.register_module()
class CoNSePCocoDataset(WSICocoDataset):
    CLASSES = ('other', 'inflammatory', 'epithelial', 'spindle-shaped')
    ori_shape = (1000, 1000)
    img_size = (256, 256)
    img_stride = 93

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
        # self.cat2label = {cat_id: 0 for i, cat_id in enumerate(self.cat_ids)}
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

    def get_img_inst_map(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Labels'
        label_li = os.listdir(label_path)
        ori_img_dict = {}
        ori_label_dict = {}
        for label in label_li:
            name, ext = os.path.splitext(label)
            ori_img_dict[name] = []
            ori_label_dict[name] = []
            img_data = sio.loadmat(f'{label_path}/{name}.mat')
            inst_map = img_data['inst_map']
            inst_type = img_data['inst_type']
            inst_map = inst_map.astype(np.int)
            inst_map = np.eye(inst_map.max() + 1, k=-1, dtype=np.uint8)[inst_map][:, :, :-1]
            inst_type = inst_type.astype(np.int).flatten()
            inst_type[(inst_type == 3) | (inst_type == 4)] = 3
            inst_type[(inst_type == 5) | (inst_type == 6) | (inst_type == 7)] = 4
            inst_type = inst_type - 1
            ori_img_dict[name] = list(map(lambda x: maskUtils.encode(np.asfortranarray(x)), inst_map.transpose((2, 0, 1)))) 
            ori_label_dict[name] = inst_type
            del inst_map
            del inst_type
        return ori_img_dict, ori_label_dict

    def get_img(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Images'
        label_li = os.listdir(label_path)
        ori_img_dict = {}
        for label in label_li:
            name, ext = os.path.splitext(label)
            inst_map = np.array(Image.open(f'{label_path}/{name}.png'))
            ori_img_dict[name] = inst_map
            del inst_map
        return ori_img_dict

    def get_img_name(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Images'
        label_li = os.listdir(label_path)
        name_li = []
        for label in label_li:
            name, ext = os.path.splitext(label)
            name_li.append(name)
        return name_li

    def mask2RLE(self, args):
        pred_mask, shift_x, shift_y = args
        tmp_pred_mask = np.zeros(self.ori_shape, dtype=np.uint8)
        tmp_pred_mask[shift_y:shift_y + self.img_size[1],
        shift_x:shift_x + self.img_size[0]] = pred_mask
        return maskUtils.encode(np.asfortranarray(tmp_pred_mask))

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
                 mode='instance',
                 overlay=False,
                 save_path='infer',
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

        seg_mask = [mmcv.concat_list(tmp_mask[1]) for tmp_mask in results]
        fg_thr = 0.1
        VIS_CLASSES = list(self.CLASSES + ('Background',))
        num_classes = len(self.CLASSES)
        confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
        ori_img_id = self.get_img_name()
        if not hasattr(self, 'ori_img'):
            self.ori_img = self.get_img()
        if not hasattr(self, 'ori_mask'):
            self.ori_mask, self.ori_label = self.get_img_inst_map()
        ori_img_dict = self.ori_img
        ori_masks_dict, ori_labels_dict = self.ori_mask, self.ori_label
        if save:
            save_path = f'{save_path}/{self.__class__.__name__}'
            os.makedirs(save_path, exist_ok=True)
        pred_masks_dict = {}
        pred_scores_dict = {}
        pred_labels_dict = {}
        for img_id in ori_img_id:
            pred_masks_dict[img_id] = []
            pred_scores_dict[img_id] = []
            pred_labels_dict[img_id] = []
        if seg_mask != []:
            bbox_results = [np.concatenate(tmp_res[0]) for tmp_res in results]
            labels = [np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(tmp_res[0])])
                      for tmp_res in results]
            fg_scores = [tmp_box[:, 4] for tmp_box in bbox_results]
            random_idx = np.random.randint(len(seg_mask))
            for idx, tmp_mask in enumerate(seg_mask):
                img_id = self.img_ids[idx]
                img_info = self.coco.imgs[img_id]
                img_name = "_".join(os.path.splitext(img_info['file_name'])[0].split('_')[:-1])
                img_loc = int(os.path.splitext(img_info['file_name'])[0].split('_')[-1])
                patch_x, patch_y = img_loc % 9, img_loc // 9
                fg_score = fg_scores[idx]
                tmp_area = maskUtils.area(tmp_mask)
                select_id = (fg_score >= fg_thr)
                # select_id = (fg_score >= fg_thr) & (tmp_area>20)

                # shift_pred_bboxes = np.zeros((pred_bboxes.shape[0], 7))
                discard_offset = 4
                shift_x, shift_y = patch_x * self.img_stride, patch_y * self.img_stride
                if shift_x != 0:
                    select_id[bbox_results[idx][:, 0] < discard_offset] = False
                if shift_x != self.ori_shape[0] - self.img_size[0]:
                    select_id[bbox_results[idx][:, 2] > self.img_size[0]-discard_offset] = False
                if shift_y != 0:
                    select_id[bbox_results[idx][:, 1] < discard_offset] = False
                if shift_y != self.ori_shape[1] - self.img_size[1]:
                    select_id[bbox_results[idx][:, 3] > self.img_size[0]-discard_offset] = False
                pred_bboxes = bbox_results[idx][select_id][:, :4]
                pred_scores = bbox_results[idx][select_id][:, 4]
                pred_labels = labels[idx][select_id]

                # if tmp_mask = [], then an error will occur
                if tmp_mask != []:
                    pred_mask_li = coco.maskUtils.decode(tmp_mask)[:, :, select_id].transpose((2, 0, 1))
                else:
                    pred_mask_li = []

                if len(pred_mask_li) != 0:
                    # multiprocess version
                    # pool = Pool(processes=5)
                    # RLE_encode_mask = pool.map(self.mask2RLE, list(zip(pred_mask_li, [shift_x]*len(pred_mask_li), [shift_y]*len(pred_mask_li))))
                    # pool.close()
                    # pool.join()

                    RLE_encode_mask = map(self.mask2RLE, list(zip(pred_mask_li, [shift_x]*len(pred_mask_li), [shift_y]*len(pred_mask_li))))
                    RLE_encode_mask = list(RLE_encode_mask)
                    pred_scores = list(pred_scores)
                    pred_masks_dict[img_name] += RLE_encode_mask
                    pred_scores_dict[img_name] += pred_scores
                    pred_labels_dict[img_name] += pred_labels.tolist()

                if idx == random_idx and len(pred_mask_li) != 0 and wandb.run is not None:
                    ann = self.get_ann_info(idx)
                    gt_bboxes = ann['bboxes']
                    gt_labels = ann['labels']
                    true_mask_li = ann['masks']
                    if len(true_mask_li) != 0 and type(true_mask_li) == list:
                        true_mask_li = coco.maskUtils.decode(true_mask_li).transpose((2, 0, 1))

                    img_ori = np.array(Image.open(f'{self.img_prefix}/{self.coco.load_imgs(img_id)[0]["file_name"]}'))
                    gt_masks = np.array(true_mask_li)
                    gt_bboxes = np.array(gt_bboxes)
                    gt_labels = np.array(gt_labels)

                    pairwise_inter, pairwise_union = get_pairwise_iou(true_mask_li, pred_mask_li)
                    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

                    true_id_list = range(len(true_mask_li))
                    pred_id_list = range(len(pred_mask_li))

                    match_iou = 0.5
                    pairwise_iou[pairwise_iou <= match_iou] = 0.0
                    paired_true, paired_pred = np.nonzero(pairwise_iou)

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
                            pred_mask_li[unpaired_pred],
                            class_names=VIS_CLASSES,
                            show=False)
                        log_image(f"evaluate/fp seg", img_fp_seg, interval=1)

            confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
            stat_res = {}
            mpq_info_list = []
            ori_img_id.sort()
            for img_id in ori_img_id:
                true_masks = ori_masks_dict[img_id]
                true_labels = ori_labels_dict[img_id]
                pred_masks = pred_masks_dict[img_id]
                pred_scores = pred_scores_dict[img_id]
                pred_labels = pred_labels_dict[img_id]

                pred_masks, nms_idx = self.mask_nms(pred_masks, pred_scores, thr=0.02)
                pred_labels = np.array(pred_labels)[nms_idx]
                # TODO hard to swap to multi process due to large image numbers
                tmp_res = self.stat_calc(true_masks, pred_masks, match_iou=0.5)
                tmp_multi_res = self.mutlti_stat_calc(true_masks, pred_masks, true_labels, pred_labels)
                mpq_info = []
                # aggregate the stat info per class
                for single_class_pq in tmp_multi_res:
                    tp = single_class_pq[0]
                    fp = single_class_pq[1]
                    fn = single_class_pq[2]
                    sum_iou = single_class_pq[3]
                    mpq_info.append([tp, fp, fn, sum_iou])
                mpq_info_list.append(mpq_info)

                res_info = f'\n{img_id.rjust(8)}'
                for k, v in tmp_res.items():
                    if k not in stat_res.keys():
                        stat_res[k] = [v]
                    else:
                        stat_res[k].append(v)
                    if k not in ['tp', 'fp', 'fn', 'iou']:
                        res_info = f"{res_info}, {k}:{v:.4f}"

                if len(pred_masks) != 0:
                    self.calculate_confusion_matrix(confusion_matrix, true_masks, pred_masks, true_labels, pred_labels)
                else:
                    self.calculate_confusion_matrix(confusion_matrix, true_masks, pred_masks, true_labels, [])

                if save:
                    img_info = {
                        'height': ori_img_dict[img_id].shape[0],
                        'width': ori_img_dict[img_id].shape[1]
                    }
                    pred_inst_masks = self.convert_format(pred_masks, pred_labels, img_info, data_format='conic')
                    pred_bboxs = maskUtils.toBbox(pred_masks)
                    pred_centroids = np.zeros((pred_bboxs.shape[0], 2))
                    pred_centroids[:, 0] = pred_bboxs[:, 0] + pred_bboxs[:, 2]/2
                    pred_centroids[:, 1] = pred_bboxs[:, 1] + pred_bboxs[:, 3]/2
                    pred_mat = {
                        'inst_map': pred_inst_masks[:, :, 0],
                        'inst_type': np.reshape(pred_labels+1, (-1, 1)),
                        'inst_centroid': pred_centroids,
                        'inst_uid': np.array(range(1, len(pred_labels)+1)).reshape((-1, 1))

                    }
                    sio.savemat(f'{save_path}/{img_id}.mat', pred_mat)

                if overlay:
                    img_overlay = np.copy(ori_img_dict[img_id])
                    inst_rng_colors = self.random_colors(len(pred_masks))
                    inst_rng_colors = np.array(inst_rng_colors) * 255
                    inst_rng_colors = inst_rng_colors.astype(np.uint8)
                    line_thickness=2
                    for i in range(len(pred_masks)):
                        inst_colour = inst_rng_colors[i].tolist()
                        inst_map = maskUtils.decode(pred_masks[i])
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
                    Image.fromarray(img_overlay).convert('RGB').save(f'{save_path}/overlay/{img_id}_overlay.png')

                print(res_info)

        eval_results = OrderedDict()
        if stat_res:
            for k, v in stat_res.items():
                if k not in ['tp', 'fp', 'fn', 'iou']:
                    print(f"\n{k}:{np.mean(v)}")
                    eval_results[k] = np.mean(v)

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

        if wandb and wandb.run is not None:
            cm_table = wandb.Table(columns=VIS_CLASSES, rows=VIS_CLASSES, data=confusion_matrix)
            wandb.log({"evaluate/confusion matrix": cm_table}, commit=False)
            self.plot_confusion_matrix(confusion_matrix, VIS_CLASSES, wandb_log=True)

        return eval_results
