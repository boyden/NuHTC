# Adapted from WSI_coco.py (last updated: 09/09/2025)
# Calculate metrics for Coco-formatted inference results

import os
import cv2
import sys
import csv
import json
import mmcv
import torch
import random
import logging
import colorsys
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mmcv.utils import print_log
from argparse import ArgumentParser
from sklearn.metrics import r2_score

proj_path = '/home/sul084/immune-decoder/segmentation/NuHTC'
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)

from pycocotools import coco
from itertools import compress
from terminaltables import AsciiTable
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment

from collections import OrderedDict, defaultdict
from pycocotools.mask import decode as decode_rle
from mmdet.datasets import DATASETS, CocoDataset # Need this
from mmdet.datasets.api_wrappers import COCO, COCOeval # Need this

from nuhtc.utils.logger import log_image
from nuhtc.utils.hooks.mask_vis_hook import imshow_det_bboxes
from nuhtc.utils.stats_utils import get_fast_aji, get_fast_aji_plus, get_fast_pq, get_fast_dice, pair_coordinates, get_pairwise_iou # this
from nuhtc.datasets.plt_confusion import calculate_confusion_matrix, plot_confusion_matrix
from matplotlib.ticker import MultipleLocator
wandb = None

# --------------------------
# CLI Arguments
# --------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--gt_json", help="Path to COCO-formatted ground truth json file")
    parser.add_argument("--pred_json", help="Path to COCO-formatted prediction json file")
    parser.add_argument("--csv_save_path", help="Path to save metrics summary csv file")
    parser.add_argument("--exp_name", help="Name of experiment, for figure naming")
    parser.add_argument("--eval_type", default="both", choices=["instance", "both"],
                        help="Type of evaluation: 'instance' (instance metrics only) or 'both' (instance + semantic metrics)")
    return parser.parse_args()
    
# --------------------------
# Logger
# --------------------------
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --------------------------
# COCO Dataset Wrapper
# --------------------------
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
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.CLASSES = tuple([c['name'] for c in cats])
        self.cat_ids = [c['id'] for c in cats]
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

    
    def min_size(self):
        return 1 # 10

    
    def min_area(self):
        return 1 # 10

    
    def _filter_imgs(self, min_size=10):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if img_info['width'] * img_info['height'] > min_size:
                valid_inds.append(i)
        return valid_inds

    
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
        gt_bboxes = []            # shape (N, 4)
        gt_labels = []            # shape (N,)
        gt_masks_ann = []         # list of N masks
        gt_ismask = []
        gt_scores = []            # added 5/30 -> for evaluation purposes

        no_gt_reason = ''
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                no_gt_reason = 'inter_w * inter_h == 0'
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                no_gt_reason = "ann['area'] <= 0 or w < 1 or h < 1"
                continue
            ## Commented out 10/02/2025 due to filtering out many cases with mismatched class labels
            if ann['category_id'] not in self.cat_ids:
                no_gt_reason = "ann['category_id'] not in self.cat_ids"
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            # gt_labels.append(0)
            gt_masks_ann.append(ann.get('segmentation'))
            gt_ismask.append(ann.get('ismask', 1))
            gt_scores.append(ann.get('score', 1.0))  # Added
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            name_print = img_info['filename'].replace('jpg', 'png')
            print(f'No GT instances for {name_print}, {no_gt_reason}')
            # print(ann_info)
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            ismask=np.array(gt_ismask, dtype=np.uint8),
            masks=gt_masks_ann,
            scores=np.array(gt_scores, dtype=np.float32),  # Added
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

        mask_overlap = (pred_mask_li[:, None, :, :] * pred_mask_li[None, :, :, :]).sum(axis=(2, 3))
        mask_area = np.diag(pred_mask_area)
        mask_overlap = mask_overlap - mask_area
        mask_ratio = mask_overlap / pred_mask_area
        mask_ratio = mask_ratio.max(axis=1)
        select_id = mask_ratio < 0.999

        pred_mask_li = pred_mask_li[select_id]
        return pred_mask_li, select_id

    def evaluate(self,
                 results,
                 metric='bbox',
                 exp_name='',
                 csv_save_path='',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000), # for mAP
                 iou_thrs=None,
                 metric_items=None,
                 save=False,
                 save_path='infer',
                 overlay=False,
                 eval_type='',
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

        assert isinstance(results, list) and len(results) == 2, "Expected results to be [bbox_results, segm_results]"
        bbox_results, segm_results = results
        num_classes = len(self.CLASSES)
        fg_thr = 0.1  # score threshold to filter predictions
        VIS_CLASSES = list(self.CLASSES + ('Background',))
        confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
        stat_res = {}
        metrics_save = [] # To save each image's metrics as csv
        mpq_info_list = []
        
        assert len(self) == len(bbox_results) == len(segm_results), "Mismatch in number of images"

        if save:
            pred_array = []
            save_path = f'{save_path}/{self.__class__.__name__}'
            os.makedirs(save_path, exist_ok=True)

        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            img_info = self.coco.imgs[img_id]
    
            # Ground truth
            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            true_mask_li = ann['masks']

            # RLE encode GT if polygons
            if len(true_mask_li) != 0 and isinstance(true_mask_li[0], list):
                true_mask_li = [coco.maskUtils.encode(coco.annToMask(gt_seg)) for gt_seg in true_mask_li]
    
            # Prediction: flatten per-class lists to per-instance
            pred_bboxes = []
            pred_labels = []
            pred_masks = []
            pred_scores = []
    
            for cls_idx in range(num_classes):
                for box, mask in zip(bbox_results[idx][cls_idx], segm_results[idx][cls_idx]):
                    score = box[4]
                    if score < fg_thr:
                        continue
                    pred_bboxes.append(box[:4])
                    pred_scores.append(score)
                    pred_labels.append(cls_idx)
                    pred_masks.append(mask)
                # if(cls_idx==0):
                #     if len(bbox_results[idx][cls_idx]) == 0:
                #         print(f'GT len 0 for image {idx} class {cls_idx}')

            # If mask is not empty, filter by foreground threshold
            if len(pred_masks) > 0:
                pred_mask_li = [pred_masks[i] for i in range(len(pred_masks))]  # already filtered by fg_thr
                pred_score_li = pred_scores
                pred_label_li = pred_labels
            else:
                pred_mask_li = []
                pred_score_li = []
                pred_label_li = []

            # Compute per-image stats
            tmp_res = self.stat_calc(true_mask_li, pred_mask_li, match_iou=0.5)
            metrics_save.append(tmp_res)
            tmp_multi_res = self.multi_stat_calc(true_mask_li, pred_mask_li, gt_labels, pred_label_li)

            # Aggregate the stat info per class across all images -> mpq+
            mpq_info = []
            for single_class_pq in tmp_multi_res:
                tp = single_class_pq[0]
                fp = single_class_pq[1]
                fn = single_class_pq[2]
                iou_sum = single_class_pq[3]
                mpq_info.append([tp, fp, fn, iou_sum])
            mpq_info_list.append(mpq_info)

            if len(pred_mask_li) > 0:
                calculate_confusion_matrix(confusion_matrix, true_mask_li, pred_mask_li, gt_labels, pred_label_li)
                # pred_label_li replaces labels[idx][select_id], as we now extract labels per instance directly after flattening predictions by class
            else:
                calculate_confusion_matrix(confusion_matrix, true_mask_li, pred_mask_li, gt_labels, [])

            if tmp_res:
                for k, v in tmp_res.items():
                    if k not in stat_res.keys():
                        stat_res[k] = [v]
                    else:
                        stat_res[k].append(v)

            del pred_mask_li
            del true_mask_li

        # Save per-image metrics as csv
        with open(f'{csv_save_path}_per_image.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_save[0].keys())
            writer.writeheader()
            writer.writerows(metrics_save)

        # To average metrics across all images
        print(f'\nPrinting averaged metrics across all images...')
        eval_results = OrderedDict()

        # For regression analysis (reference: CoNIC)
        true_counts = []
        pred_counts = []

        if stat_res:
            
            # Save aji, aji+, dice, precision, and recall
            for k, v in stat_res.items():
                if k == 'true_count':
                    true_counts.extend(v)
                elif k == 'pred_count':
                    pred_counts.extend(v)
                elif k not in ['tp', 'fp', 'fn', 'iou_sum']:
                    print(f"\n{k}:{np.mean(v)}")
                    eval_results[k] = np.mean(v)
                    
            # Calculate pq+, dq+, and sq+, echoing mpq+ calculation method below
            total_tp = np.sum(stat_res['tp'])
            total_fp = np.sum(stat_res['fp'])
            total_fn = np.sum(stat_res['fn'])
            total_iou_sum = np.sum(stat_res['iou_sum'])

            # Calculate DQ+ (F1)
            dq_plus = total_tp / ((total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
            print(f"\ndq+(f1):{dq_plus}")
            eval_results['dq+(f1)'] = dq_plus
            
            # Calculate SQ+
            sq_plus = total_iou_sum / (total_tp + 1.0e-6)
            print(f"\nsq+:{sq_plus}")
            eval_results['sq+'] = sq_plus
            
            # Calculate PQ+
            pq_plus = dq_plus * sq_plus
            print(f"\npq+:{pq_plus}")
            eval_results['pq+'] = pq_plus
                    
        del stat_res

        # ----------------------------
        # Calculate regression - compute R² once for the dataset
        # ----------------------------
        if len(true_counts) > 1:
            r2 = r2_score(true_counts, pred_counts)
        else:
            r2 = float('nan')
        eval_results['regression_r2'] = r2
        print(f"\nregression_r2:{r2}")

        # ----------------------------
        # For multi-class calculations
        # ----------------------------
        if len(mpq_info_list) != 0 and eval_type == 'both':

            # Sum over all the images -> mPQ+
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            total_mpq_info_metrics = np.nansum(mpq_info_metrics, axis=0)
            mpq_list = []

            # mPQ+
            # For each class, get the multiclass PQ
            for cat_idx in range(total_mpq_info_metrics.shape[0]):
                total_tp = total_mpq_info_metrics[cat_idx][0]
                total_fp = total_mpq_info_metrics[cat_idx][1]
                total_fn = total_mpq_info_metrics[cat_idx][2]
                total_iou_sum = total_mpq_info_metrics[cat_idx][3]

                # Get F1-score i.e DQ
                dq_plus = total_tp / (
                    (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
                
                # Get SQ, when not paired, it has 0 IoU so does not impact
                sq_plus = total_iou_sum / (total_tp + 1.0e-6)
                pq_plus = dq_plus * sq_plus
                print(f"\nmulti-class pq+ class {cat_idx}: {pq_plus}")
                mpq_list.append(pq_plus)
                eval_results[f'pq+_{cat_idx}'] = pq_plus
                
            mpq_metrics_plus = np.mean(mpq_list)
            print(f'\nmulti_pq+:{mpq_metrics_plus}')
            eval_results['multi_pq+'] = mpq_metrics_plus

        # Save aggregate metrics averaged across test dataset as csv
        pd.DataFrame([eval_results]).to_csv(f'{csv_save_path}_avg.csv', index=False)
            
        # ----------------------------
        # Generate confusion matrix
        # ----------------------------
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = confusion_matrix.astype(np.float32) * 100 / (per_label_sums + 1e-5)
        plot_confusion_matrix(confusion_matrix, VIS_CLASSES,
            save_path=f'/home/sul084/immune-decoder/segmentation/evaluate/figures/{exp_name}_confusion.png', title=f'{exp_name} Normalized Confusion Matrix')

        return eval_results
        

    def stat_calc(self, true_masks, pred_masks, match_iou=0.5):
        if len(true_masks) == 0 and len(pred_masks) == 0:
            return None
        if len(true_masks) == 0 and len(pred_masks) != 0:
            stat_res = {
                'aji': 0,
                'aji_plus': 0,
                'dice': 0,
                'precision': 0,
                'recall': 0,
                'tp': 0,
                'fp': len(pred_masks),
                'fn': 0,
                'iou_sum': 0
            }
            return stat_res
        if len(true_masks) != 0 and len(pred_masks) == 0:
            stat_res = {
                'aji': 0,
                'aji_plus': 0,
                'dice': 0,
                'precision': 0,
                'recall': 0,
                'tp': 0,
                'fp': 0,
                'fn': len(true_masks),
                'iou_sum': 0
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
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        iou_sum = pq[0][1] * (tp + 1e-6) # pq[0][1] is sq
        dice = get_fast_dice(true_masks, pred_masks,
                             pairwise_inter=pairwise_inter,
                             pairwise_union=pairwise_union,
                             paired_true=paired_true,
                             paired_pred=paired_pred, )
        # Regression metrics
        true_count = len(true_masks)
        pred_count = len(pred_masks)
    
        stat_res = {
            'aji': aji,
            'aji_plus': aji_plus,
            # 'dq (f1)': pq[0][0], # Same as f1 = 2 * precision * recall / (precision + recall + 1e-9)
            # 'sq': pq[0][1],
            # 'pq': pq[0][2],
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'iou_sum': iou_sum,
            'true_count': true_count,
            'pred_count': pred_count
        }
        return stat_res


    def multi_stat_calc(self, true_mask_li, pred_mask_li, gt_labels, pred_labels, match_iou=0.5):
        nr_classes = len(self.CLASSES)
        pq = []
        
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)

        # Calculate mpq
        for idx in range(nr_classes):
            pred_inst_oneclass = list(compress(pred_mask_li, pred_labels==idx))
            true_inst_oneclass = list(compress(true_mask_li, gt_labels==idx))

            pq_oneclass_info = self.stat_calc(true_inst_oneclass, pred_inst_oneclass)
            if pq_oneclass_info:
                pq_oneclass_stats = [
                    pq_oneclass_info['tp'],
                    pq_oneclass_info['fp'],
                    pq_oneclass_info['fn'],
                    pq_oneclass_info['iou_sum'],
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
        """
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
                if tmp_iou > thr:
                    keep_idx[j] = 0
        return tmp_masks[keep_idx==1].tolist(), sort_idx[keep_idx==1]


## Added 5/29/2025
def extract_pred(pred_coco, num_classes):
    """
    Returns: 
        [bbox_results, segm_results], where
        bbox_results[i][j] = list of [x1, y1, x2, y2, score] for class j in image i
        segm_results[i][j] = list of segmentation (polygon or RLE) for class j in image i
    """
    bbox_results = []
    segm_results = []

    for idx in range(len(pred_coco)):
        ann_info = pred_coco.get_ann_info(idx)
        bboxes = ann_info['bboxes']
        labels = ann_info['labels']
        masks = ann_info['masks']
        scores = ann_info.get('scores', [1.0] * len(labels))  # If scores missing, set as 1.0

        bbox_per_class = [[] for _ in range(num_classes)]
        segm_per_class = [[] for _ in range(num_classes)]

        for bbox, label, mask, score in zip(bboxes, labels, masks, scores):
            # bbox is in [x1, y1, x2, y2] format; append score
            bbox_result = np.append(bbox, score)
            bbox_per_class[label].append(bbox_result)
            segm_per_class[label].append(mask)

        bbox_results.append(bbox_per_class)
        segm_results.append(segm_per_class)

    # Checking 
    assert len(bbox_results) == len(pred_coco), "bbox mismatch in number of images"
    assert len(segm_results) == len(pred_coco), "segm mismatch in number of images"

    for i, (bbox_per_image, segm_per_image) in enumerate(zip(bbox_results, segm_results)):
        assert len(bbox_per_image) == num_classes, f"Image {i} bbox class count mismatch"
        assert len(segm_per_image) == num_classes, f"Image {i} segm class count mismatch"

    # for i in range(3):  # check first 3 images
    #     print(f"Image {i}")
    #     for cls_idx in range(num_classes):
    #         print(f"  Class {cls_idx}: {len(bbox_results[i][cls_idx])} boxes")

    return [bbox_results, segm_results]


def main():

    """
    Evaluate COCO-format instance segmentation predictions against ground truth.
    Args:
        gt_json_path (str): Path to the ground truth JSON file in COCO format.
        pred_json_path (str): Path to the prediction JSON file in COCO format.

    Returns:
        Calculated metrics
    """
    args = parse_args()
    gt_json = args.gt_json
    pred_json = args.pred_json

    sys.stdout = Logger(f"/home/sul084/immune-decoder/segmentation/evaluate/logs/{args.exp_name}.txt")

    pipeline=[]
    gt_coco = WSICocoDataset(ann_file=gt_json, pipeline=pipeline)
    print(f'COCO-formatted ground truth: {gt_coco}')
        
    pred_coco = WSICocoDataset(ann_file=pred_json, pipeline=pipeline)
    print(f'COCO-formatted predictions: {pred_coco}')

    # Prepare results
    results = extract_pred(pred_coco=pred_coco, num_classes=len(gt_coco.CLASSES))
    
    if args.eval_type == "instance":
        print("Evaluating instance segmentation metrics only...")
        eval_results = gt_coco.evaluate(results, metric='segm', exp_name=args.exp_name, csv_save_path=args.csv_save_path)
        # Remove semantic metrics from CSV
        for k in ['multi_pq+']:
            if k in eval_results:
                del eval_results[k]
    else:
        print("Evaluating both instance and semantic metrics...")
        eval_results = gt_coco.evaluate(results, metric='segm', exp_name=args.exp_name, csv_save_path=args.csv_save_path, eval_type=args.eval_type)

    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
