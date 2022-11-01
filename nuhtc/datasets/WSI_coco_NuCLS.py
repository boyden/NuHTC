import copy

from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations
from nuhtc.datasets.pipelines import FOVCrop
from collections import OrderedDict
import mmcv
import numpy as np
from pycocotools import mask as maskUtils
import torch
from . import WSICocoDataset
from .coco_utils import get_coco_api_from_dataset, _update_classification_metrics, map_bboxes_using_hungarian_algorithm
from .coco_eval import CocoEvaluator

from .nucls.configs.nucleus_style_defaults import Interrater, DefaultAnnotationStyles, NucleusCategories
from .nucls.GeneralUtils import connect_to_sqlite, reverse_dict

@DATASETS.register_module()
class NuCLSCocoDataset(WSICocoDataset):
    CLASSES = ('tumor_nonMitotic', 'tumor_mitotic', 'nonTILnonMQ_stromal', 'macrophage', 'lymphocyte', 'plasma_cell', 'other_nucleus', 'AMBIGUOUS')

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

    def _parse_ann_info_eval(self, img_info, ann_info):
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
        gt_iscrowd = []
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
            # exclude AMBIGUOUS category
            gt_iscrowd.append(ann.get('iscrowd'))
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # default ismask unless specify it
                gt_masks_ann.append(self.coco.annToMask(ann))
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
            image_id=img_info['id'],
            boxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            ismask=np.array(gt_ismask, dtype=np.uint8),
            iscrowd=np.array(gt_iscrowd),
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def set_labelmaps(self):
        ncg = NucleusCategories
        if not self.do_classification:
            self.categs_names = ncg.puredet_categs
            self.labelcodes = ncg.raw_to_puredet_categs_codes
            self.rlabelcodes = reverse_dict(ncg.puredet_categs_codes)
        elif self.use_supercategs:
            self.categs_names = ncg.super_categs
            self.labelcodes = ncg.raw_to_super_categs_codes
            self.rlabelcodes = reverse_dict(ncg.super_categs_codes)
        else:
            self.categs_names = ncg.main_categs
            self.labelcodes = ncg.raw_to_main_categs_codes
            self.rlabelcodes = reverse_dict(ncg.main_categs_codes)

    def set_supercateg_labelmaps(self):
        ncg = NucleusCategories
        if self.do_classification:
            self.supercategs_names = ncg.super_categs
            self.supercategs_rlabelcodes = reverse_dict(ncg.super_categs_codes)
            self.main_codes_to_supercategs_names = {
                ncg.main_categs_codes[mc]: sc
                for mc, sc in ncg.main_to_super_categmap.items()
            }
            self.main_codes_to_supercategs_codes = {
                k: ncg.super_categs_codes[v]
                for k, v in self.main_codes_to_supercategs_names.items()
            }

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 maxDets=[1, 100, 300],
                 crop_inference_to_fov=True,
                 use_supercategs=False,
                 save=False):
        # See: https://cocodataset.org/#detection-eval
        bbox_clip_border = True
        self.use_supercategs = use_supercategs
        seg_mask = [mmcv.concat_list(tmp_mask[1]) for tmp_mask in results]

        if seg_mask != []:
            bbox_results = [np.concatenate(tmp_res[0]) for tmp_res in results]
            prob_results = [tmp_res[2] for tmp_res in results]
            labels = [np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(tmp_res[0])]) for tmp_res in results]
            fg_scores = [tmp_box[:, 4] for tmp_box in bbox_results]
        targets = []
        outputs= []
        for idx in range(len(seg_mask)):
            if len(seg_mask[idx]) != 0:
                img_id = self.img_ids[idx]
                ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
                ann_info = self.coco.load_anns(ann_ids)
                ann_info = self._parse_ann_info_eval(self.data_infos[idx], ann_info)
                if crop_inference_to_fov:
                    tar_info = dict(img_info=self.data_infos[idx], ann_info=self.get_ann_info(idx))
                    self.pre_pipeline(tar_info)
                    tar_info = LoadImageFromFile()(tar_info)
                    tar_info = LoadAnnotations(with_bbox=True, with_mask=True, with_seg=False)(tar_info)
                    #TODO bbox crop incorrectly
                    tar_crop = FOVCrop(allow_negative_crop=bbox_clip_border)(tar_info)
                    tar_crop['boxes'] = tar_crop['gt_bboxes']
                    tar_crop['masks'] = tar_crop['gt_masks']
                    tar_crop['labels'] = tar_crop['gt_labels'] + 1
                    tar_crop['image_id'] = img_id
                    tar_crop['iscrowd'] = ann_info['iscrowd']
                    targets.append(tar_crop)
                else:
                    ann_info = copy.deepcopy(ann_info)
                    ann_info['labels'] += 1
                    targets.append(ann_info)

                pred_mask_li = seg_mask[idx]
                pred_score_li = fg_scores[idx]
                if len(pred_mask_li) != 0:
                    pred_mask_li, nms_idx = self.mask_nms(pred_mask_li, pred_score_li, thr=0.05)
                out = {
                    'boxes': bbox_results[idx][:, :4][nms_idx],
                    'labels': labels[idx][nms_idx] + 1,
                    'scores': fg_scores[idx][nms_idx],
                    'probabs': prob_results[idx][nms_idx],
                    'masks': maskUtils.decode(pred_mask_li).transpose((2, 0, 1))
                }

                outputs.append(out)

        # combined detection & classification precision/recall
        res = {
            target["image_id"]: output
            for target, output in zip(targets, outputs)}

        if res == {}:
            return {}
        # iou_types = _get_iou_types(model)
        iou_types = ['bbox']  # segmAP is meaningless in my hybrid bbox/segm dataset
        maxDets = [1, 10, 100] if maxDets is None else maxDets

        # combined detection & classification precision/recall
        dst = self
        dst.do_classification = True
        dst.do_segmentation = True
        dst.set_labelmaps()
        dst.set_supercateg_labelmaps()
        coco = get_coco_api_from_dataset(dst, crop_inference_to_fov=crop_inference_to_fov, targets_all=targets)
        coco_evaluator = CocoEvaluator(coco, iou_types, maxDets=maxDets)

        # precision/recall for just detection (objectness)
        classification = dst.do_classification
        if classification:

            # IMPORTANT: REVERSE ME AFTER DEFINING COCO API
            dst.do_classification = False
            dst.set_labelmaps()

            coco_objectness = get_coco_api_from_dataset(dst, crop_inference_to_fov=crop_inference_to_fov, targets_all=targets)
            coco_evaluator_objectness = CocoEvaluator(coco_objectness, iou_types, maxDets=maxDets)

            # IMPORTANT: THIS LINE IS CRITICAL
            dst.do_classification = True
            dst.set_labelmaps()
            # dst.set_supercateg_labelmaps()

        else:
            # noinspection PyUnusedLocal
            coco_objectness = None
            coco_evaluator_objectness = None

        n_true = 0
        n_pred = 0
        n_matched = 0
        cltargets = []
        clprobabs = []
        cloutlabs = []
        seg_intersects = []
        seg_sums = []

        def _get_categnames(prefix):
            if prefix == '':
                return dst.categs_names
            return dst.supercategs_names

        coco_evaluator.update(res)
        # for k, v in res.items():
        #     coco_evaluator.update({k: v})

        probabs_exist = 'probabs' in outputs[0]

        if classification:

            # Match truth to outputs and only count matched objects for
            # classification accuracy stats
            for target, output in zip(targets, outputs):

                # Match, ignoring ambiguous nuclei. Note that the model
                #  already filters out anything predicted as ignore_label
                #  in inference mode, so we only need to do this for gtruth
                # keep = target['iscrowd'] == 0
                keep = np.ones(len(target['boxes']), dtype=np.bool)
                cltrg_boxes = np.int32(target['boxes'][keep])
                cltrg_labels = np.int32(target['labels'][keep])
                keep_target, keep_output, _, _ = \
                    map_bboxes_using_hungarian_algorithm(
                        bboxes1=cltrg_boxes,
                        bboxes2=np.int32(output['boxes']),
                        min_iou=0.5)

                # classification performance
                n_true += cltrg_boxes.shape[0]
                n_pred += output['boxes'].shape[0]
                n_matched += len(keep_output)
                cltargets.extend(cltrg_labels[keep_target].tolist())
                if probabs_exist:
                    clprobabs.extend(
                        np.float32(output['probabs'])[keep_output, :].tolist()
                    )
                else:
                    cloutlabs.extend(
                        np.int32(output['labels'])[keep_output].tolist()
                    )

                # FIXME: for now, we just assess this if classification because
                #   otherwise I'll need to refactor the function output
                # segmentation performance
                if 'masks' in target:
                    ismask = np.int32(target['ismask'])[keep_target] == 1
                    tmask = np.int32(target['masks'])[keep_target, ...][ismask, ...]
                    # TODO check carefully
                    densify_mask = False
                    if not densify_mask:
                        omask = np.int32(output['masks'] > 0.5)
                    else:
                        omask = np.int32(output['masks'])
                        obj_ids = np.arange(1, omask.max() + 1)
                        omask = omask == obj_ids[:, None, None]
                        omask = 0 + omask
                    omask = omask[keep_output, ...][ismask, ...]
                    for i in range(tmask.shape[0]):
                        sms = tmask[i, ...].sum() + omask[i, ...].sum()
                        isc = np.sum(
                            0 + ((tmask[i, ...] + omask[i, ...]) == 2)
                        )
                        if (sms > 0) and (isc > 0):
                            seg_sums.append(sms)
                            seg_intersects.append(isc)

            for _, output in res.items():
                output['labels'] = 1 + (0 * output['labels'])

            # precision/recall for just detection (objectness)
            coco_evaluator_objectness.update(res)

        # combined detection & classification precision/recall
        # gather the stats from all processes & accumulate preds from all imgs
        # Implemented by the original coco evalucatio code, not released by the NuCLS Paper
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        if classification:
            # Init classification results
            classification_metrics = {
                'n_true_nuclei_excl_ambiguous': n_true,
                'n_predicted_nuclei': n_pred,
                'n_matched_for_classif': n_matched,
            }
            for prefix in ['', 'superCateg_']:
                categs_names = _get_categnames(prefix)
                classification_metrics.update({
                    f'{prefix}{k}': np.nan
                    for k in ['accuracy', 'auroc_micro', 'auroc_macro', 'mcc']
                })
                # Class-by-class
                classification_metrics.update({
                    f'{prefix}accuracy_{cls_name}': np.nan
                    for cls_name in categs_names
                })
                classification_metrics.update({
                    f'{prefix}mcc_{cls_name}': np.nan
                    for cls_name in categs_names
                })
                if probabs_exist:
                    classification_metrics.update({
                        f'{prefix}aucroc_{cls_name}': np.nan
                        for cls_name in categs_names
                    })
            for prefix in ['', 'superCateg_']:
                categs_names = _get_categnames(prefix)
                classification_metrics.update({
                    f'{prefix}confusion_trueClass-{tc}_predictedClass-{pc}': 0
                    for tc in categs_names
                    for pc in categs_names
                })
            # segmentation -- restricted to matched nuclei with available seg
            if len(seg_sums) > 0:
                seg_intersects = np.array(seg_intersects)
                seg_sums = np.array(seg_sums)
                intersect = np.sum(seg_intersects)
                sums = np.sum(seg_sums)
                ious = seg_intersects / (seg_sums - seg_intersects)
                dices = 2. * seg_intersects / seg_sums
                classification_metrics.update({
                    # overall
                    'seg_intersect': intersect,
                    'seg_sum': sums,
                    'seg_IOU': intersect / (sums - intersect),
                    'seg_DICE': 2. * intersect / sums,
                    # by nucleus
                    'seg_n': len(ious),
                    'seg_medIOU': np.median(ious),
                    'seg_medDICE': np.median(dices),
                })

            coco_evaluator_objectness.synchronize_between_processes()
            coco_evaluator_objectness.accumulate()
            coco_evaluator_objectness.summarize()

            # NOTE: WE MAKE SURE ALL LABELMAPS BELOW START AT ZERO SINCE THE
            # FUNCTION _update_classification_metrics DOES AN ARGMAX INTERNALLY
            # SO FIRST COLUMN CORRESPONDS TO ZERO'TH CLASS, WHICH CORRESPONDS TO
            # LABEL = 1 IN OUR DATASET AND MODEL
            # classification accuracy without remapping
            clkwargs = {
                'metrics_dict': classification_metrics,
                'all_labels': np.array(cltargets) - 1,
                'rlabelcodes': {
                    k - 1: v
                    for k, v in dst.rlabelcodes.items() if v != 'AMBIGUOUS'
                },
                'codemap': None,
                'prefix': 'superCateg_' if dst.use_supercategs else '',
            }
            if probabs_exist:
                clkwargs['all_scores'] = np.array(clprobabs)
            else:
                clkwargs['output_labels'] = np.array(cloutlabs)
            _update_classification_metrics(**clkwargs)

            # FIXME (low priority): this hard-codes the name of ambiguous categ
            # classification accuracy mapped to supercategs
            if not dst.use_supercategs:
                clkwargs.update({
                    'rlabelcodes': {
                        k - 1: v
                        for k, v in dst.supercategs_rlabelcodes.items()
                        if v != 'AMBIGUOUS'
                    },
                    'codemap': {
                        k - 1: v - 1
                        for k, v in dst.main_codes_to_supercategs_codes.items()
                        if dst.supercategs_rlabelcodes[v] != 'AMBIGUOUS'
                    },
                    'prefix': 'superCateg_',
                })
                _update_classification_metrics(**clkwargs)
        else:
            classification_metrics = {}

        evl_all, evl_objectness = coco_evaluator, coco_evaluator_objectness
        tsl = {}
        classification = dst.do_classification  # noqa

        evls = [(evl_all, '')]
        if classification:
            evls.append((evl_objectness, 'objectness '))

        # Detection .. all or just objectness
        for evltuple in evls:

            evl, what = evltuple

            # by bounding box
            tsl.update({
                what + 'maxDets': maxDets[-1],
                what + 'mAP @ 0.50:0.95': evl.coco_eval['bbox'].stats[0],
                what + 'AP @ 0.5': evl.coco_eval['bbox'].stats[1],  # noqa
                what + 'AP @ 0.75': evl.coco_eval['bbox'].stats[2],  # noqa
            })
            # by mask
            if 'segm' in evl.coco_eval:
                tsl.update({
                    what + 'segm mAP @ 0.50:0.95': evl.coco_eval['segm'].stats[0],
                    # noqa
                    what + 'segm AP @ 0.5': evl.coco_eval['segm'].stats[1],  # noqa
                    what + 'segm AP @ 0.75': evl.coco_eval['segm'].stats[2],  # noqa
                })

        # just classification accuracy (for those correctly detected)
        if classification:
            tsl.update(classification_metrics)

        return tsl
