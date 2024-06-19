# Copyright (c) OpenMMLab. All rights reserved.
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from mmcv.runner import ModuleList
from mmdet.models.roi_heads import HybridTaskCascadeRoIHead
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, bbox_mapping_back, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.utils.brick_wrappers import adaptive_avg_pool2d

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from pycocotools import mask as maskUtils

@HEADS.register_module()
class HybridTaskCascadeRoIHead_Cus(HybridTaskCascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead_Cus,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    @property
    def with_seg(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'seg_head') and self.seg_head is not None:
            return True
        else:
            return False

    def forward_dummy(self, img, x, proposals):
        """Dummy forward function."""
        outs = ()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, semantic_feat=semantic_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'], rois,
                                   *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=bbox_targets,
        )
        return bbox_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_targets(sampling_results, gt_masks,
                                             rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(loss_mask=loss_mask)
        return mask_results

    def _bbox_forward(self, stage, x, rois, semantic_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        """Mask head forward function for testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    @staticmethod
    def mask2RLE(mask):
        return maskUtils.encode(np.asfortranarray(mask, dtype=np.uint8))

    @torch.no_grad()
    def binary_erosion(self, binary_mask, kernel, iterations):
        for i in range(iterations):
            binary_mask = F.conv2d(binary_mask, kernel, padding=kernel.shape[-1]//2)
            binary_mask = torch.clamp(binary_mask-kernel.sum()+1, min=0, max=1)
        return binary_mask

    @torch.no_grad()
    def binary_dilate(self, binary_mask, kernel, iterations):
        for i in range(iterations):
            binary_mask = F.conv2d(binary_mask, kernel, padding=kernel.shape[-1]//2)
            binary_mask = torch.clamp(binary_mask, min=0, max=1)
        return binary_mask

    @torch.no_grad()
    def binary_open(self, binary_mask, kernel, iterations):
        binary_mask = self.binary_erosion(binary_mask, kernel, iterations)
        return self.binary_dilate(binary_mask, kernel, iterations)

    @torch.no_grad()
    def binary_close(self, binary_mask, kernel, iterations):
        binary_mask = self.binary_dilate(binary_mask, kernel, iterations)
        return self.binary_erosion(binary_mask, kernel, iterations)

    @torch.no_grad()
    def _inst_mask_to_bbox(self, inst_mask, sample_num=None, prob=1.0, device='cpu'):
        boxes = torch.zeros((inst_mask.shape[0], 5), dtype=inst_mask.dtype).to(device)
        boxes[:, 4] = prob
        x_any = torch.any(inst_mask, dim=1)
        y_any = torch.any(inst_mask, dim=2)
        for idx in range(inst_mask.shape[0]):
            x_coord = torch.where(x_any[idx, :])[0]
            y_coord = torch.where(y_any[idx, :])[0]
            if len(x_coord) > 0 and len(y_coord) > 0:
                boxes[idx, :4] = torch.tensor(
                    [x_coord[0], y_coord[0], x_coord[-1] + 1, y_coord[-1] + 1], dtype=inst_mask.dtype
                )
        if len(boxes) == 0 or sample_num is None:
            return boxes
        elif len(boxes) == 1:
            boxes = boxes[[0]*sample_num]
        else:
            boxes = boxes[torch.multinomial(torch.arange(boxes.shape[0]).to(torch.float32), sample_num, replacement=True)]
        return boxes

    @torch.no_grad()
    def _watershed_proposal(self, semantic_pred, semantic_dist=None, proposal_list=None, img_shape=None, min_area=10, thres=0, sample_num=None):
        semantic_mask = F.interpolate(semantic_pred, size=img_shape, mode='bilinear', align_corners=True)
        semantic_mask = TF.gaussian_blur(semantic_mask, kernel_size=5)
        semantic_mask[semantic_mask>thres] = 1
        semantic_mask[semantic_mask<=thres] = 0
        device = semantic_pred.device
        mask_dtype = semantic_mask.dtype

        if semantic_dist is not None:
            semantic_dist = TF.gaussian_blur(semantic_dist.unsqueeze(1).to(torch.float), kernel_size=5)
            semantic_dist = semantic_dist.squeeze(1).sigmoid().cpu().numpy()


        # fill small holes
        # noise removal
        semantic_mask = self.binary_open(semantic_mask, self.kernel, 2)
        semantic_mask = semantic_mask.squeeze(1).detach().cpu().numpy()
        watershed_proposal = []
        max_area = semantic_mask.shape[1]*semantic_mask.shape[2]/4
        for i in range(semantic_mask.shape[0]):

            #TODO torch version binary_fill_holes
            semantic_mask[i] = ndi.binary_fill_holes(semantic_mask[i])
            if semantic_dist is None:
                distance = ndi.distance_transform_edt(semantic_mask[i])
            else:
                distance = semantic_dist[i]
            # coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=semantic_mask[i])
            # coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=semantic_mask[i], num_peaks_per_label=1)
            # mask = np.zeros(distance.shape, dtype=bool)
            # mask[tuple(coords.T)] = True
            # markers, _ = ndi.label(mask)
            # inst_mask = watershed(-distance, markers, mask=semantic_mask[i])
            dist_mask = np.array(distance>0.25, dtype=bool)
            markers, _ = ndi.label(dist_mask)
            inst_mask = watershed(-distance, markers, mask=semantic_mask[i])
            inst_mask = torch.from_numpy(inst_mask).to(device)
            inst_idx = torch.unique(inst_mask)[1:]
            # remap label
            for idx, ind in enumerate(inst_idx):
                inst_mask[inst_mask==ind] = idx + 1

            # TODO the bool type may reduce the memory usage and precision
            one_hot_mask = torch.zeros((inst_mask.max()+1,)+inst_mask.shape, dtype=torch.uint8).to(device)
            one_hot_mask.scatter_(dim=0, index=inst_mask.to(torch.int64).unsqueeze(0), src=torch.ones((inst_mask.max()+1,)+inst_mask.shape, dtype=torch.uint8).to(device))
            one_hot_mask = one_hot_mask[1:]
            one_hot_mask_area = one_hot_mask.sum(dim=(1, 2))

            inst_mask = one_hot_mask[(one_hot_mask_area>min_area) & (one_hot_mask_area<max_area), :, :]
            inst_mask = inst_mask.to(mask_dtype)
            watershed_boxes = self._inst_mask_to_bbox(inst_mask, sample_num=sample_num, device=device)
            watershed_proposal.append(watershed_boxes.to(torch.float32))
            if proposal_list is not None:
                if len(watershed_boxes) != 0:
                    # if watershed_boxes is not None, add watershed_boxes to the proposal_list
                    proposal_list[i] = torch.cat((watershed_boxes, proposal_list[i]), dim=0)
                elif sample_num != None:
                    proposal_list[i] = torch.cat((proposal_list[i][:sample_num], proposal_list[i]), dim=0)
        return proposal_list, watershed_proposal

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
            if self.seg_head:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
                losses['loss_binary_seg'], losses['loss_global_cls'] = self.seg_head.loss(seg_pred, gt_masks)
            if self.with_watershed_proposal:
                if self.with_seg:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)
                else:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    semantic_feat)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        cls_weight = [None]*len(img)
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            if self.seg_head:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
                # cls_weight = seg_cls.sigmoid()
            if self.with_watershed_proposal:
                if self.with_seg:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list, img_metas[0]['img_shape'][:2], min_area=10, thres=0)
                else:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)
        else:
            semantic_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # if self.seg_head:
                        #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
                        #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # cls_score = ms_scores[-1]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label, _ = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                cls_weight=cls_weight[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append(
                        [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, imgs, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        samples_per_gpu = len(img_metas[0])
        if self.with_semantic:
            semantic_outs = [self.semantic_head(feat) for feat in img_feats]
            semantic_feats = [semantic_out[1] for semantic_out in semantic_outs]
            semantic_preds = [semantic_out[0] for semantic_out in semantic_outs]
            if self.seg_head:
                seg_outs = [self.seg_head(img, semantic_feat) for img, semantic_feat in zip(imgs, semantic_feats)]
                seg_preds = [seg_out[1] for seg_out in seg_outs]
                seg_dists = [seg_out[2] for seg_out in seg_outs]
            if self.with_watershed_proposal:
                if self.with_seg:
                    watershed_proposals_list = [self._watershed_proposal(seg_pred, semantic_dist=seg_dist, img_shape=img_metas[0][0]['img_shape'][:2])[1] for seg_pred, seg_dist in zip(seg_preds, seg_dists)]
                else:
                    watershed_proposals_list = [self._watershed_proposal(semantic_pred, img_shape=img_metas[0][0]['img_shape'][:2])[1] for semantic_pred in semantic_preds]
            else:
                watershed_proposals_list = [[] for _ in imgs]
            #TODO be careful about the sample per gpu
            for i in range(samples_per_gpu):
                recovered_proposals = []
                for proposals, img_infos in zip(watershed_proposals_list, img_metas):
                    img_info = img_infos[i]
                    img_shape = img_info['img_shape']
                    scale_factor = img_info['scale_factor']
                    flip = img_info['flip']
                    flip_direction = img_info['flip_direction']
                    _proposals = proposals[i].clone()
                    _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                                          scale_factor, flip,
                                                          flip_direction)
                    recovered_proposals.append(_proposals)
                watershed_proposals = torch.cat(recovered_proposals, dim=0)
                proposal_list[i] = torch.cat((watershed_proposals, proposal_list[i]), dim=0)
            # merge_aug_proposals(watershed_proposals, img_metas, cfg.test)
        else:
            semantic_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic, watershed_proposals in zip(img_feats, img_metas, semantic_feats, watershed_proposals_list):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)

            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

@HEADS.register_module()
class HybridTaskCascadeRoIHeadWithoutSemantic(HybridTaskCascadeRoIHead_Cus):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHeadWithoutSemantic,
              self).__init__(num_stages,
                             stage_loss_weights,
                             semantic_roi_extractor=None,
                             semantic_head=None,
                             seg_head=seg_head,
                             semantic_fusion=None,
                             interleaved=interleaved,
                             mask_info_flow=mask_info_flow,
                             watershed_proposal_num=watershed_proposal_num,
                             watershed_proposal=watershed_proposal
                             **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()

        if self.seg_head:
            seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, x[0])
            losses['loss_binary_seg'], losses['loss_global_cls'] = self.seg_head.loss(seg_pred, seg_dist, seg_cls, gt_masks, gt_labels)
            if self.with_watershed_proposal:
                proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        semantic_feat = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    semantic_feat)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """

        if self.seg_head:
            seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, x[0])
            if self.with_watershed_proposal:
                proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list, img_metas[0]['img_shape'][:2], min_area=10, thres=0)

        semantic_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label, _ = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append(
                        [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, imgs, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        samples_per_gpu = len(img_metas[0])

        if self.with_watershed_proposal and self.seg_head:
            seg_outs = [self.seg_head(img, img_feat[0]) for img, img_feat in zip(imgs, img_feats)]
            seg_preds = [seg_out[1] for seg_out in seg_outs]
            seg_dists = [seg_out[2] for seg_out in seg_outs]
            watershed_proposals_list = [self._watershed_proposal(seg_pred, semantic_dist=seg_dist, img_shape=img_metas[0][0]['img_shape'][:2])[1] for seg_pred, seg_dist in zip(seg_preds, seg_dists)]
        else:
            watershed_proposals_list = [[] for _ in imgs]

        #TODO be careful about the sample per gpu
        for i in range(samples_per_gpu):
            recovered_proposals = []
            for proposals, img_infos in zip(watershed_proposals_list, img_metas):
                img_info = img_infos[i]
                img_shape = img_info['img_shape']
                scale_factor = img_info['scale_factor']
                flip = img_info['flip']
                flip_direction = img_info['flip_direction']
                _proposals = proposals[i].clone()
                _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                                      scale_factor, flip,
                                                      flip_direction)
                recovered_proposals.append(_proposals)
            watershed_proposals = torch.cat(recovered_proposals, dim=0)
            proposal_list[i] = torch.cat((watershed_proposals, proposal_list[i]), dim=0)
        # merge_aug_proposals(watershed_proposals, img_metas, cfg.test)

        semantic_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic, watershed_proposals in zip(img_feats, img_metas, semantic_feats, watershed_proposals_list):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)

            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

@HEADS.register_module()
class HybridTaskCascadeRoIHead_Partial(HybridTaskCascadeRoIHead_Cus):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead_Partial,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        if self.seg_head:
            if semantic_feat:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
            else:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, x[0])
            losses['loss_binary_seg'], losses['loss_global_cls'] = self.seg_head.loss_partial(seg_pred, seg_dist, seg_cls, gt_masks, gt_labels, img_metas)
        if self.with_watershed_proposal:
            if self.with_seg:
                proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)
            elif self.with_semantic:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assert len(gt_masks[j]) == len(img_metas[j]['ismask'])
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j][img_metas[j]['ismask']==1],
                                gt_bboxes_ignore[j], gt_labels[j][img_metas[j]['ismask']==1])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j][img_metas[j]['ismask']==1],
                                gt_labels[j][img_metas[j]['ismask']==1],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    semantic_feat)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        cls_weight = [None]*len(img)
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        if self.seg_head:
            if semantic_feat:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
            else:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, x[0])
            # cls_weight = seg_cls.sigmoid()
        if self.with_watershed_proposal:
            if self.with_seg:
                proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list, img_metas[0]['img_shape'][:2], min_area=10, thres=0)
            elif self.with_semantic:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # if self.seg_head:
                        #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
                        #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # cls_score = ms_scores[-1]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_scores = []
        for i in range(num_imgs):
            det_bbox, det_label, det_score = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                cls_weight=cls_weight[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_scores.append(det_score.cpu().numpy())
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append(
                        [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], det_scores))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, imgs, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        samples_per_gpu = len(img_metas[0])
        if self.with_semantic:
            semantic_outs = [self.semantic_head(feat) for feat in img_feats]
            semantic_feats = [semantic_out[1] for semantic_out in semantic_outs]
            semantic_preds = [semantic_out[0] for semantic_out in semantic_outs]
            # merge_aug_proposals(watershed_proposals, img_metas, cfg.test)
        else:
            semantic_feats = [None] * len(img_metas)

        if self.seg_head:
            if semantic_feats[0]:
                seg_outs = [self.seg_head(img, semantic_feat) for img, semantic_feat in zip(imgs, semantic_feats)]
            else:
                seg_outs = [self.seg_head(img, semantic_feat[0]) for img, semantic_feat in zip(imgs, img_feats)]
            seg_preds = [seg_out[1] for seg_out in seg_outs]
            seg_dists = [seg_out[2] for seg_out in seg_outs]
        if self.with_watershed_proposal:
            if self.with_seg:
                watershed_proposals_list = [self._watershed_proposal(seg_pred, semantic_dist=seg_dist, img_shape=img_metas[0][0]['img_shape'][:2])[1] for seg_pred, seg_dist in zip(seg_preds, seg_dists)]
            elif self.with_semantic:
                watershed_proposals_list = [self._watershed_proposal(semantic_pred, img_shape=img_metas[0][0]['img_shape'][:2])[1] for semantic_pred in semantic_preds]
        else:
            watershed_proposals_list = [[] for _ in imgs]
        if self.with_semantic or self.with_seg:
            #TODO be careful about the sample per gpu
            for i in range(samples_per_gpu):
                recovered_proposals = []
                for proposals, img_infos in zip(watershed_proposals_list, img_metas):
                    img_info = img_infos[i]
                    img_shape = img_info['img_shape']
                    scale_factor = img_info['scale_factor']
                    flip = img_info['flip']
                    flip_direction = img_info['flip_direction']
                    _proposals = proposals[i].clone()
                    _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                                          scale_factor, flip,
                                                          flip_direction)
                    recovered_proposals.append(_proposals)
                watershed_proposals = torch.cat(recovered_proposals, dim=0)
                proposal_list[i] = torch.cat((watershed_proposals, proposal_list[i]), dim=0)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(img_feats, img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)

            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

@HEADS.register_module()
class HybridTaskCascadeRoIHead_Lite(HybridTaskCascadeRoIHead_Cus):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead_Lite,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head]

        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [mask_roi_extractor]
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def forward_dummy(self, img, x, proposals):
        """Dummy forward function."""
        outs = ()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None

            for i in range(len(self.mask_head)):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        mask_roi_extractor = self.mask_roi_extractor[0]
        mask_head = self.mask_head[0]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_targets(sampling_results, gt_masks,
                                             rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(loss_mask=loss_mask)
        return mask_results

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        """Mask head forward function for testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg!=0)
            losses['loss_semantic_seg'] = loss_seg
            if self.seg_head:
                seg_feat, seg_pred = self.seg_head(img, semantic_feat)
                losses['loss_binary_seg'] = self.seg_head.loss(seg_pred, gt_semantic_seg)
            if self.with_watershed_proposal:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                if i == self.num_stages - 1:
                    mask_results = self._mask_forward_train(
                        i, x, sampling_results, gt_masks, rcnn_train_cfg,
                        semantic_feat)
                    for name, value in mask_results['loss_mask'].items():
                        losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        cls_weight = [None]*len(img)
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            # #TODO remove
            # for i in range(len(img_metas)):
            #     img_name =img_metas[i]['ori_filename'].split('.')[0]
            #     np.save(f'/home/bao/code/NuHTC/results/res/{img_name}.npy', semantic_pred[i].cpu().detach().numpy())
            if self.seg_head:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
                # cls_weight = seg_cls.sigmoid()
            if self.with_watershed_proposal:
                if self.with_seg:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list, img_metas[0]['img_shape'][:2], min_area=10, thres=0)
                else:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)
        else:
            semantic_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # if self.seg_head:
                        #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
                        #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # cls_score = ms_scores[-1]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label, _ = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                cls_weight=cls_weight[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)

                mask_head = self.mask_head[0]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)

                # split batch mask prediction back to each image
                mask_pred = mask_pred.split(num_bbox_per_img, 0)
                aug_masks.append(
                    [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, imgs, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        samples_per_gpu = len(img_metas[0])
        if self.with_semantic:
            semantic_outs = [self.semantic_head(feat) for feat in img_feats]
            semantic_feats = [semantic_out[1] for semantic_out in semantic_outs]
            semantic_preds = [semantic_out[0] for semantic_out in semantic_outs]
            if self.seg_head:
                seg_outs = [self.seg_head(img, semantic_feat) for img, semantic_feat in zip(imgs, semantic_feats)]
                seg_preds = [seg_out[1] for seg_out in seg_outs]
                seg_dists = [seg_out[2] for seg_out in seg_outs]
            if self.with_watershed_proposal:
                if self.with_seg:
                    watershed_proposals_list = [self._watershed_proposal(seg_pred, semantic_dist=seg_dist, img_shape=img_metas[0][0]['img_shape'][:2])[1] for seg_pred, seg_dist in zip(seg_preds, seg_dists)]
                else:
                    watershed_proposals_list = [self._watershed_proposal(semantic_pred, img_shape=img_metas[0][0]['img_shape'][:2])[1] for semantic_pred in semantic_preds]
            else:
                watershed_proposals_list = [[] for _ in imgs]
            #TODO be careful about the sample per gpu
            for i in range(samples_per_gpu):
                recovered_proposals = []
                for proposals, img_infos in zip(watershed_proposals_list, img_metas):
                    img_info = img_infos[i]
                    img_shape = img_info['img_shape']
                    scale_factor = img_info['scale_factor']
                    flip = img_info['flip']
                    flip_direction = img_info['flip_direction']
                    if len(proposals) == 0:
                        continue
                    _proposals = proposals[i].clone()
                    _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                                          scale_factor, flip,
                                                          flip_direction)
                    recovered_proposals.append(_proposals)
                if len(recovered_proposals) == 0:
                    continue
                watershed_proposals = torch.cat(recovered_proposals, dim=0)
                proposal_list[i] = torch.cat((watershed_proposals, proposal_list[i]), dim=0)
            # merge_aug_proposals(watershed_proposals, img_metas, cfg.test)
        else:
            semantic_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic, watershed_proposals in zip(img_feats, img_metas, semantic_feats, watershed_proposals_list):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)

            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None

                    mask_head = self.mask_head[0]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(
                            mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                    aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

@HEADS.register_module()
class HybridTaskCascadeRoIHead_Lite_Partial(HybridTaskCascadeRoIHead_Lite):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead_Lite_Partial,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
            if self.seg_head:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
                losses['loss_binary_seg'], losses['loss_global_cls'] = self.seg_head.loss(seg_pred, seg_dist, seg_cls, gt_masks, gt_labels)
            if self.with_watershed_proposal:
                if self.with_seg:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)
                else:
                    proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assert len(gt_masks[j]) == len(img_metas[j]['ismask'])
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j][img_metas[j]['ismask']==1],
                                gt_bboxes_ignore[j], gt_labels[j][img_metas[j]['ismask']==1])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j][img_metas[j]['ismask']==1],
                                gt_labels[j][img_metas[j]['ismask']==1],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                if i == self.num_stages - 1:
                    pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                    if len(pos_rois) != 0:
                        mask_results = self._mask_forward_train(
                            i, x, sampling_results, gt_masks, rcnn_train_cfg,
                            semantic_feat)
                        for name, value in mask_results['loss_mask'].items():
                            losses[f's{i}.{name}'] = (
                                value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        cls_weight = [None]*len(img)
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        if self.seg_head:
            if semantic_feat:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, semantic_feat)
            else:
                seg_feat, seg_pred, seg_dist, seg_cls = self.seg_head(img, x[0])
            # cls_weight = seg_cls.sigmoid()
        if self.with_watershed_proposal:
            if self.with_seg:
                proposal_list, watershed_proposal_list = self._watershed_proposal(seg_pred, seg_dist, proposal_list, img_metas[0]['img_shape'][:2], min_area=10, thres=0)
            elif self.with_semantic:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # if self.seg_head:
                        #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
                        #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # cls_score = ms_scores[-1]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_scores = []
        for i in range(num_imgs):
            det_bbox, det_label, det_score = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                cls_weight=cls_weight[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_scores.append(det_score.cpu().numpy())
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)

                mask_head = self.mask_head[0]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)

                # split batch mask prediction back to each image
                mask_pred = mask_pred.split(num_bbox_per_img, 0)
                aug_masks.append(
                    [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], det_scores))
        else:
            results = ms_bbox_result['ensemble']

        return results


@HEADS.register_module()
class HybridTaskCascadeRoIHead_Lite_Fuse(HybridTaskCascadeRoIHead_Lite):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 seg_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 watershed_proposal_num=500,
                 watershed_proposal=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead_Lite_Fuse,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.watershed_proposal_num = watershed_proposal_num
        self.with_watershed_proposal = watershed_proposal
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            # loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg!=0)
            #TODO major revision
            gt_semantic_seg_down = (gt_semantic_seg!=0).to(torch.uint8)
            gt_semantic_seg_down = F.interpolate(gt_semantic_seg_down, size=(128, 128), mode='nearest')
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg_down)
            losses['loss_semantic_seg'] = loss_seg
            if self.seg_head:
                seg_feat, seg_pred = self.seg_head(img, semantic_feat)
                losses['loss_binary_seg'] = self.seg_head.loss(seg_pred, gt_semantic_seg)
            if self.with_watershed_proposal:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0, sample_num=self.watershed_proposal_num)

        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                    semantic_feat=semantic_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []

                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x],
                                semantic_feat=semantic_feat)
                            sampling_results.append(sampling_result)

                if i == self.num_stages - 1:
                    mask_results = self._mask_forward_train(
                        i, x, sampling_results, gt_masks, rcnn_train_cfg,
                        semantic_feat)
                    for name, value in mask_results['loss_mask'].items():
                        losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        cls_weight = [None]*len(img)
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)

            if self.seg_head:
                seg_feat, seg_pred = self.seg_head(img, semantic_feat)
                # cls_weight = seg_cls.sigmoid()
            if self.with_watershed_proposal:
                proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)
        else:
            semantic_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # if self.seg_head:
                        #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
                        #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # cls_score = ms_scores[-1]
        watershed_bbox_result_li, watershed_segm_result_li = [], []
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label, _ = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                cls_weight=cls_weight[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            watershed_bbox_result, watershed_segm_result = self._watershed_segmentation(seg_pred[[i]], ori_shape=ori_shapes[i][:2])
            watershed_bbox_result_li.append(watershed_bbox_result)
            watershed_segm_result_li.append(watershed_segm_result)

        bbox_result = []
        for i in range(num_imgs):
            bbox_result.append(bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes))
            for cls_idx in range(self.mask_head[-1].num_classes):
                bbox_result[i][cls_idx] = np.concatenate((bbox_result[i][cls_idx], watershed_bbox_result_li[i][cls_idx]), axis=0)
        # ms_bbox_result['ensemble'] = bbox_result
        ms_bbox_result['ensemble'] = watershed_bbox_result_li

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[watershed_segm_result_li[img_idx][cls_idx] for cls_idx in range(mask_classes)]
                                for img_idx in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)

                mask_head = self.mask_head[0]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)

                # split batch mask prediction back to each image
                mask_pred = mask_pred.split(num_bbox_per_img, 0)
                aug_masks.append(
                    [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [watershed_segm_result_li[i][cls_idx]
                             for cls_idx in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)

                        for cls_idx in range(self.mask_head[-1].num_classes):
                            segm_result[cls_idx] = segm_result[cls_idx] + watershed_segm_result_li[i][cls_idx]
                        segm_results.append(segm_result)
            # ms_segm_result['ensemble'] = segm_results
            ms_segm_result['ensemble'] = watershed_segm_result_li

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    # def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
    #
    #     if self.with_semantic:
    #         semantic_pred, semantic_feat = self.semantic_head(x)
    #
    #         if self.seg_head:
    #             seg_feat, seg_pred = self.seg_head(img, semantic_feat)
    #             # cls_weight = seg_cls.sigmoid()
    #         if self.with_watershed_proposal:
    #             proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)
    #     else:
    #         semantic_feat = None
    #
    #     num_imgs = len(proposal_list)
    #     ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
    #
    #     # "ms" in variable names means multi-stage
    #     ms_bbox_result = {}
    #     ms_segm_result = {}
    #
    #     watershed_bbox_result_li, watershed_segm_result_li = [], []
    #     # apply bbox post-processing to each image individually
    #     for i in range(num_imgs):
    #         watershed_bbox_result, watershed_segm_result = self._watershed_segmentation(seg_pred[[i]], ori_shape=ori_shapes[i][:2])
    #         watershed_bbox_result_li.append(watershed_bbox_result)
    #         watershed_segm_result_li.append(watershed_segm_result)
    #     # ms_bbox_result['ensemble'] = bbox_result
    #     ms_bbox_result['ensemble'] = watershed_bbox_result_li
    #
    #     if self.with_mask:
    #         # ms_segm_result['ensemble'] = segm_results
    #         ms_segm_result['ensemble'] = watershed_segm_result_li
    #
    #     if self.with_mask:
    #         results = list(
    #             zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
    #     else:
    #         results = ms_bbox_result['ensemble']
    #
    #     return results


    # def simple_test(self, img, x, proposal_list, img_metas, rescale=False):
    #     """Test without augmentation.
    #
    #     Args:
    #         x (tuple[Tensor]): Features from upstream network. Each
    #             has shape (batch_size, c, h, w).
    #         proposal_list (list(Tensor)): Proposals from rpn head.
    #             Each has shape (num_proposals, 5), last dimension
    #             5 represent (x1, y1, x2, y2, score).
    #         img_metas (list[dict]): Meta information of images.
    #         rescale (bool): Whether to rescale the results to
    #             the original image. Default: True.
    #
    #     Returns:
    #         list[list[np.ndarray]] or list[tuple]: When no mask branch,
    #         it is bbox results of each image and classes with type
    #         `list[list[np.ndarray]]`. The outer list
    #         corresponds to each image. The inner list
    #         corresponds to each class. When the model has mask branch,
    #         it contains bbox results and mask results.
    #         The outer list corresponds to each image, and first element
    #         of tuple is bbox results, second element is mask results.
    #     """
    #     cls_weight = [None]*len(img)
    #     if self.with_semantic:
    #         semantic_pred, semantic_feat = self.semantic_head(x)
    #         proposal_list, watershed_proposal_list = self._watershed_proposal(semantic_pred, proposal_list=proposal_list, img_shape=img_metas[0]['img_shape'][:2], min_area=10, thres=0)
    #     else:
    #         semantic_feat = None
    #
    #     num_imgs = len(proposal_list)
    #     img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    #     ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
    #     scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
    #
    #     # "ms" in variable names means multi-stage
    #     ms_bbox_result = {}
    #     ms_segm_result = {}
    #     ms_scores = []
    #     rcnn_test_cfg = self.test_cfg
    #
    #     rois = bbox2roi(proposal_list)
    #
    #     if rois.shape[0] == 0:
    #         # There is no proposal in the whole batch
    #         bbox_results = [[
    #             np.zeros((0, 5), dtype=np.float32)
    #             for _ in range(self.bbox_head[-1].num_classes)
    #         ]] * num_imgs
    #
    #         if self.with_mask:
    #             mask_classes = self.mask_head[-1].num_classes
    #             segm_results = [[[] for _ in range(mask_classes)]
    #                             for _ in range(num_imgs)]
    #             results = list(zip(bbox_results, segm_results))
    #         else:
    #             results = bbox_results
    #
    #         return results
    #
    #     for i in range(self.num_stages):
    #         bbox_head = self.bbox_head[i]
    #         bbox_results = self._bbox_forward(
    #             i, x, rois, semantic_feat=semantic_feat)
    #         # split batch bbox prediction back to each image
    #         cls_score = bbox_results['cls_score']
    #         bbox_pred = bbox_results['bbox_pred']
    #         num_proposals_per_img = tuple(len(p) for p in proposal_list)
    #         rois = rois.split(num_proposals_per_img, 0)
    #         cls_score = cls_score.split(num_proposals_per_img, 0)
    #
    #         bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    #         ms_scores.append(cls_score)
    #
    #         if i < self.num_stages - 1:
    #             refine_rois_list = []
    #             for j in range(num_imgs):
    #                 if rois[j].shape[0] > 0:
    #                     # if self.seg_head:
    #                     #     cls_score[j][:, :-1] = cls_score[j][:, :-1].sigmoid()*seg_score[i].unsqueeze(0)
    #                     #     cls_score[j][:, :-1] = torch.log(cls_score[j][:, :-1]/(1-cls_score[j][:, :-1]))
    #                     bbox_label = cls_score[j][:, :-1].argmax(dim=1)
    #                     refine_rois = bbox_head.regress_by_class(
    #                         rois[j], bbox_label, bbox_pred[j], img_metas[j])
    #                     refine_rois_list.append(refine_rois)
    #             rois = torch.cat(refine_rois_list)
    #
    #     # average scores of each image by stages
    #     cls_score = [
    #         sum([score[i] for score in ms_scores]) / float(len(ms_scores))
    #         for i in range(num_imgs)
    #     ]
    #     # cls_score = ms_scores[-1]
    #
    #     # apply bbox post-processing to each image individually
    #     det_bboxes = []
    #     det_labels = []
    #     for i in range(num_imgs):
    #         det_bbox, det_label, _ = self.bbox_head[-1].get_bboxes(
    #             rois[i],
    #             cls_score[i],
    #             bbox_pred[i],
    #             img_shapes[i],
    #             scale_factors[i],
    #             cls_weight=cls_weight[i],
    #             rescale=rescale,
    #             cfg=rcnn_test_cfg)
    #         det_bboxes.append(det_bbox)
    #         det_labels.append(det_label)
    #     bbox_result = [
    #         bbox2result(det_bboxes[i], det_labels[i],
    #                     self.bbox_head[-1].num_classes)
    #         for i in range(num_imgs)
    #     ]
    #     ms_bbox_result['ensemble'] = bbox_result
    #
    #     if self.with_mask:
    #         if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
    #             mask_classes = self.mask_head[-1].num_classes
    #             segm_results = [[[] for _ in range(mask_classes)]
    #                             for _ in range(num_imgs)]
    #         else:
    #             if rescale and not isinstance(scale_factors[0], float):
    #                 scale_factors = [
    #                     torch.from_numpy(scale_factor).to(det_bboxes[0].device)
    #                     for scale_factor in scale_factors
    #                 ]
    #             _bboxes = [
    #                 det_bboxes[i][:, :4] *
    #                 scale_factors[i] if rescale else det_bboxes[i]
    #                 for i in range(num_imgs)
    #             ]
    #             mask_rois = bbox2roi(_bboxes)
    #             aug_masks = []
    #             mask_roi_extractor = self.mask_roi_extractor[-1]
    #             mask_feats = mask_roi_extractor(
    #                 x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
    #             if self.with_semantic and 'mask' in self.semantic_fusion:
    #                 mask_semantic_feat = self.semantic_roi_extractor(
    #                     [semantic_feat], mask_rois)
    #                 mask_feats += mask_semantic_feat
    #             last_feat = None
    #
    #             num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
    #
    #             mask_head = self.mask_head[0]
    #             if self.mask_info_flow:
    #                 mask_pred, last_feat = mask_head(mask_feats, last_feat)
    #             else:
    #                 mask_pred = mask_head(mask_feats)
    #
    #             # split batch mask prediction back to each image
    #             mask_pred = mask_pred.split(num_bbox_per_img, 0)
    #             aug_masks.append(
    #                 [mask.sigmoid().cpu().numpy() for mask in mask_pred])
    #
    #             # apply mask post-processing to each image individually
    #             segm_results = []
    #             for i in range(num_imgs):
    #                 if det_bboxes[i].shape[0] == 0:
    #                     segm_results.append(
    #                         [[]
    #                          for _ in range(self.mask_head[-1].num_classes)])
    #                 else:
    #                     aug_mask = [mask[i] for mask in aug_masks]
    #                     merged_mask = merge_aug_masks(
    #                         aug_mask, [[img_metas[i]]] * self.num_stages,
    #                         rcnn_test_cfg)
    #                     segm_result = self.mask_head[-1].get_seg_masks(
    #                         merged_mask, _bboxes[i], det_labels[i],
    #                         rcnn_test_cfg, ori_shapes[i], scale_factors[i],
    #                         rescale)
    #                     segm_results.append(segm_result)
    #         ms_segm_result['ensemble'] = segm_results
    #
    #     if self.with_mask:
    #         results = list(
    #             zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
    #     else:
    #         results = ms_bbox_result['ensemble']
    #
    #     return results

    @torch.no_grad()
    def _watershed_segmentation(self, semantic_pred, semantic_dist=None, thres=0, min_area=10, ori_shape=(256, 256)):
        num_classes = self.seg_head.num_classes
        seg_bbox_li = [[] for cls_idx in range(num_classes)]
        seg_mask_li = [[] for cls_idx in range(num_classes)]
        semantic_pred = F.interpolate(semantic_pred, size=ori_shape, mode='bilinear', align_corners=True)
        for cls_idx in range(num_classes):
            semantic_mask = TF.gaussian_blur(semantic_pred[:, [cls_idx], :, :], kernel_size=5)
            semantic_mask[semantic_mask>thres] = 1
            semantic_mask[semantic_mask<=thres] = 0
            device = semantic_pred.device
            mask_dtype = semantic_mask.dtype

            # fill small holes
            # noise removal
            semantic_mask = self.binary_open(semantic_mask, self.kernel, 2)
            semantic_mask = semantic_mask.squeeze(1).detach().cpu().numpy()
            watershed_proposal = []
            max_area = semantic_mask.shape[1]*semantic_mask.shape[2]/4
            i = 0
            #TODO torch version binary_fill_holes
            semantic_mask[i] = ndi.binary_fill_holes(semantic_mask[i])
            if semantic_dist is None:
                distance = ndi.distance_transform_edt(semantic_mask[i])
            else:
                distance = semantic_dist[i]

            dist_mask = np.array(distance>0.25, dtype=bool)
            markers, _ = ndi.label(dist_mask)
            inst_mask = watershed(-distance, markers, mask=semantic_mask[i])
            inst_idx = np.unique(inst_mask)[1:]
            # remap label
            for ind in inst_idx:
                nuclei_area = np.sum(inst_mask==ind)
                if nuclei_area > min_area and nuclei_area < max_area:
                    seg_mask_li[cls_idx].append(inst_mask==ind)
                    y_axis, x_axis = np.where(inst_mask==ind)
                    bbox = np.array([x_axis.min(), y_axis.min(), x_axis.max()+1, y_axis.max()+1, 0.36], dtype=float)
                    seg_bbox_li[cls_idx].append(bbox)
            seg_bbox_li[cls_idx] = np.array(seg_bbox_li[cls_idx]).reshape(-1, 5)
        return seg_bbox_li, seg_mask_li
