# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.roi_heads.mask_heads import HTCMaskHead
from torch import nn
import torch.nn.functional as F
import einops

@DETECTORS.register_module()
class HTCMaskHead_Cus(HTCMaskHead):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(HTCMaskHead, self).__init__(
            backbone,
            rpn_head,
            roi_head,
            train_cfg,
            test_cfg,
            neck=neck,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.conv_last = nn.Conv2d(4+3, 1, kernel_size=3, padding=1)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        x_up = einops.rearrange(x[0], 'b (p1 p2 c) h w-> b c (h p1) (w p2)', p1=4, p2=4)
        x_last = self.conv_last(torch.cat((img, x_up), dim=1)).squeeze(1)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        mid_losses = self.loss_first_seg(x_last, gt_masks)
        losses.update(roi_losses)
        losses.update(mid_losses)

        return losses

    def loss_first_seg(self, x, gt_masks):
        losses = dict()
        # TODO further should set the mode option
        #  x = x[0].mean(dim=1).sigmoid()
        # x = x[0].max(dim=1)[0].sigmoid()
        # x = einops.rearrange(x[0], 'b (p1 p2 c) h w-> b c (h p1) (w p2)', p1=4, p2=4).max(dim=1)[0]
        # batch*128*128
        binary_gt_mask = []
        for i in range(len(gt_masks)):
            gt_mask = gt_masks[i]
            if gt_mask.masks.shape[0] != 0:
                binary_gt_mask.append(torch.max(torch.tensor(gt_mask.masks), dim=0)[0])
            else:
                binary_gt_mask.append(torch.zeros(gt_mask.masks.shape[1:], dtype=torch.uint8))
        gt_masks = torch.stack(binary_gt_mask, dim=0)
        # gt_masks = F.interpolate(gt_masks.unsqueeze(1), size=128, mode='nearest').squeeze(1)

        loss = F.binary_cross_entropy_with_logits(x, gt_masks.to(torch.float16).to(x.device))
        losses['loss_mid_seg'] = loss

        return  losses