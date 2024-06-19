# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from six.moves import map
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.core import multi_apply
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss
from scipy import ndimage as ndi
import skimage

@HEADS.register_module()
class HTCSegHead(BaseModule):
    r"""Multi-level fused semantic segmentation head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        x -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*2)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 num_convs=2,
                 num_classes=2,
                 in_channels=256,
                 conv_out_channels=256,
                 out_size=512,
                 conv_cfg=None,
                 norm_cfg=None,
                 ignore_label=None,
                 loss_weight=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=0.2),
                 loss_dist=dict(
                     type='L1Loss',
                     loss_weight=0.2),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.2),
                 init_cfg=dict(
                     type='Kaiming', override=dict(name='convs'))):
        super(HTCSegHead, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.num_convs = num_convs
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        if not (isinstance(out_size, list) or isinstance(out_size, tuple)):
            self.out_size = (out_size, out_size)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.num_ins if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_embedding = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = conv_out_channels*2 if i == 0 else conv_out_channels
            self.conv_embedding.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_seg = nn.Conv2d(conv_out_channels, 1, 1)
        self.conv_dist = nn.Conv2d(conv_out_channels, 1, 1)
        self.conv_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(conv_out_channels, conv_out_channels),
            nn.ReLU(),
            nn.Linear(conv_out_channels, self.num_classes)
        )
        if ignore_label:
            loss_seg['ignore_index'] = ignore_label
        if loss_weight:
            loss_seg['loss_weight'] = loss_weight
        if ignore_label or loss_weight:
            warnings.warn('``ignore_label`` and ``loss_weight`` would be '
                          'deprecated soon. Please set ``ingore_index`` and '
                          '``loss_weight`` in ``loss_seg`` instead.')
        self.seg_criterion = build_loss(loss_seg)
        self.dist_criterion = build_loss(loss_dist)
        self.cls_criterion = build_loss(loss_cls)
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    @torch.no_grad()
    def binary_erosion(self, binary_mask, kernel, iterations):
        for i in range(iterations):
            binary_mask = F.conv2d(binary_mask, kernel, padding=kernel.shape[-1]//2)
            binary_mask = torch.clamp(binary_mask-kernel.sum()+1, min=0, max=1)
        return binary_mask

    @torch.no_grad()
    def dist_trans(self, loc):
        b_loc, f_loc = loc[0].to(torch.int32), loc[1].to(torch.int32)
        if b_loc.shape[0] == 0 or f_loc.shape[0] == 0:
            return None
        tmp_dist = torch.pow(b_loc.reshape((1, -1, 2))-f_loc.reshape((-1, 1, 2)), 2).sum(dim=-1)
        tmp_dist = torch.sqrt(tmp_dist).min(dim=1)[0]
        tmp_dist = tmp_dist/tmp_dist.max()
        return tmp_dist

    @auto_fp16()
    def forward(self, img, semantic_feat):
        x = img
        for i in range(self.num_convs):
            x = self.convs[i](x)
        semantic_feat = F.interpolate(semantic_feat, size=img.shape[2:], mode='bilinear', align_corners=True).squeeze(1)
        fused_feat = torch.cat((x, semantic_feat), dim=1)
        for i in range(self.num_convs):
            fused_feat = self.conv_embedding[i](fused_feat)

        seg_pred = self.conv_seg(fused_feat)
        seg_dist = self.conv_dist(fused_feat)
        seg_cls = self.conv_cls(fused_feat)

        #TODO point detection
        return fused_feat, seg_pred.squeeze(1), seg_dist.squeeze(1), seg_cls

    @force_fp32(apply_to=('seg_pred', 'seg_dist', 'seg_cls'))
    def loss(self, seg_pred, seg_dist, seg_cls, gt_masks, gt_labels):
        # seg_pred = seg_pred.sigmoid() Dice loss will use the sigmoid func
        #TODO more elegant loss computation
        gt_num = len(gt_masks)
        seg_dist = seg_dist.sigmoid()
        gt_seg = torch.zeros(seg_pred.shape, dtype=seg_pred.dtype, device=seg_pred.device)
        gt_dist = torch.zeros(seg_pred.shape, dtype=seg_pred.dtype, device=seg_pred.device)
        gt_cls = torch.zeros((gt_num, self.num_classes), dtype=gt_labels[0].dtype, device=gt_labels[0].device)

        for idx in range(gt_num):
            gt_mask_idx = gt_masks[idx].masks
            gt_mask_num = gt_mask_idx.shape[0]
            gt_cls[idx, gt_labels[idx].unique()] = 1
            if gt_mask_num != 0:
                gt_seg[idx] = torch.from_numpy(gt_mask_idx.max(axis=0)).to(seg_pred.device)
                gt_mask_idx = torch.from_numpy(gt_mask_idx[gt_mask_idx.sum(axis=(1, 2))>20]).to(self.kernel.device)
                if gt_mask_idx.shape[0] != 0:
                    gt_mask_idx_erosion = self.binary_erosion(gt_mask_idx.unsqueeze(1).to(self.kernel.dtype), self.kernel, 1).squeeze(1)
                    # gt_dist[idx] = gt_mask_idx_erosion.max(dim=0)[0]
                    b_idx, b_y, b_x = torch.where(torch.logical_xor(gt_mask_idx, gt_mask_idx_erosion)==True)
                    b_loc = [torch.stack((b_y[b_idx==i], b_x[b_idx==i]), dim=1) for i in range(gt_mask_num)]
                    f_idx, f_y, f_x = torch.where(gt_mask_idx_erosion==1)
                    f_loc = [torch.stack((f_y[f_idx==i], f_x[f_idx==i]), dim=1) for i in range(gt_mask_num)]
                    loc_li = [(b_loc[i], f_loc[i]) for i in range(gt_mask_num)]
                    tmp_dist_li = list(map(self.dist_trans, loc_li))
                    for i in range(gt_mask_num):
                        if tmp_dist_li[i] is None:
                            continue
                        gt_dist[idx][f_y[f_idx==i], f_x[f_idx==i]] = tmp_dist_li[i]
                    del loc_li
                    del tmp_dist_li

                    # #TODO distance_transform_edt does not support batch
                    # inst_dist = ndi.distance_transform_edt(gt_mask_idx)
                    # if inst_dist.max() != 0:
                    #     inst_dist = torch.from_numpy(inst_dist/inst_dist.max()).to(seg_dist.dtype).to(seg_dist.device)
                    #     gt_dist[idx] += inst_dist

        loss_semantic_seg = self.seg_criterion(seg_pred, gt_seg)+self.dist_criterion(seg_dist, gt_dist)
        loss_global_cls = self.cls_criterion(seg_cls, gt_cls)
        #TODO acc cal
        return loss_semantic_seg, loss_global_cls

    @force_fp32(apply_to=('seg_pred', 'seg_dist', 'seg_cls'))
    def loss_partial(self, seg_pred, seg_dist, seg_cls, gt_masks, gt_labels, img_metas):
        # seg_pred = seg_pred.sigmoid() Dice loss will use the sigmoid func
        #TODO more elegant loss computation
        gt_num = len(gt_masks)
        seg_dist = seg_dist.sigmoid()
        gt_seg = torch.zeros(seg_pred.shape, dtype=seg_pred.dtype, device=seg_pred.device)
        gt_dist = torch.zeros(seg_pred.shape, dtype=seg_pred.dtype, device=seg_pred.device)
        gt_cls = torch.zeros((gt_num, self.num_classes), dtype=gt_labels[0].dtype, device=gt_labels[0].device)
        gt_pos_mask = torch.ones(seg_pred.shape, dtype=seg_pred.dtype, device=seg_pred.device)
        for idx in range(gt_num):
            gt_mask_idx = gt_masks[idx].masks
            gt_mask_valid = img_metas[idx]['ismask']
            gt_mask_non = gt_mask_idx[gt_mask_valid==0]
            gt_mask_idx = gt_mask_idx[gt_mask_valid==1]
            gt_mask_num = gt_mask_idx.shape[0]
            gt_cls[idx, gt_labels[idx].unique()] = 1
            if gt_mask_num != 0:
                gt_seg[idx] = torch.from_numpy(gt_mask_idx.max(axis=0)).to(seg_pred.device)
                gt_mask_idx = torch.from_numpy(gt_mask_idx[gt_mask_idx.sum(axis=(1, 2))>20]).to(self.kernel.device)
                if gt_mask_idx.shape[0] != 0:
                    gt_mask_idx_erosion = self.binary_erosion(gt_mask_idx.unsqueeze(1).to(self.kernel.dtype), self.kernel, 1).squeeze(1)
                    # gt_dist[idx] = gt_mask_idx_erosion.max(dim=0)[0]
                    b_idx, b_y, b_x = torch.where(torch.logical_xor(gt_mask_idx, gt_mask_idx_erosion)==True)
                    b_loc = [torch.stack((b_y[b_idx==i], b_x[b_idx==i]), dim=1) for i in range(gt_mask_num)]
                    f_idx, f_y, f_x = torch.where(gt_mask_idx_erosion==1)
                    f_loc = [torch.stack((f_y[f_idx==i], f_x[f_idx==i]), dim=1) for i in range(gt_mask_num)]
                    loc_li = [(b_loc[i], f_loc[i]) for i in range(gt_mask_num)]
                    tmp_dist_li = list(map(self.dist_trans, loc_li))
                    for i in range(gt_mask_num):
                        if tmp_dist_li[i] is None:
                            continue
                        gt_dist[idx][f_y[f_idx==i], f_x[f_idx==i]] = tmp_dist_li[i]
                    del loc_li
                    del tmp_dist_li
            if gt_mask_non.shape[0] != 0:
                gt_pos_mask[idx] = torch.from_numpy(1 - gt_mask_non.max(axis=0)).to(seg_pred.device)

        gt_pos_mask = gt_pos_mask.detach()
        loss_semantic_seg = self.seg_criterion(seg_pred, gt_seg, mask=gt_pos_mask)+self.dist_criterion(seg_dist*gt_pos_mask, gt_dist)
        loss_global_cls = self.cls_criterion(seg_cls, gt_cls)
        #TODO acc cal
        return loss_semantic_seg, loss_global_cls


@HEADS.register_module()
class HTCSegBranch(BaseModule):
    r"""Multi-level fused semantic segmentation head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        x -> 1x1 conv  - -|||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*2)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 num_convs=2,
                 num_classes=2,
                 in_channels=256,
                 conv_out_channels=256,
                 out_size=512,
                 conv_cfg=None,
                 norm_cfg=None,
                 ignore_label=None,
                 loss_weight=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=0.2),
                 init_cfg=dict(
                     type='Kaiming', override=dict(name='convs'))):
        super(HTCSegBranch, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.num_convs = num_convs
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        if not (isinstance(out_size, list) or isinstance(out_size, tuple)):
            self.out_size = (out_size, out_size)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.num_ins if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_embedding = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = conv_out_channels*2 if i == 0 else conv_out_channels
            self.conv_embedding.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_seg = nn.Conv2d(conv_out_channels, self.num_classes, 1)

        if ignore_label:
            loss_seg['ignore_index'] = ignore_label
        if loss_weight:
            loss_seg['loss_weight'] = loss_weight
        if ignore_label or loss_weight:
            warnings.warn('``ignore_label`` and ``loss_weight`` would be '
                          'deprecated soon. Please set ``ingore_index`` and '
                          '``loss_weight`` in ``loss_seg`` instead.')
        self.seg_criterion = build_loss(loss_seg)
        kernel = torch.ones((1, 1, 5, 5))
        self.register_buffer('kernel', kernel)

    @torch.no_grad()
    def binary_erosion(self, binary_mask, kernel, iterations):
        for i in range(iterations):
            binary_mask = F.conv2d(binary_mask, kernel, padding=kernel.shape[-1]//2)
            binary_mask = torch.clamp(binary_mask-kernel.sum()+1, min=0, max=1)
        return binary_mask

    @auto_fp16()
    def forward(self, img, semantic_feat):
        x = img
        for i in range(self.num_convs):
            x = self.convs[i](x)
        semantic_feat = F.interpolate(semantic_feat, size=img.shape[2:], mode='bilinear', align_corners=True).squeeze(1)
        fused_feat = torch.cat((x, semantic_feat), dim=1)
        for i in range(self.num_convs):
            fused_feat = self.conv_embedding[i](fused_feat)

        seg_pred = self.conv_seg(fused_feat)

        return fused_feat, seg_pred.squeeze(1)

    @force_fp32(apply_to=('seg_pred'))
    def loss(self, seg_pred, labels):
        device = labels.device
        b, c, h, w = labels.shape
        # labels = labels.squeeze(1).long()
        one_hot_labels = torch.zeros((b, self.num_classes+1, h, w), dtype=torch.uint8).to(device)
        one_hot_labels = one_hot_labels.scatter_(dim=1, index=labels.to(torch.int64), src=torch.ones((b, self.num_classes+1, h, w), dtype=torch.uint8).to(device))[:, 1:, :, :]

        loss_semantic_seg = self.seg_criterion(seg_pred, one_hot_labels)
        return loss_semantic_seg
