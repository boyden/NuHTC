# Copyright (c) OpenMMLab. All rights reserved.
import torch
import einops
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import BaseRoIExtractor
import torch.nn.functional as F

@ROI_EXTRACTORS.register_module()
class SelectedRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 selected_levels=None,
                 **kwargs):
        super(SelectedRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        self.selected_levels = selected_levels
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size
        if self.selected_levels is not None:
            num_levels = self.selected_levels
        else:
            num_levels = range(len(feats))
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in num_levels:
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels

        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats

@ROI_EXTRACTORS.register_module()
class LocalGlobalRoIExtractor(BaseRoIExtractor):

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 **kwargs):
        super(LocalGlobalRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size

        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            if i < 2:
                roi_feats_t = self.roi_layers[i](feats[i], rois)
            else:
                roi_feats_global = F.interpolate(feats[i], size=self.roi_layers[i].output_size, mode='bilinear', align_corners=True)
                feat_id = rois[:, 0]
                feat_index = einops.repeat(einops.rearrange(feat_id, 'l -> l 1 1 1'), 'l b h w -> l (b n) (h h1) (w w1)', n=roi_feats_global.size(1), h1=roi_feats_global.size(2), w1=roi_feats_global.size(3))
                roi_feats_t = torch.gather(roi_feats_global, 0, index=feat_index.to(torch.int64))
            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels

        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats

@ROI_EXTRACTORS.register_module()
class AttentionRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels using Transformer.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 start_level=2,
                 thres=0,
                 **kwargs):
        super(AttentionRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        if type(start_level) != list:
            self.start_level = list(range(int(start_level), 10))
        else:
            self.start_level = start_level
        self.thres = thres
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size

        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            if i not in self.start_level:
                roi_feats_t = self.roi_layers[i](feats[i], rois)
            else:
                feat_shape = feats[i].shape
                scale_factor = 4*2**i
                feat_id = rois[:, 0]
                roi_x = torch.div((rois[:, 1] + rois[:, 3]), 2*scale_factor, rounding_mode='floor')
                roi_y = torch.div((rois[:, 2] + rois[:, 4]), 2*scale_factor, rounding_mode='floor')
                roi_x = torch.clamp(roi_x, 0, feat_shape[3]-1)
                roi_y = torch.clamp(roi_y, 0, feat_shape[2]-1)
                roi_loc = torch.stack((feat_id, roi_y, roi_x), dim=1)
                roi_loc_uni, roi_loc_inverse = roi_loc.unique(dim=0, return_inverse=True)
                roi_loc_uni = roi_loc_uni.to(torch.long)

                feat = feats[i].to(torch.float16)
                roi_vec = feat[roi_loc_uni[:, 0], :, roi_loc_uni[:, 1], roi_loc_uni[:, 2]].clone().detach()
                feat_vec = feat[roi_loc_uni[:, 0]].permute(0, 2, 3, 1).view(-1, feat_shape[2]*feat_shape[3], feat_shape[1]).clone().detach()
                roi_sim = F.relu(F.cosine_similarity(roi_vec.unsqueeze(1), feat_vec, dim=2)-self.thres) + self.thres
                roi_sim = roi_sim.view(-1, 1, feat_shape[2], feat_shape[3])
                roi_sim_feat = torch.mean(feat[roi_loc_uni[:, 0]]*roi_sim, dim=(2, 3))
                roi_feats_global = roi_sim_feat[roi_loc_inverse]
                roi_feats_t = einops.repeat(einops.rearrange(roi_feats_global, 'l c-> l c 1 1'), 'l b h w -> l b (h h1) (w w1)', h1=out_size[0], w1=out_size[1])

            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels

        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats

@ROI_EXTRACTORS.register_module()
class PosAttentionRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels using Transformer.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 start_level=2,
                 **kwargs):
        super(PosAttentionRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        self.start_level = start_level
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size

        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            if i < self.start_level:
                roi_feats_t = self.roi_layers[i](feats[i], rois)
            else:
                #TODO complete the self atten ROI Extractor that only contain pos ROIS
                feat_shape = feats[i].shape
                scale_factor = 4*2**i
                feat_id = rois[:, 0]
                roi_x = torch.div((rois[:, 1] + rois[:, 3]), 2*scale_factor, rounding_mode='floor')
                roi_y = torch.div((rois[:, 2] + rois[:, 4]), 2*scale_factor, rounding_mode='floor')
                roi_x = torch.clamp(roi_x, 0, feat_shape[0]-1)
                roi_y = torch.clamp(roi_y, 0, feat_shape[1]-1)
                roi_loc = torch.stack((feat_id, roi_x, roi_y), dim=1)
                roi_loc_uni, roi_loc_inverse = roi_loc.unique(dim=0, return_inverse=True)
                roi_loc_uni = roi_loc_uni.to(torch.long)

                roi_vec = feats[i][roi_loc_uni[:, 0], :, roi_loc_uni[:, 1], roi_loc_uni[:, 2]].clone().detach()
                roi_sim = F.cosine_similarity(roi_vec.unsqueeze(1), roi_vec.unsqueeze(0), dim=2)
                roi_sim = roi_sim/roi_sim.shape[1]
                roi_sim_feat = roi_sim.mm(roi_vec)
                roi_feats_global = roi_sim_feat[roi_loc_inverse]
                roi_feats_t = einops.repeat(einops.rearrange(roi_feats_global, 'l c-> l c 1 1'), 'l b h w -> l b (h h1) (w w1)', h1=out_size[0], w1=out_size[1])

            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels

        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats