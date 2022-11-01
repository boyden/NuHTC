# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.random_sampler import RandomSampler


@BBOX_SAMPLERS.register_module()
class InstanceBalancedPosSampler_Cus(RandomSampler):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() == 0:
            return pos_inds
        elif pos_inds.numel() <= num_expected:
            num_pos = pos_inds.numel()
            unique_gt_labels = assign_result.labels[pos_inds].unique()
            num_gts = len(unique_gt_labels)
            num_per_gt = int(round(num_pos / float(num_gts)) + 1)
            sampled_inds = []
            for i in unique_gt_labels:
                inds = torch.nonzero(
                    assign_result.labels == i.item(), as_tuple=False)
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue

                inds_idx = torch.randint(len(inds), (num_per_gt,))
                inds = inds[inds_idx]
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            return sampled_inds
        else:
            unique_gt_labels = assign_result.labels[pos_inds].unique()
            num_gts = len(unique_gt_labels)
            num_per_gt = int(round(num_expected / float(num_gts)) + 1)
            sampled_inds = []
            for i in unique_gt_labels:
                inds = torch.nonzero(
                    assign_result.labels == i.item(), as_tuple=False)
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue

                inds_idx = torch.randint(len(inds), (num_per_gt,))
                inds = inds[inds_idx]
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                sampled_inds = self.random_choice(sampled_inds, num_expected)
            return sampled_inds
