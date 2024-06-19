# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import types

from mmdet.models.builder import LOSSES
from mmdet.models.losses import SeesawLoss, CrossEntropyLoss
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.losses.focal_loss import sigmoid_focal_loss, py_sigmoid_focal_loss
from mmdet.models.losses.accuracy import accuracy

def get_seasaw_accuracy(self, cls_score, labels):
    """Get custom accuracy w.r.t. cls_score and labels.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C + 2).
        labels (torch.Tensor): The learning label of the prediction.

    Returns:
        Dict [str, torch.Tensor]: The accuracy for objectness and classes,
             respectively.
    """
    pos_inds = labels < self.num_classes
    obj_labels = (labels == self.num_classes).long()
    cls_score_classes, cls_score_objectness = self._split_cls_score(
        cls_score)
    acc_objectness = accuracy(cls_score_objectness, obj_labels)
    pos_cls_score_classes = cls_score_classes[pos_inds]
    pos_labels = labels[pos_inds]
    acc_classes = accuracy(pos_cls_score_classes, pos_labels)
    acc = dict()
    acc['acc_objectness'] = acc_objectness
    acc['acc_classes'] = acc_classes
    for label in pos_labels.unique():
        label_id = pos_labels == label
        acc[f'acc_class_{label}'] =  accuracy(pos_cls_score_classes[label_id], pos_labels[label_id])
    return acc

def get_ce_accuracy(self, cls_score, labels):
    """Get custom accuracy w.r.t. cls_score and labels.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C + 2).
        labels (torch.Tensor): The learning label of the prediction.

    Returns:
        Dict [str, torch.Tensor]: The accuracy for objectness and classes,
             respectively.
    """
    num_classes = cls_score.shape[1]
    pos_inds = labels < num_classes-1
    pos_cls_score_classes = cls_score[pos_inds]
    pos_labels = labels[pos_inds]
    acc_objectness = accuracy(cls_score[~pos_inds], labels[~pos_inds])
    acc_classes = accuracy(pos_cls_score_classes, pos_labels)
    acc = dict()
    acc['acc_objectness'] = acc_objectness
    acc['acc_classes'] = acc_classes
    for label in pos_labels.unique():
        label_id = pos_labels == label
        acc[f'acc_class_{label}'] =  accuracy(pos_cls_score_classes[label_id], pos_labels[label_id])
    return acc

SeesawLoss.get_accuracy = get_seasaw_accuracy

CrossEntropyLoss.custom_activation = True
CrossEntropyLoss.get_accuracy = get_ce_accuracy


def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              avg_factor=None):
    """Calculate dice loss, which is proposed in
    `V-Net: Fully Convolutional Neural Networks for Volumetric
    Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)

        if weight.sum() == 0:
            weight = torch.ones_like(weight)
        else:
            weight = weight/weight.sum()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def trunc_loss(pred,
              target,
              weight=None,
              gamma=0.3,
              reduction='mean',
              eps=1e-3,
              avg_factor=None):
    """Calculate dice loss, which is proposed in
    `V-Net: Fully Convolutional Neural Networks for Volumetric
    Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()
    assert gamma>0 and gamma <1
    gamma = torch.tensor(gamma, dtype=target.dtype, device=target.device)
    ret = torch.where(target==1, input, 1-input)

    smooth_truncate = 0.5 - torch.log(gamma) + ((target-1)*((1-input)**2)-target*input**2)/(2*gamma**2)
    condition = ret < gamma

    smooth_truncate_loss = torch.where(condition, smooth_truncate, -torch.log(ret+eps))
    loss = torch.mean(smooth_truncate_loss, dim=1)

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)

        if weight.sum() == 0:
            weight = torch.ones_like(weight)
        else:
            weight = weight/weight.sum()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module()
class SmoothTruncatedLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-3,
                 gamma=0.3):
        super(SmoothTruncatedLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.gamma = gamma
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                mask=None,):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
                if mask is not None:
                    pred = pred*mask
            else:
                raise NotImplementedError

        loss_dice = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)
        loss_trunc = self.loss_weight*trunc_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor
        )
        loss = loss_dice + loss_trunc

        return loss

@LOSSES.register_module()
class PartialDiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-3,
                 gamma=0.3):
        super(PartialDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.gamma = gamma
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                mask=None,):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
                pred = pred.reshape((-1, pred.shape[-2], pred.shape[-1]))
                target = target.reshape((-1, target.shape[-2], target.shape[-1]))
                if mask is None:
                    mask = (target== 1) | (target==0)
                pred = pred*mask
                target = target*mask

            else:
                raise NotImplementedError

        loss_dice = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)

        loss = loss_dice

        return loss

@LOSSES.register_module()
class MultiLabelFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 size_average=True,
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(MultiLabelFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss
            loss = torch.zeros((1,target.shape[1]), device=target.device)
            for idx in range(target.shape[1]):
                loss_cls = self.loss_weight * calculate_loss_func(
                    pred[:, [idx]],
                    target[:, idx],
                    weight,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    reduction=reduction,
                    avg_factor=avg_factor)
                loss[0, idx] = loss_cls

        else:
            raise NotImplementedError
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
