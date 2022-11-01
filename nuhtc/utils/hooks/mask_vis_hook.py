import mmcv
from mmcv.runner.hooks import HOOKS, Hook
from ..logger import log_image_with_boxes, color_transform, log_image, log_image_with_masks
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib
import einops
import mmdet
from PIL import Image

matplotlib.use('Agg')

EPS = 1e-2
def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=1,
                      font_size=10,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False, dpi=50)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img

@HOOKS.register_module()
class Mask_Vis_Hook(Hook):
    def __init__(self, interval=100, score_thr=0.3, min_area=10):
        self.interval = interval
        self.score_thr = score_thr
        self.CLASSES = None
        self.min_area = min_area

    def _mask_post_process(self, pred_mask_li):
        if len(pred_mask_li) == 0 or pred_mask_li is None:
            return pred_mask_li, None
        pred_mask_area = pred_mask_li.sum(axis=(1, 2))
        select_area_id = pred_mask_area >= self.min_area
        # if np.sum(select_id) == 0:
        #     return pred_mask_li[select_id], select_id
        # else:
        #     pred_mask_li = pred_mask_li[select_id]
        #     pred_mask_area = pred_mask_area[select_id]

        mask_overlap = (pred_mask_li[:, None, :, :]*pred_mask_li[None, :, :, :]).sum(axis=(2, 3))
        mask_area = np.diag(pred_mask_area)
        mask_overlap = mask_overlap - mask_area

        # the ratio between the maximum overlap and own area
        pred_mask_area[pred_mask_area==0] = 1
        mask_ratio = mask_overlap/pred_mask_area
        mask_ratio = mask_ratio.max(axis=1)
        select_raio_id = mask_ratio < 0.95
        select_id = select_raio_id & select_area_id

        pred_mask_li = pred_mask_li[select_id]
        return pred_mask_li, select_id

    def vis_inst(self, runner, data, segms, bboxes, labels, state='val'):
        if not segms is None:

            log_image_with_boxes(
                f"{state}/gt box",
                data["img"].data[0][0],
                data['gt_bboxes'].data[0][0],
                bbox_tag="gt_label",
                labels=data['gt_labels'].data[0][0],
                class_names=self.CLASSES,
                interval=1,
                img_norm_cfg=data["img_metas"].data[0][0]["img_norm_cfg"],
            )

            if self.score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                segms = segms[inds, ...]

            img_ori = color_transform(data["img"].data[0][0], **data["img_metas"].data[0][0]['img_norm_cfg']).astype(np.uint8)
            img_ori = Image.fromarray(img_ori).resize(data["img_metas"].data[0][0]['ori_shape'][:2], Image.BILINEAR)
            img_ori = np.array(img_ori)
            img_seg = imshow_det_bboxes(
                    img_ori,
                    bboxes,
                    labels,
                    segms,
                    class_names=self.CLASSES,
                    show=False)

            log_image(f"{state}/vis seg", img_seg, interval=1)

            label_dict = {i+1:cls for i, cls in enumerate(self.CLASSES)}

            segms_masks = np.zeros(segms.shape[1:], dtype=np.uint8)

            for seg_idx in range(len(segms)):
                segms_masks[segms[seg_idx]==True] = labels[seg_idx]+1

            # TODO The visualization of img_seg differ much from segms_masks
            #  Fixed by adding score_thr
            segms_dict = {
                'predictions':{
                    'mask_data': segms_masks,
                    'class_labels': label_dict
                }
            }

            log_image_with_masks(f"{state}/inst seg",
                                 img_ori,
                                 segms_dict,
                                 interval=1,
                                 )

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if curr_step%self.interval != 0:
            return
        if isinstance(runner, mmcv.runner.iter_based_runner.IterBasedRunner):
            self.CLASSES = runner.data_loader._dataloader.dataset.CLASSES
            data = next(runner.data_loader).copy()
        if isinstance(runner, mmcv.runner.epoch_based_runner.EpochBasedRunner):
            self.CLASSES = runner.data_loader.dataset.CLASSES
            data = next(iter(runner.data_loader)).copy()

        if isinstance(runner.model.module, mmdet.models.detectors.yolact.YOLACT):
            self.vis_yolo(runner, data)
        elif isinstance(runner.model.module, mmdet.models.detectors.HybridTaskCascade):
            # return
            self.vis_htc(runner, data)
        elif isinstance(runner.model.module, mmdet.models.detectors.CascadeRCNN):
            return
        else:
            self.vis_rcnn(runner, data)

    def after_val_epoch(self, runner):
        # TODO does not work
        if isinstance(runner.model.module, mmdet.models.detectors.yolact.YOLACT):
            return
        # if isinstance(runner.model.module, mmdet.models.detectors.HybridTaskCascade):
        #     return
        if isinstance(runner, mmcv.runner.iter_based_runner.IterBasedRunner):
            self.CLASSES = runner.data_loader._dataloader.dataset.CLASSES
            data = next(runner.data_loader).copy()
        if isinstance(runner, mmcv.runner.epoch_based_runner.EpochBasedRunner):
            self.CLASSES = runner.data_loader.dataset.CLASSES
            data = next(iter(runner.data_loader)).copy()
        with torch.no_grad():
            result = runner.model(img=data['img'].data, img_metas=data['img_metas'].data, return_loss=False, rescale=True)
            if isinstance(runner.model.module, mmdet.models.detectors.HybridTaskCascade):
                img_in = data['img'].data[0][0].unsqueeze(0).to(f'cuda:{runner.model.output_device}')
                feat = runner.model.module.extract_feat(img_in)
                if hasattr(runner.model.module.roi_head, 'semantic_head') and runner.model.module.roi_head.semantic_head:
                    semantic_mask, semantic_feat = runner.model.module.roi_head.semantic_head(feat)
                    semantic_mask = (torch.sigmoid(semantic_mask.squeeze())*255).to(torch.uint8).cpu().numpy()
                    log_image(f"val/semantic mask", semantic_mask, interval=1)

                if hasattr(runner.model.module.roi_head, 'seg_head') and runner.model.module.roi_head.seg_head:
                    if hasattr(runner.model.module.roi_head, 'semantic_head') and runner.model.module.roi_head.semantic_head:
                        seg_feat, seg_mask, seg_dist, _ = runner.model.module.roi_head.seg_head(img_in, semantic_feat)
                    else:
                        seg_feat, seg_mask, seg_dist, _ = runner.model.module.roi_head.seg_head(img_in, feat[0])
                    seg_mask = (torch.sigmoid(seg_mask.squeeze())*255).to(torch.uint8).cpu().numpy()
                    seg_dist = (torch.sigmoid(seg_dist.squeeze())*255).to(torch.uint8).cpu().numpy()
                    log_image(f"val/seg mask", seg_mask, interval=1)
                    log_image(f"val/seg dist", seg_dist, interval=1)
        # copy from thirdparty/mmdetection/mmdet/models/detectors/base.py#L310
        bbox_results, mask_results = result[0][0], result[0][1]
        # TODO agnostic mask vis
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)]
        bboxes = np.concatenate(bbox_results)
        labels = np.concatenate(labels)

        segms = None
        if mask_results is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(mask_results)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        self.vis_inst(runner, data, segms, bboxes, labels, state='val')

    def vis_yolo(self, runner, data):
        with torch.no_grad():
            result = runner.model(img=data['img'].data, img_metas=data['img_metas'].data, return_loss=False, rescale=True)
            img_idx = torch.randperm(len(data['img'].data[0]))[0]
            img_in = data['img'].data[0][0].unsqueeze(0).to(f'cuda:{runner.model.output_device}')
            feat = runner.model.module.extract_feat(img_in)
            feat_up = einops.rearrange(feat[0], 'b (p1 p2 c) h w-> b c (h p1) (w p2)', p1=4, p2=4)
            feat_up = (torch.sigmoid(feat_up.mean(dim=1).squeeze(0))*255).to(torch.uint8).cpu().numpy()

        img_ori = np.array(Image.open(data["img_metas"].data[0][0]['filename']))
        # copy from thirdparty/mmdetection/mmdet/models/detectors/base.py#L310
        bbox_results, mask_results = result[0][0], result[0][1]
        # TODO agnostic mask vis
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)]
        bboxes = np.concatenate(bbox_results)
        labels = np.concatenate(labels)

        segms = None
        if mask_results is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(mask_results)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if not segms is None:
            log_image_with_boxes(
                "train/gt box",
                data["img"].data[0][0],
                data['gt_bboxes'].data[0][0],
                bbox_tag="gt_label",
                labels=data['gt_labels'].data[0][0],
                class_names=self.CLASSES,
                interval=1,
                img_norm_cfg=data["img_metas"].data[0][0]["img_norm_cfg"],
            )

            if self.score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]

            img_seg = imshow_det_bboxes(
                    img_ori,
                    bboxes,
                    labels,
                    segms,
                    class_names=self.CLASSES,
                    show=False)

            log_image("train/vis seg", img_seg, interval=1)

            # for feat_idx in range(feat_up.shape[0]):
            #     log_image(f"train/feat map:{feat_idx}", feat_up[feat_idx], interval=1)
            log_image(f"train/feat map", feat_up, interval=1)

            label_dict = {i+1:cls for i, cls in enumerate(self.CLASSES)}

            segms_masks = np.zeros(segms.shape[1:], dtype=np.uint8)

            for seg_idx in range(len(segms)):
                segms_masks[segms[seg_idx]==True] = labels[seg_idx]+1

            # TODO The visualization of img_seg differ much from segms_masks
            #  Fixed by annd score_thr
            segms_dict = {
                'predictions':{
                    'mask_data': segms_masks,
                    'class_labels': label_dict
                }
            }

            log_image_with_masks("train/inst seg",
                                 img_ori,
                                 segms_dict,
                                 interval=1,)

    def vis_htc(self, runner, data):
        with torch.no_grad():
            # handle the problem of different image shape
            if 'NuCLS' in runner.data_loader.dataset.__class__.__name__:
                for i in range(len(data['img_metas'].data[0])):
                    img_shape = data['img_metas'].data[0][i]['img_shape']
                    scale_factor = data['img_metas'].data[0][i]['scale_factor'][0]
                    data['img_metas'].data[0][i]['ori_shape'] = (int(img_shape[0]/scale_factor), int(img_shape[1]/scale_factor), img_shape[2])
            result = runner.model(img=data['img'].data, img_metas=data['img_metas'].data, return_loss=False, rescale=True)
            img_in = data['img'].data[0][0].unsqueeze(0).to(f'cuda:{runner.model.output_device}')
            feat = runner.model.module.extract_feat(img_in)
            if hasattr(runner.model.module.roi_head, 'semantic_head') and runner.model.module.roi_head.semantic_head:
                semantic_mask, semantic_feat = runner.model.module.roi_head.semantic_head(feat)
                semantic_mask = (torch.sigmoid(semantic_mask.squeeze())*255).to(torch.uint8).cpu().numpy()
                log_image(f"train/semantic mask", semantic_mask, interval=1)

            if hasattr(runner.model.module.roi_head, 'seg_head') and runner.model.module.roi_head.seg_head:
                if hasattr(runner.model.module.roi_head, 'semantic_head') and runner.model.module.roi_head.semantic_head:
                    seg_feat, seg_mask, seg_dist, _ = runner.model.module.roi_head.seg_head(img_in, semantic_feat)
                else:
                    seg_feat, seg_mask, seg_dist, _ = runner.model.module.roi_head.seg_head(img_in, feat[0])
                seg_mask = (torch.sigmoid(seg_mask.squeeze())*255).to(torch.uint8).cpu().numpy()
                seg_dist = (torch.sigmoid(seg_dist.squeeze())*255).to(torch.uint8).cpu().numpy()
                log_image(f"train/seg mask", seg_mask, interval=1)
                log_image(f"train/seg dist", seg_dist, interval=1)

        img_ori = color_transform(data["img"].data[0][0], **data["img_metas"].data[0][0]['img_norm_cfg']).astype(np.uint8)
        img_ori = Image.fromarray(img_ori).resize(data["img_metas"].data[0][0]['ori_shape'][:2], Image.BILINEAR)
        img_ori = np.array(img_ori)
        # copy from thirdparty/mmdetection/mmdet/models/detectors/base.py#L310
        bbox_results, mask_results = result[0][0], result[0][1]
        # TODO agnostic mask vis
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)]
        bboxes = np.concatenate(bbox_results)
        labels = np.concatenate(labels)


        segms = None
        if mask_results is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(mask_results)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        self.vis_inst(runner, data, segms, bboxes, labels, state='train')

    def vis_mask_rcnn(self, runner, data):

        with torch.no_grad():
            result = runner.model(img=data['img'].data, img_metas=data['img_metas'].data, return_loss=False, rescale=False)
            img_idx = torch.randperm(len(data['img'].data[0]))[0]
            img_in = data['img'].data[0][0].unsqueeze(0).to(f'cuda:{runner.model.output_device}')
            feat = runner.model.module.extract_feat(img_in)
            feat_up = einops.rearrange(feat[0], 'b (p1 p2 c) h w-> b c (h p1) (w p2)', p1=4, p2=4)
            feat_up = runner.model.module.conv_last(torch.cat((img_in, feat_up), dim=1))[0][0]
            feat_up = (torch.sigmoid(feat_up)*255).to(torch.uint8).cpu().numpy()
        # copy from thirdparty/mmdetection/mmdet/models/detectors/base.py#L310
        bbox_results, mask_results = result[0][0], result[0][1]
        # TODO agnostic mask vis
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)]
        bboxes = np.concatenate(bbox_results)
        labels = np.concatenate(labels)

        segms = None
        if mask_results is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(mask_results)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if not segms is None:
            log_image_with_boxes(
                "train/gt box",
                data["img"].data[0][0],
                data['gt_bboxes'].data[0][0],
                bbox_tag="gt_label",
                labels=data['gt_labels'].data[0][0],
                class_names=self.CLASSES,
                interval=1,
                img_norm_cfg=data["img_metas"].data[0][0]["img_norm_cfg"],
            )

            if self.score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]

            img_seg = imshow_det_bboxes(
                    color_transform(data["img"].data[0][0], **data["img_metas"].data[0][0]['img_norm_cfg']),
                    bboxes,
                    labels,
                    segms,
                    class_names=self.CLASSES,
                    show=False)

            log_image("train/vis seg", img_seg, interval=1)

            # for feat_idx in range(feat_up.shape[0]):
            #     log_image(f"train/feat map:{feat_idx}", feat_up[feat_idx], interval=1)
            log_image(f"train/feat map", feat_up, interval=1)

            label_dict = {i+1:cls for i, cls in enumerate(self.CLASSES)}

            segms_masks = np.zeros(segms.shape[1:], dtype=np.uint8)

            for seg_idx in range(len(segms)):
                segms_masks[segms[seg_idx]==True] = labels[seg_idx]+1

            # TODO The visualization of img_seg differ much from segms_masks
            #  Fixed by annd score_thr
            segms_dict = {
                'predictions':{
                    'mask_data': segms_masks,
                    'class_labels': label_dict
                }
            }

            log_image_with_masks("train/inst seg",
                                 data["img"].data[0][0],
                                 segms_dict,
                                 interval=1,
                                 img_norm_cfg=data["img_metas"].data[0][0]["img_norm_cfg"],)

    def vis_rcnn(self, runner, data):

        with torch.no_grad():
            result = runner.model(img=data['img'].data, img_metas=data['img_metas'].data, return_loss=False, rescale=False)
        # copy from thirdparty/mmdetection/mmdet/models/detectors/base.py#L310
        bbox_results, mask_results = result[0][0], result[0][1]
        # TODO agnostic mask vis
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)]
        bboxes = np.concatenate(bbox_results)
        labels = np.concatenate(labels)

        segms = None
        if mask_results is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(mask_results)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        self.vis_inst(runner, data, segms, bboxes, labels, state='train')

