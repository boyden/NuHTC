import copy
import os
import shutil
# from PIL import Image

import numpy as np
import torch
import torch.utils.data
import torchvision
import mmdet

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.special import softmax
# from histomicstk.annotations_and_masks.annotation_and_mask_utils import np_vec_no_jit_iou
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from .nucls.nucls_model.torchvision_detection_utils import transforms as T
from .nucls.nucls_model.torchvision_detection_utils import transforms as tvdt

####
def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
# noinspection PyShadowingNames
def _crop_all_to_fov(
        images, targets, outputs=None, cropper=None, crop_targets=True):
    """Crop to fov so that the model looks at a wider field than it
    does inference? This is also important since the dataset I made
    deliberately looks beyond FOV to include full extent of nuclei that
    spill over the FOV edge.
    """
    imgs = deepcopy(images)
    trgs = deepcopy(targets)
    outs = [None] * len(imgs) if outputs is None else deepcopy(outputs)
    cropper = tvdt.Cropper() if cropper is None else cropper
    for imidx, (img, trg, out) in enumerate(zip(imgs, trgs, outs)):
        img, _ = tvdt.ToPILImage()(img.cpu(), {})
        xmin, ymin, xmax, ymax = [int(j) for j in trg['fovloc']]
        rgb1 = None
        rgb2 = None
        if crop_targets:
            rgb1, trg = cropper(
                rgb=img, targets=trg,
                i=ymin, h=ymax - ymin, j=xmin, w=xmax - xmin,
            )
        if out is not None:
            out = {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}
            rgb2, out = cropper(
                rgb=img, targets=out,
                i=ymin, h=ymax - ymin, j=xmin, w=xmax - xmin,
            )
        # now assign
        rgb = rgb1 or rgb2
        imgs[imidx], _ = tvdt.PILToTensor()(rgb, {})
        trgs[imidx] = trg
        outs[imidx] = out

    return imgs, trgs, outs

def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset

def convert_to_coco_api(ds, crop_inference_to_fov=False):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    cropper = tvdt.Cropper() if crop_inference_to_fov else None
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]

        if crop_inference_to_fov:
            img, targets, _ = _crop_all_to_fov(
                images=[img], targets=[targets], cropper=cropper)
            img = img[0]
            targets = targets[0]

        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(ds, crop_inference_to_fov=False, targets_all=None):
    if isinstance(ds, mmdet.datasets.coco.CocoDataset) and not crop_inference_to_fov:
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in ds.img_ids:
            img_dict = ds.coco.imgs[img_idx]
            image_id = img_dict['id']
            dataset['images'].append(img_dict)
            ann_ids = ds.coco.get_ann_ids(img_ids=[image_id])
            targets = ds.coco.load_anns(ann_ids)
            num_objs = len(targets)
            for i in range(num_objs):
                ann = targets[i]
                if not ds.do_classification:
                    label_name = ds.CLASSES[ds.cat2label[ann['category_id']]]
                    if label_name != 'AMBIGUOUS':
                        label = 1
                    else:
                        label = 2
                else:
                    label = ann['category_id']
                ann['category_id'] = label
                categories.add(label)
                dataset['annotations'].append(ann)
                ann_id += 1

        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds
    if isinstance(ds, mmdet.datasets.coco.CocoDataset) and crop_inference_to_fov:
        targets_all = copy.deepcopy(targets_all)
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in range(len(targets_all)):
            targets = targets_all[img_idx]
            image_id = targets['image_id']
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['height'] = targets['img_shape'][0]
            img_dict['width'] = targets['img_shape'][1]
            dataset['images'].append(img_dict)
            bboxes = targets["boxes"]
            bboxes[:, 2:] -= bboxes[:, :2]
            areas = bboxes[:, -2] * bboxes[:, -1]
            bboxes = bboxes.tolist()
            areas = areas.tolist()
            labels = targets['labels'].tolist()
            iscrowd = targets['iscrowd'].tolist()
            if 'masks' in targets:
                masks = targets['masks']
            if 'keypoints' in targets:
                keypoints = targets['keypoints']
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                categories.add(labels[i])
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id
                if not ds.do_classification:
                    label_name = ds.CLASSES[ds.cat2label[labels[i]]]
                    if label_name != 'AMBIGUOUS':
                        label = 1
                    else:
                        label = 2
                else:
                    label = labels[i]
                ann['category_id'] = label
                categories.add(label)
                if 'masks' in targets:
                    ann["segmentation"] = coco_mask.encode(masks[i].masks)
                if 'keypoints' in targets:
                    ann['keypoints'] = keypoints[i]
                    ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
                dataset['annotations'].append(ann)
                ann_id += 1

        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds
    if isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    if isinstance(ds,mmdet.datasets.coco.CocoDataset):
        return ds.coco
    return convert_to_coco_api(
        ds, crop_inference_to_fov=crop_inference_to_fov)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode='instances'):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, transforms):
    return get_coco(root, image_set, transforms, mode="person_keypoints")

# noinspection LongLine
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    # Mohamed: I edited this to refer to my MaskRCNN
    if isinstance(model_without_ddp, mmdet.models.detectors.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def _update_classification_metrics(
        metrics_dict, all_labels, rlabelcodes,
        all_scores=None, output_labels=None,
        codemap=None, prefix='',):
    """IMPORTANT NOTE: This assumes that all_labels start at zero and
    are contiguous, and that all_scores is shaped (n_nuclei, n_classes),
    where n_classes is the REAL number of classes.
    """
    if len(all_labels) < 1:
        return metrics_dict

    pexist = all_scores is not None
    if not pexist:
        assert output_labels is not None

    # maybe group scores from classes that belong to the same supercateg
    if codemap is not None:
        tmp_lbls = all_labels.copy()
        tmp_scrs = np.zeros(all_scores.shape) if pexist else None
        for k, v in codemap.items():
            # remap labels
            tmp_lbls[all_labels == k] = v
            # aggregate probab. for classes to be grouped
            if pexist:
                tmp_scrs[:, v] += all_scores[:, k]
        all_labels = tmp_lbls
        all_scores = tmp_scrs

    unique_classes = np.unique(all_labels).tolist()
    n_classes = len(unique_classes)

    if pexist:
        all_preds = np.argmax(all_scores, 1)
    else:
        all_preds = output_labels

    if n_classes > 0:
        # accuracy
        metrics_dict[f'{prefix}accuracy'] = np.mean(0 + (all_preds == all_labels))

        # Mathiew's Correlation Coefficient
        metrics_dict[f'{prefix}mcc'] = matthews_corrcoef(y_true=all_labels, y_pred=all_preds)

        # Class confusions (unnormalized, just numbers)
        for tcc, tc in rlabelcodes.items():
            for pcc, pc in rlabelcodes.items():
                coln = f'{prefix}confusion_trueClass-{tc}_predictedClass-{pc}'
                keep1 = 0 + (all_labels == tcc)
                keep2 = 0 + (all_preds == pcc)
                metrics_dict[coln] = np.sum(0 + ((keep1 + keep2) == 2))

    if n_classes > 1:

        # Class-by-class accuracy

        trg = np.zeros((len(all_labels), n_classes))
        scr = np.zeros((len(all_labels), n_classes))

        for cid, cls in enumerate(unique_classes):

            cls_name = rlabelcodes[cls]

            # Accuracy
            tr = 0 + (all_labels == cls)
            pr = 0 + (all_preds == cls)
            metrics_dict[f'{prefix}accuracy_{cls_name}'] = np.mean(0 + (tr == pr))

            # Mathiew's Correlation Coefficient
            metrics_dict[f'{prefix}mcc_{cls_name}'] = matthews_corrcoef(y_true=tr, y_pred=pr)

            # ROC AUC. Note that it's only defined for classes present in gt
            if pexist:
                trg[:, cid] = 0 + (all_labels == cls)
                scr[:, cid] = all_scores[:, cls]
                metrics_dict[f'{prefix}aucroc_{cls_name}'] = roc_auc_score(
                    y_true=trg[:, cid], y_score=all_scores[:, cid])

        # renormalize with softmax & get rocauc
        if pexist:
            scr = softmax(scr, -1)
            metrics_dict[f'{prefix}auroc_micro'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='micro')
            metrics_dict[f'{prefix}auroc_macro'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='macro')

        print(f"\nClassification results: {prefix}")
        for k, v in metrics_dict.items():
            if k.startswith(prefix) and ('confusion_' not in k):
                print(f'{k}: {v}')

def map_bboxes_using_hungarian_algorithm(bboxes1, bboxes2, min_iou=1e-4):
    """Map bounding boxes using hungarian algorithm.

    Adapted from Lee A.D. Cooper.

    Parameters
    ----------
    bboxes1 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    bboxes2 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    min_iou : float
        minumum iou to match two bboxes to match to each other

    Returns
    -------
    np.array
        matched indices relative to x1, y1

    np.array
        matched indices relative to x2, y2, correspond to the first output

    np.array
        unmatched indices relative to x1, y1

    np.array
        unmatched indices relative to x2, y2

    """
    # generate cost matrix for mapping cells from user to anchors
    max_cost = 1 - min_iou
    costs = 1 - np_vec_no_jit_iou(bboxes1=bboxes1, bboxes2=bboxes2)
    costs[costs > max_cost] = 99.

    # perform hungarian algorithm mapping
    source, target = linear_sum_assignment(costs)

    # discard mappings that are non-allowable
    allowable = costs[source, target] <= max_cost
    source = source[allowable]
    target = target[allowable]

    # find indices of unmatched
    def _find_unmatched(coords, matched):
        potential = np.arange(coords.shape[0])
        return potential[~np.in1d(potential, matched)]
    unmatched1 = _find_unmatched(bboxes1, source)
    unmatched2 = _find_unmatched(bboxes2, target)

    return source, target, unmatched1, unmatched2
