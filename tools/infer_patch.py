#!/usr/bin/env python
"""
Script to segment nuclei from images listed in a CSV file and save results in COCO format.

Each row of the CSV is one image, segmented as a whole. Images too large to fit
the model's field of view can instead be cut into patches with ``--patch``,
which follows ``tools/infer_wsi.py``: patches of ``--patch_size`` every
``--step_size``, instances touching a patch edge dropped by ``--margin``, mask
NMS within the patch, and no comparison between patches. Coordinates in the
output are always full-image, so a nucleus does not have to be tracked back to
the patch it came from.

Usage:
python tools/infer_patch.py \
    --csv data/labels.csv \
    --config configs/config.py \
    --checkpoint models/checkpoint.pth \
    --output output/nuclei_coco.json \
    --device cuda \
    --mag 40 \
    --batch-size 32

Patched, for images larger than a training crop:
python tools/infer_patch.py ... --patch --patch_size 512 --step_size 448

"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools import coco
from torch.utils.data import Dataset, DataLoader

# Add project paths
proj_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)

import mmcv
from mmcv import Config
from mmdet.apis import inference_detector
from nuhtc.apis.inference import init_detector
from nuhtc.utils import patch_config
from tools.nuclei_merge import merge_overlap


INST_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
               [255, 0, 255]]


class ImageDataset(Dataset):
    """Dataset for loading images from file paths."""
    def __init__(self, image_paths):
        """
        Args:
            image_paths (list): List of image file paths
        """
        self.image_paths = image_paths
        # self._id = 0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        # self._id += 1
        return img, {
            'id': idx+1,
            'file_name': os.path.basename(img_path),
            'img_path': img_path,
            'height': h,
            'width': w
        }


def patch_origins(length, size, step):
    """Patch start coordinates along one axis, the last one flush with the end.

    Walking in strides of ``step`` usually leaves a remainder narrower than a
    patch, so a final patch is anchored to the far edge. It overlaps its
    predecessor more than the others do, which costs a little compute but keeps
    every pixel covered at full patch context.
    """
    if length <= size:
        return [0]
    xs = list(range(0, length - size + 1, step))
    if xs[-1] + size < length:
        xs.append(length - size)
    return xs


class PatchDataset(Dataset):
    """The patches of one image, the role Whole_Slide_Bag_FP plays for a slide."""
    def __init__(self, img, size, step):
        self.img = img
        self.size = size
        self.coords = [(x, y)
                       for y in patch_origins(img.shape[0], size, step)
                       for x in patch_origins(img.shape[1], size, step)]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        return self.img[y:y + self.size, x:x + self.size], (x, y)


def collate_fn(batch):
    """Custom collate function to handle variable-sized images."""
    images = []
    infos = []
    
    for item in batch:
        img, info = item
        images.append(img)
        infos.append(info)
    return images, infos


def mask_nms(masks, pred_scores, thr=0.1, min_area=None):
    """https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L98

    Returns:
        tuple: kept dets and indice.
    """
    if isinstance(masks[0], np.ndarray):
        masks = [coco.maskUtils.encode(np.asfortranarray(mask)) for mask in masks]
    sort_idx = np.argsort(pred_scores)[::-1]
    mask_len = len(masks)
    tmp_masks = np.array(masks)[sort_idx]
    mask_iou = coco.maskUtils.iou(tmp_masks.tolist(), tmp_masks.tolist(), [0] * mask_len)

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
        # Vectorized elimination of subsequent masks with high IoU
        # keep_idx[i+1:] &= (mask_iou[i, i+1:] <= thr)
    return tmp_masks[keep_idx==1].tolist(), sort_idx[keep_idx==1]


def post_process(result, min_area=10, mask_nms_thr=0.1, margin=0):
    """Instances kept from one model result: masks, RLEs, scores and labels.

    Drops what min_area rejects and, when margin is set, boxes coming that
    close to the edge, which on a patch means the nucleus was cut in half.
    Callers on the patched path want the masks, to place them on the image;
    the whole-image path wants the RLEs, already in image coordinates.
    """
    empty = np.zeros(0, dtype=np.uint8), [], np.zeros(0), np.zeros(0, dtype=np.int32)
    masks = np.array(mmcv.concat_list(result[1]), dtype=np.uint8)
    if len(masks) == 0:
        return empty
    dets = np.concatenate(result[0])
    labels = np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32)
                             for i, bbox in enumerate(result[0])])
    keep = (masks == 1).sum(axis=(1, 2)) >= min_area
    if margin:
        h, w = masks.shape[1:]
        keep &= ((dets[:, 0] >= margin) & (dets[:, 1] >= margin) &
                 (dets[:, 2] <= w - margin) & (dets[:, 3] <= h - margin))
    if not keep.any():
        return empty
    masks, dets, labels = masks[keep], dets[keep], labels[keep]
    rles, idx = mask_nms(masks, dets[:, 4], thr=mask_nms_thr)
    return masks[idx], rles, dets[idx][:, 4], labels[idx]


def map_instance(mask, x, y, score, label):
    """One instance as its outline in image coordinates, as infer_wsi.py does.

    The mask is dropped here: every consumer downstream -- the cross-patch
    merge, the geojson, the COCO segmentation and box -- works off the polygon,
    and the coordinate correction is then the patch offset added to a few dozen
    points. The alternative was to place the mask on a full-image canvas and
    RLE-encode it, 9 MB per nucleus on a 3000x3000 image, which profiling put
    at 57% of the wall clock.

    Only the largest contour is kept: an instance is one blob, and the stray
    interpolation specks the mask head leaves behind would otherwise turn into
    their own polygons in QuPath. Returns None when nothing traceable is left.
    """
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    # the default puts an empty mask on the same path as a degenerate outline:
    # below three points there is no ring for shapely to intersect later
    contour = max(contours, key=cv2.contourArea, default=np.empty((0, 1, 2)))
    if len(contour) < 3:
        return None
    contour = np.concatenate([contour[:, 0, :], contour[[0], 0, :]]) + [x, y]
    lo, hi = contour.min(axis=0), contour.max(axis=0)
    return dict(contour=contour,
                box=np.concatenate([lo, hi - lo + 1]).astype(float),
                # the pixel count, not the polygon's area: the outline runs
                # along pixel centres and encloses slightly less
                area=int(np.count_nonzero(mask)),
                score=float(score),
                label=int(label))


def merge_instances(insts, thr):
    """Drop the copies of a nucleus that overlapping patches each reported.

    Patches overlap wherever the last one of a row was pulled back to sit flush
    with the edge, and always when step_size < patch_size, so a nucleus in the
    band was seen twice and neither patch's own NMS could know about the other
    copy. This is the same job tools/nuclei_merge.py does for a whole slide,
    and it is that function doing it here: outlines into a shapely STRtree,
    IoU against everything that intersects, highest score wins. Comparing
    polygons is what lets the masks stay patch-local.
    """
    features = [{
        'geometry': {'type': 'Polygon', 'coordinates': [inst['contour'].tolist()]},
        # inst_id survives the merge in the properties, which is how the kept
        # rows are mapped back onto the instances they came from
        'properties': {'score': inst['score'], 'inst_id': i},
    } for i, inst in enumerate(insts)]
    kept = merge_overlap(features, overlap_threshold=thr,
                         merge_strategy='probability')
    return [insts[p['inst_id']] for p in kept['properties']]


def infer_patch(model, img, args):
    """Segment one image patch by patch, in full-image coordinates.

    The same recipe as tools/infer_wsi.py: instances whose box comes within
    --margin of a patch edge are dropped as truncated, small ones are dropped,
    mask NMS runs within the patch, and what survives is placed back into the
    image. Unlike infer_wsi.py, overlapping patches then get a second mask NMS
    across the whole image, without which the nuclei inside an overlap are
    reported once per patch that sees them.

    Returns what the unpatched path returns: a list of instances, each an
    outline in image coordinates, as built by :func:`map_instance`.
    """
    dataset = PatchDataset(img, args.patch_size, args.step_size or args.patch_size)
    infer_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    insts = []
    for patches, coords in tqdm(infer_dataloader, leave=False,
                                desc=f'{len(dataset)} patches'):
        results = inference_detector(model, patches)
        for (x, y), res in zip(coords, results):
            masks, _, patch_scores, patch_labels = post_process(
                res, args.min_area, args.mask_nms_thr, margin=args.margin)
            insts += [inst for inst in
                      (map_instance(m, x, y, s, l)
                       for m, s, l in zip(masks, patch_scores, patch_labels))
                      if inst is not None]

    if not insts:
        return []
    return merge_instances(insts, args.mask_nms_thr)


def result_dir(args, img_path):
    """Where one image's results go, mirroring infer_wsi.py's layout."""
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    return f'{args.save_dir}/nuclei/{img_id}'


def geojson_path(args, img_path):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    return f'{result_dir(args, img_path)}/{img_id}.geojson'


def coco_path(args, img_path):
    return f'{result_dir(args, img_path)}/coco_nuclei.json'


def to_geojson(insts, classes):
    """QuPath 0.4.4 features for one image: the nucleus outlines."""
    polygons = []
    for inst in insts:
        contour = inst['contour']
        label = inst['label']
        polygons.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [contour.tolist()],
            },
            'properties': {
                'objectType': 'annotation',
                'label': int(label),
                'score': inst['score'],
                'classification': {
                    'name': classes[label],
                    'color': INST_COLORS[label % len(INST_COLORS)],
                },
                'isLocked': False,
            },
        })
    return polygons


def to_coco_anns(insts, img_info, first_id):
    """COCO annotations for one image, in full-image coordinates.

    ``segmentation`` is the instance outline as a COCO polygon. A consumer that
    wants a mask can rasterise it with ``maskUtils.frPyObjects``, which is the
    same work the old full-image RLE did eagerly for every nucleus.
    """
    return [{
        'id': first_id + i,
        'bbox': inst['box'].tolist(),
        'area': inst['area'],
        'image_id': img_info['id'],
        'category_id': inst['label'],
        'iscrowd': 0,
        'segmentation': [inst['contour'].reshape(-1).tolist()],
        'score': inst['score'],
    } for i, inst in enumerate(insts)]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Segment nuclei from images and save to COCO format'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file containing image_path column'
    )
    parser.add_argument(
        '--image-col',
        type=str,
        default='image_path',
        help='Column name for image paths in the CSV (default: image_path)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='nuclei_coco.json',
        help='Output COCO JSON file path, holding every image of the run. '
             'Ignored under --patch, where each image gets its own '
             'coco_nuclei.json under --save_dir instead'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.35,
        help='score threshold, for the --vis-dir drawings only. What enters '
             'the results is decided by test_cfg.rcnn.score_thr in the config'
    )
    parser.add_argument(
        '--mask-nms-thr',
        type=float,
        default=0.1,
        help='IoU threshold for mask NMS (default: 0.1, set to 1 to disable; '
             '0 suppresses on any overlap at all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:1',
        help='Device for inference (cuda:0, cuda:1, cpu, etc.). Use CUDA_VISIBLE_DEVICES=1 to use second GPU.'
    )
    parser.add_argument(
        '--mag',
        type=int,
        default=40,
        help='Magnification (default: 40x)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Images per forward pass, or patches per forward pass under '
             '--patch, where images are always handled one at a time '
             '(default: 16)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of workers for data loading (default: 8)'
    )
    parser.add_argument(
        '--vis-dir',
        type=str,
        default=None,
        help='Directory to save visualization images (optional, saves first N samples)'
    )
    parser.add_argument(
        '--vis-samples',
        type=int,
        default=10,
        help='Number of sample images to visualize (default: 10)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='coco',
        choices=['coco', 'qupath', 'all'],
        help='mode of save format. coco writes --output; qupath writes the '
             'nucleus outlines under --save_dir as tools/infer_wsi.py does'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='directory to save processed data, required by --mode qupath'
    )
    parser.add_argument(
        '--patch',
        default=False,
        action='store_true',
        help='Cut images larger than --patch_size into overlapping patches '
             'before inference, as tools/infer_wsi.py does. Off by default, '
             'i.e. each image is segmented whole'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='patch_size'
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=256,
        help='step_size; the difference from --patch_size is the overlap, '
             'which is what lets a nucleus cut by one patch be recovered from '
             'the next'
    )
    parser.add_argument(
        '--margin',
        type=int,
        default=0,
        help='discard the contour which distance is less than margin number '
             'pixels to edges'
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=10,
        help='discard the area less than min_area'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_coco = args.mode == 'coco' or args.mode == 'all'
    write_qupath = args.mode == 'qupath' or args.mode == 'all'
    # where the coco output lands. An image big enough to need patching carries
    # too many full-image masks to hold every image's worth in memory until the
    # end, so patching writes one coco json per image, next to its geojson, the
    # moment that image is done. Without patching the annotations accumulate and
    # leave as a single merged file at --output
    split_coco = write_coco and args.patch
    merged_coco = write_coco and not args.patch
    if (write_qupath or split_coco) and not args.save_dir:
        raise ValueError(f"--mode {args.mode} with --patch needs --save_dir")
    if merged_coco and os.path.exists(args.output):
        print(f"Skipping {args.output}.")
        return

    # Load CSV
    print(f"Loading CSV from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.image_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.image_col}' column")
    
    print(f"Found {len(df)} images in CSV")

    image_paths = df[args.image_col].tolist()
    if args.patch:
        # patching writes every output per image, so a killed run can pick up
        # where it stopped. An image counts as done once every file its --mode
        # asks for is there
        def is_done(p, mode):
            if mode == 'qupath':
                return os.path.exists(geojson_path(args, p))
            elif mode == 'coco':
                return os.path.exists(coco_path(args, p))
            else:
                return os.path.exists(geojson_path(args, p)) and os.path.exists(coco_path(args, p))

        todo = [p for p in image_paths if not is_done(p, args.mode)]
        print(f'skip {len(image_paths) - len(todo)} images due to existing results')
        image_paths = todo
    
    # Load model
    print(f"Model Config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    
    # Adjust scale factor for magnification
    for test_pipe in cfg['data']['test']['pipeline']:
        if test_pipe['type'] == 'MultiScaleFlipAug':
            scale_factor = float(80 / args.mag)
            test_pipe['scale_factor'] = scale_factor
            print(f'MultiScaleFlipAug scale factor set to: {test_pipe["scale_factor"]}')
            # Also update SmartResize if it exists in transforms
            if 'transforms' in test_pipe:
                for transform in test_pipe['transforms']:
                    if transform['type'] == 'SmartResize':
                        transform['scale_factor'] = scale_factor
                        print(f'SmartResize scale factor set to: {transform["scale_factor"]}')
    
    model = init_detector(cfg, args.checkpoint, device=args.device)
    MAIN_CLASSES = ('nucleus',)
    model.CLASSES = MAIN_CLASSES
    
    # Initialize COCO structure
    categories = [{'id': i, 'name': name, 'supercategory': 'nucleus'}
                  for i, name in enumerate(MAIN_CLASSES)]
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories,
    }
    total_images = 0
    total_nuclei = 0
    
    # Create dataset and dataloader. Under --patch the batch and the workers
    # belong to the patches of a single image, as in tools/infer_wsi.py, so the
    # images themselves are walked one at a time here
    dataset = ImageDataset(image_paths)
    if args.patch:
        dataloader = (collate_fn([dataset[i]]) for i in range(len(dataset)))
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_fn  # Use custom collate function
        )
    
    # Process images in batches
    nuclei_id = 1
    vis_count = 0
    
    # Create visualization directory if specified
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Visualization images will be saved to {args.vis_dir}")

    if write_qupath:
        print(f"QuPath geojson will be saved to {args.save_dir}/nuclei")
    
    print(f"Processing {len(image_paths)} images, "
          f"{args.batch_size} {'patches' if args.patch else 'images'} at a time...")

    desc = 'Processing images' if args.patch else 'Processing batches'
    progress = tqdm(dataloader, desc=desc, total=len(image_paths) if args.patch
                    else None)
    for batch_idx, batch_data in enumerate(progress):
        try:
            # Unpack batch data - collate_fn returns (images, infos)
            batch_images, batch_infos = batch_data
            
            # under --patch the loop walks one image at a time and args.batch_size
            # belongs to that image's patches. Without it the whole batch goes
            # through the model at once. An image smaller than a patch needs no
            # cutting up either way
            img = batch_images[0]
            if args.patch and max(img.shape[:2]) > args.patch_size:
                outputs = [infer_patch(model, img, args)]
            else:
                # margin=0: nothing was cut here, so a nucleus touching the
                # border is a real one and has to stay
                det_results = [post_process(res, min_area=args.min_area,
                                            mask_nms_thr=args.mask_nms_thr, margin=0)
                               for res in inference_detector(model, batch_images)]
                # nothing was cut up here, so the patch offset is the origin
                outputs = [[inst for inst in
                            (map_instance(m, 0, 0, s, l)
                             for m, s, l in zip(masks, scores, labels))
                            if inst is not None]
                           for masks, _, scores, labels in det_results]

            for img_info, insts in zip(batch_infos, outputs):
                img_anns = to_coco_anns(insts, img_info, nuclei_id)
                nuclei_id += len(img_anns)
                total_images += 1
                total_nuclei += len(img_anns)

                if write_qupath or split_coco:
                    os.makedirs(result_dir(args, img_info['img_path']),
                                exist_ok=True)
                if write_qupath:
                    with open(geojson_path(args, img_info['img_path']), 'w') as f:
                        json.dump(to_geojson(insts, model.CLASSES), f)
                if split_coco:
                    with open(coco_path(args, img_info['img_path']), 'w') as f:
                        json.dump({'images': [img_info],
                                   'annotations': img_anns,
                                   'categories': categories}, f)
                elif write_coco:
                    coco_data['images'].append(img_info)
                    coco_data['annotations'].extend(img_anns)

                if img_anns and args.vis_dir and vis_count < args.vis_samples:
                    img = Image.open(img_info['img_path']).convert('RGB')
                    img_draw = ImageDraw.Draw(img)
                    for annt in img_anns:
                        if annt['score'] < args.score_thr:
                            continue
                        x, y, w, h = annt['bbox']
                        img_draw.rectangle([x, y, x + w, y + h], fill=None, outline='green', width=1)
                        img_draw.text((x, y), f"{annt['score']:.2f}", fill='black')
                    vis_path = os.path.join(args.vis_dir, f"{vis_count:04d}_{img_info['file_name']}")
                    img.save(vis_path)
                    vis_count += 1

        except Exception as e:
            import traceback
            print(f"Error processing batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Save COCO JSON
    if merged_coco:
        print(f"Saving COCO format to {args.output}...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(f"{args.output}", 'w') as f:
            json.dump(coco_data, f, indent=2)

    print(f"\nDone!")
    print(f"  Total images processed: {total_images}")
    print(f"  Total nuclei: {total_nuclei}")


if __name__ == '__main__':
    main()
