#!/usr/bin/env python
"""
Script to segment nuclei from images listed in a CSV file and save results in COCO format.

Usage:
python tools/infer_patch.py \
    --csv data/labels.csv \
    --config configs/config.py \
    --checkpoint models/checkpoint.pth \
    --output output/nuclei_coco.json \
    --score-thr 0.35 \
    --device cuda \
    --mag 40 \
    --batch-size 32

"""

import time
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
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
from nuhtc.apis.inference import init_detector, save_result
from nuhtc.utils import patch_config


class ImageDataset(Dataset):
    """Dataset for loading images from file paths."""
    def __init__(self, image_paths):
        """
        Args:
            image_paths (list): List of image file paths
        """
        self.image_paths = image_paths
        self._id = 0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        self._id += 1
        return img, {
            'id': self._id,
            'file_name': os.path.basename(img_path),
            'img_path': img_path,
            'height': h,
            'width': w
        }


def collate_fn(batch):
    """Custom collate function to handle variable-sized images."""
    images = []
    infos = []
    
    for item in batch:
        img, info = item
        images.append(img)
        infos.append(info)
    return images, infos


def mask_nms(masks, pred_scores, thr=0.9, min_area=None):
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
        # Vectorized elimination of subsequent masks with high IoU
        keep_idx[i+1:] &= (mask_iou[i, i+1:] <= thr)
    return tmp_masks[keep_idx==1].tolist(), sort_idx[keep_idx==1]


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
        help='Output COCO JSON file path'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.35,
        help='Score threshold for detections'
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
        help='Batch size for processing images (default: 16)'
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
        '--mask-nms-thr',
        type=float,
        default=0.05,
        help='IoU threshold for mask NMS (default: 0.05, set to 0 to disable)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load CSV
    print(f"Loading CSV from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.image_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.image_col}' column")
    
    print(f"Found {len(df)} images in CSV")
    
    # Randomly sample subset for visualization if requested (from the processed subset)
    if args.vis_dir and args.vis_samples > 0:
        num_images = len(df)
        num_samples = min(args.vis_samples, num_images)
        df = df.sample(n=num_samples, random_state=None).reset_index(drop=True)
        print(f"Randomly selected {num_samples} images for visualization: {num_images}")
    
    image_paths = df[args.image_col].tolist()
    
    # Load model
    print(f"Model Config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    
    # Adjust scale factor for magnification
    for test_pipe in cfg['data']['test']['pipeline']:
        if test_pipe['type'] == 'MultiScaleFlipAug':
            test_pipe['scale_factor'] = float(80 / args.mag)
            print(f'Scale factor set to: {test_pipe["scale_factor"]}')
    
    model = init_detector(cfg, args.checkpoint, device=args.device)
    MAIN_CLASSES = ('T', 'I', 'C', 'D', 'E')
    model.CLASSES = MAIN_CLASSES
    
    # Initialize COCO structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 0,
            'name': 'nucleus',
            'supercategory': 'nucleus'
        }]
    }
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    # Process images in batches
    nuclei_id = 0
    vis_count = 0
    
    # Create visualization directory if specified
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Visualization images will be saved to {args.vis_dir}")
    
    print(f"Processing {len(image_paths)} images in batches of {args.batch_size}...")
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
        try:
            # Unpack batch data - collate_fn returns (images, infos)
            batch_images, batch_infos = batch_data
            coco_data['images'].extend(batch_infos)
            
            # Run batch inference (following infer_wsi.py pattern)
            # Load images for inference
            results = inference_detector(model, batch_images)
            seg_mask = [np.array(mmcv.concat_list(tmp_mask[1]), dtype=np.uint8)
                        for tmp_mask in results]
            rle_mask = [[] for tmp_mask in results]

            bbox_results = [np.concatenate(tmp_res[0])[:, :4] for tmp_res in results]
            fg_scores = [np.concatenate(tmp_res[0])[:, 4] for tmp_res in results]
            labels = [np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32)
                                      for i, bbox in enumerate(tmp_res[0])])
                      for tmp_res in results]
                      
            for mask_id in range(len(seg_mask)):
                if len(seg_mask[mask_id]) == 0:
                    continue
                # seg_area = (seg_mask[mask_id] == 1).sum(axis=(1, 2))
                # select_id = (seg_area>=args.min_area)
                # bbox_results[mask_id] = bbox_results[mask_id][select_id]
                # seg_mask[mask_id] = seg_mask[mask_id][select_id]
                # labels[mask_id] = labels[mask_id][select_id]
                # fg_scores[mask_id] = fg_scores[mask_id][select_id]
                # if len(seg_mask[mask_id]) == 0:
                #     continue
                
                # Mask NMS
                tmp_masks, nms_idx = mask_nms(seg_mask[mask_id], fg_scores[mask_id], thr=args.mask_nms_thr)
                
                bbox_results[mask_id] = bbox_results[mask_id][nms_idx]
                seg_mask[mask_id] = seg_mask[mask_id][nms_idx]
                rle_mask[mask_id] = tmp_masks
                labels[mask_id] = labels[mask_id][nms_idx]
                fg_scores[mask_id] = fg_scores[mask_id][nms_idx]
                
                image_id = batch_infos[mask_id]['id']
                img_anns = []
                for i in range(len(seg_mask[mask_id])):
                    rle_inst = rle_mask[mask_id][i]
                    rle_inst['counts'] = rle_inst['counts'].decode('ascii')
                    bbox = coco.maskUtils.toBbox(rle_inst).tolist()
                    area = bbox[2]*bbox[3]
                    annt_dict = {
                        'id': nuclei_id,
                        'bbox': bbox,
                        'area': area,
                        'image_id': image_id,
                        'category_id': int(labels[mask_id][i]),
                        'iscrowd': 0,
                        'segmentation': rle_inst,
                        'score': float(fg_scores[mask_id][i]),
                    }
                    img_anns.append(annt_dict)
                    nuclei_id += 1

                coco_data['annotations'].extend(img_anns)

                if args.vis_dir and vis_count < args.vis_samples:
                    img_info = batch_infos[mask_id]
                    img_path = img_info['img_path']
                    img = Image.open(img_path).convert('RGB')
                    img_draw = ImageDraw.Draw(img)
                    
                    for annt in img_anns:
                        rle_inst = annt['segmentation'].copy()
                        if isinstance(rle_inst['counts'], bytes):
                            rle_inst['counts'] = rle_inst['counts'].decode('ascii')
                        bbox = coco.maskUtils.toBbox(rle_inst).tolist()
                        # Draw rectangle instead of polygon
                        x, y, w, h = bbox
                        img_draw.rectangle([x, y, x + w, y + h], fill=None, outline='green', width=1)
                        # Draw probability (score) in top-left corner of each bbox
                        prob = annt.get('score', None)
                        if prob is not None:
                            text = f"{prob:.2f}"
                            # Use a smaller font size
                            try:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
                            except:
                                try:
                                    font = ImageFont.load_default()
                                except:
                                    font = None
                            img_draw.text((x, y), text, fill="black", font=font)
                    
                    vis_path = os.path.join(args.vis_dir, f"{vis_count:04d}_{img_info['file_name']}")
                    img.save(vis_path)
                    vis_count += 1

        except Exception as e:
            import traceback
            print(f"Error processing batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Save COCO JSON
    print(f"Saving COCO format to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(f"{args.output}", 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nDone!")
    print(f"  Total images processed: {len(coco_data['images'])}")
    print(f"  Total nuclei: {nuclei_id}")
    print(f"  Output saved to: {args.output}")


if __name__ == '__main__':
    main()
