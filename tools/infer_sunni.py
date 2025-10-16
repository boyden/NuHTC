## Refactored infer.py script for NuHTC (Last updated: 09/13/2025)
## Input images: .png
## Directly outputs COCO json format!

import os
import cv2
import sys
import glob
import json
import mmcv
import asyncio
import numpy as np
from argparse import ArgumentParser
from pycocotools import mask as maskUtils
proj_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
from nuhtc.apis.inference import init_detector, save_result
from nuhtc.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--gt_coco_path", help="Ground truth coco json file")
    parser.add_argument("--score-thr", type=float, default=0.35, help="bbox score threshold")
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
    args = parser.parse_args()
    return args


def parse_results(result, img_h, img_w, score_thr, image_id, ann_start=1):
    
    """ Convert mmdet result to COCO-style annotations (source: infer_wsi.py) """

    bbox_results, segm_results = result  # unpack (bbox, segm)
    annotations = []
    ann_id = ann_start

    for cls_id, (bboxes, segms) in enumerate(zip(bbox_results, segm_results)):
        for i in range(len(bboxes)):
            score = bboxes[i][-1]
            if score < score_thr:
                continue

            bbox = bboxes[i][:4]
            segm = segms[i]

            rle = maskUtils.encode(np.asfortranarray(segm.astype(np.uint8)))
            # rle["counts"] = rle["counts"].decode("ascii")  # json serializable
            rle["counts"] = rle["counts"].decode("utf-8")
            coco_bbox = maskUtils.toBbox(rle).tolist()
            area = float(maskUtils.area(rle))

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls_id + 1),  # categories are 1-indexed
                "bbox": [float(x) for x in coco_bbox],
                "area": area,
                "segmentation": rle,
                "iscrowd": 0,
                "score": float(score),
            }
            annotations.append(ann)
            ann_id += 1

    return annotations, ann_id


def main():
    
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    # Load COCO template (images + categories)
    with open(args.gt_coco_path, "r") as f:
        gt_coco = json.load(f)
    images = gt_coco["images"]
    categories = gt_coco["categories"]

    # Index images by file name
    image_lookup = {img["file_name"]: img["id"] for img in images}

    # Collect images to run inference on
    imgs = sorted(glob.glob(os.path.join(args.img, "*.png")))

    annotations = []
    ann_id = 1

    for img_path in imgs:
        file_name = os.path.basename(img_path)
        if file_name not in image_lookup:
            print(f"Warning: {file_name} not in COCO images list, skipping.")
            continue

        image_id = image_lookup[file_name]
        img_h, img_w, _ = cv2.imread(img_path).shape
        result = inference_detector(model, img_path)

        anns, ann_id = parse_results(result, img_h, img_w, args.score_thr, image_id, ann_id)
        annotations.extend(anns)

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(args.save_path, "w") as f:
        json.dump(coco_json, f)
    print(f"Saved segmentation results to {args.save_path}")

    
if __name__ == '__main__':
    main()

