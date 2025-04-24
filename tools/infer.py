import asyncio
import glob
import os
import sys
from argparse import ArgumentParser
proj_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)

from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot

from nuhtc.apis.inference import init_detector, save_result
from nuhtc.utils import patch_config

# python ./tools/infer.py demo/imgs models/htc_lite_PanNuke_infer.py models/pannuke.pth --out demo/imgs_infer
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.35, help="bbox score threshold"
    )
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.CLASSES = ('T', 'I', 'C', 'D', 'E')
    imgs = glob.glob(f"{args.img}/*png")
    for img in imgs:
        # test a single image
        result = inference_detector(model, img)
        # show the results
        if args.output is None:
            show_result_pyplot(model, img, result, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, result, score_thr=args.score_thr, out_file=out_file_path, font_size=6, thickness=1
            )

if __name__ == '__main__':
    main()
