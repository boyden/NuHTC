"""Feature extraction script for NuHTC model."""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# Set max image pixels to avoid decompression bomb warnings
Image.MAX_IMAGE_PIXELS = 100000000000

# Add project paths to sys.path
sys.path.insert(0, 'thirdparty/mmdetection')
sys.path.insert(0, '.')

from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from nuhtc.apis.inference import init_detector
from nuhtc.utils import patch_config

from wsi_core.WholeSlideImage import Dataset_All_Bags, Whole_Slide_Bag_FP
from wsi_core.wsi_utils import save_hdf5

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def model_feat(model, imgs):
    """Extract features from image(s) using the detector model.

    Args:
        model (nn.Module): The loaded detector model.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.

    Returns:
        np.ndarray: Extracted features from the model.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        features_lvl = model.extract_feat(data['img'][0])
        feat_li = []
        for lvl in range(len(features_lvl)):
            feat_li.append(features_lvl[lvl].mean(dim=(2, 3)))
        features = torch.cat(feat_li, dim=1)
        features = features.cpu().numpy()
    return features


def collate_features(batch):
    """Collate function for feature extraction batches.

    Args:
        batch: List of (image, coordinates) tuples.

    Returns:
        list: [concatenated images, stacked coordinates]
    """
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def compute_w_loader(
    file_path,
    output_path,
    wsi,
    model,
    batch_size=8,
    verbose=0,
    print_every=20,
    pretrained=True,
    transform=False,
    custom_downsample=1,
    target_patch_size=-1,
    stain_norm=False,
):
    """Compute features for a whole slide image using a data loader.

    Args:
        file_path (str): Directory of bag (.h5 file).
        output_path (str): Directory to save computed features (.h5 file).
        wsi: Whole slide image object.
        model: PyTorch model for feature extraction.
        batch_size (int): Batch size for computing features in batches.
        verbose (int): Level of feedback.
        print_every (int): Print progress every N batches.
        pretrained (bool): Use weights pretrained on imagenet.
        transform (bool): Whether to apply transforms.
        custom_downsample (int): Custom defined downscale factor of image patches.
        target_patch_size (int): Custom defined, rescaled image size before embedding.
        stain_norm (bool): Whether to apply stain normalization.

    Returns:
        str: Path to the output file.
    """
    dataset = Whole_Slide_Bag_FP(
        file_path=file_path,
        wsi=wsi,
        pretrained=pretrained,
        custom_downsample=custom_downsample,
        target_patch_size=target_patch_size,
        transform=transform,
        stain_norm=stain_norm,
    )
    dataset[0]  # Initialize dataset
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print(
                    'batch {}/{}, {} files processed'.format(
                        count, len(loader), count * batch_size
                    )
                )
            batch = [elem.numpy() for elem in batch]
            features = model_feat(model, batch)
            asset_dict = {'features': features, 'coords': coords.numpy()}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file"
    )
    parser.add_argument('--data_h5_dir', type=str, default=None)
    parser.add_argument('--data_slide_dir', type=str, default=None)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument(
        '--no_auto_skip', default=False, action='store_true'
    )
    parser.add_argument(
        '--stain_norm', default=False, action='store_true'
    )
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)
    return parser.parse_args()


# Example command:
# CUDA_VISIBLE_DEVICES=1 python extract_features_nuhtc.py \
#     --config /home/bao/code/NuHTC/models/htc_lite_PanNuke_generic_infer_20.py \
#     --checkpoint /home/bao/code/NuHTC/models/ema_pannuke_generic.pth \
#     --data_h5_dir /data/data1/WSI/TCGA_BRCA/TCGA_BRCA_FEAT \
#     --data_slide_dir /data/data1/WSI/TCGA_BRCA/WSI \
#     --slide_ext .svs \
#     --csv_path /data/data1/WSI/TCGA_BRCA/TCGA_BRCA_FEAT/he_process_list.csv \
#     --feat_dir /data/data1/WSI/TCGA_BRCA/TCGA_BRCA_FEAT/feat_nuhtc_norm \
#     --target_patch_size 256 \
#     --batch_size 16 \
#     --stain_norm


if __name__ == '__main__':
    args = parse_args()

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError("CSV path must be provided")

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print('loading model checkpoint')
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=device)
    model.rpn_head = None
    model.roi_head = None
    torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(
            args.data_slide_dir, slide_id + args.slide_ext
        )
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        try:
            wsi = openslide.open_slide(slide_file_path)
            output_file_path = compute_w_loader(
                h5_file_path,
                output_path,
                wsi,
                model=model,
                batch_size=args.batch_size,
                verbose=1,
                print_every=100,
                transform=False,
                custom_downsample=args.custom_downsample,
                target_patch_size=args.target_patch_size,
                stain_norm=args.stain_norm,
            )
            time_elapsed = time.time() - time_start
            print(
                '\ncomputing features for {} took {} s'.format(
                    output_file_path, time_elapsed
                )
            )
            file = h5py.File(output_file_path, "r")

            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(
                features,
                os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'),
            )
        except Exception as e:
            print('ERROR:', bag_name)
            print(f'Exception: {e}')
