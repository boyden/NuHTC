import glob, sys, os, time, json, warnings
import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.coco import maskUtils
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from histomicstk.features.compute_nuclei_features import compute_nuclei_features
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine
from skimage.transform import resize

warnings.filterwarnings('ignore')


# ignore histomicstk/features/compute_morphometry_features.py:147: RankWarning: Polyfit may be poorly conditioned

def annToMask(ann_li):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = [ann['segmentation'] for ann in ann_li]
    m = maskUtils.decode(rle)
    return m


def extract_feat(datadir, start, end, min_num=16, reverse=False, patch_size=512):
    slide_id_li = glob.glob(f'{datadir}/*')
    slide_id_li = [os.path.basename(slide_id) for slide_id in slide_id_li]
    slide_id_li.sort()
    if reverse:
        slide_id_li = slide_id_li[::-1]
    if end is not None:
        slide_id_li = slide_id_li[start:end]
    else:
        slide_id_li = slide_id_li[start:]
    for slide_id in slide_id_li:
        print(f'\nprocess: {slide_id}')
        coco_file_path = f'{datadir}/{slide_id}/coco_nuclei.json'
        if not os.path.exists(coco_file_path):
            print(f'skipped:{slide_id}\n')
            continue

        coco_file = COCO(coco_file_path)
        cat_info = coco_file.cats
        img_id_li_ttp = [img_id for img_id in coco_file.imgs.keys() if coco_file.imgs[img_id]['n_objects'] >= min_num]

        if os.path.exists(f'{datadir}/nuclei_feat/{slide_id}/nuclei_feat.csv'):
            feat_exists = pd.read_csv(f'{datadir}/nuclei_feat/{slide_id}/nuclei_feat.csv', index_col=0)
            img_id_exists = feat_exists.loc[feat_exists['img_objs'] >= min_num, 'img_id'].values
            if len(set(img_id_exists) ^ set(img_id_li_ttp)) == 0:
                print(f'skipped:{slide_id}\n')
                continue

            img_id_li_ttp = list(set(img_id_li_ttp) - set(img_id_exists))
            ifeatures = feat_exists
        else:
            os.makedirs(f'{datadir}/nuclei_feat/{slide_id}', exist_ok=True)
            ifeatures = None

        img_info_li = coco_file.loadImgs(img_id_li_ttp)

        for idx, img_id in enumerate(tqdm(img_id_li_ttp)):

            imgInfo = img_info_li[idx]
            w = imgInfo['width']
            filename = imgInfo["file_name"]
            if w != patch_size:
                rgb_img = np.array(Image.open(f'{datadir}/imgs/{slide_id}/{filename}').resize((patch_size, patch_size)))
            else:
                rgb_img = np.array(Image.open(f'{datadir}/imgs/{slide_id}/{filename}'))
            stains, _, _ = color_deconvolution_routine(rgb_img)
            htx = 255 - stains[..., 0]

            annt_id_li = coco_file.getAnnIds(img_id)
            annt_li = coco_file.loadAnns(annt_id_li)
            rle_li = [ann['segmentation'] for ann in annt_li]
            ann_cls_id = np.array([cat_info[ann['category_id']]['name'] for ann in annt_li])
            ann_mask = maskUtils.decode(rle_li).transpose((2, 0, 1))
            mask_id_list = np.array(range(1, len(ann_mask) + 1))
            ann_mask = np.max(ann_mask * mask_id_list.reshape((-1, 1, 1)), axis=0)
            if w != patch_size:
                ann_mask = resize(ann_mask, (patch_size, patch_size), order=0, preserve_range=True)

            try:
                fdf = compute_nuclei_features(im_label=ann_mask.astype(np.uint8), im_nuclei=htx)
                # To avoid the situation that one instance may cover another instance
                annt_id_li = np.array(annt_id_li)
                lb_idx = fdf['Label'].values.astype(np.int64)-1
                df_idx = annt_id_li[lb_idx]
                ann_cls_id = ann_cls_id[lb_idx]
                fdf.index = df_idx

            except:
                df_idx = annt_id_li
                fdf = pd.DataFrame(index=df_idx)

            fdf['cell_type'] = ann_cls_id
            fdf['img_id'] = img_id
            fdf['img_type'] = imgInfo['type']
            fdf['img_objs'] = len(df_idx)
            fdf['file_name'] = filename

            if ifeatures is None and len(fdf.columns) != 0:
                ifeatures = fdf
            else:
                ifeatures = pd.concat([ifeatures, fdf], axis=0)

            if (idx + 1) % 10000 == 0 or idx + 1 == len(img_id_li_ttp):
                ifeatures.to_csv(f'{datadir}/nuclei_feat/{slide_id}/nuclei_feat.csv', mode='w')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="path to folder containing raw wsi image files")
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--min_num', type=int, default=8, help='exclude images contain less than min number of nuclei')
    parser.add_argument('--patch_size', type=int, default=512, help='image patch size, should be equal to the size of input image(40X) during inference')
    parser.add_argument('--reverse', action='store_true', default=False, help='reverse the order of slide ids')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    extract_feat(args.datadir, args.start, args.end, min_num=args.min_num, reverse=args.reverse, patch_size=args.patch_size)
