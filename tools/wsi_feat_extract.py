import glob, sys, os, time, json
import cv2
import openslide
import warnings
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from histomicstk.features.compute_nuclei_features import compute_nuclei_features
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine
from multiprocessing import Pool

warnings.filterwarnings('ignore')

# ignore histomicstk/features/compute_morphometry_features.py:147: RankWarning: Polyfit may be poorly conditioned

def process_nuclei_batch(rgb_img, nu_mask, nu_info):
    stains, _, _ = color_deconvolution_routine(rgb_img)
    htx = 255 - stains[..., 0]
    try:
        fdf = compute_nuclei_features(im_label=nu_mask.astype(np.uint8), im_nuclei=htx)
    except:
        return None
    for k in ['label', 'score']:
        if k in nu_info['properties']:
            fdf[k] = nu_info['properties'][k]
    fdf['type'] = nu_info['properties']['classification']['name']
    fdf['nuclei_id'] = nu_info['properties']['nuclei_id']
    fdf[['x_min', 'y_min', 'x_max', 'y_max']] = [nu_info['x_min'], nu_info['y_min'], nu_info['x_max'], nu_info['y_max']]
    return fdf

def extract_feat(datadir, segdir, 
            start=0, 
            end=-1, 
            mag=40,
            bs_size=128,
            num_workers=16,
            min_num=16, 
            reverse=False):
    slide_id_li = glob.glob(f'{datadir}/*')
    slide_id_li = [os.path.basename(slide_id) for slide_id in slide_id_li]
    slide_id_li = [os.path.splitext(slide_id)[0] for slide_id in slide_id_li]
    slide_id_li.sort()
    if reverse:
        slide_id_li = slide_id_li[::-1]
    if end is not None:
        slide_id_li = slide_id_li[start:end]
    else:
        slide_id_li = slide_id_li[start:]
    for slide_id in slide_id_li:
        print(f'\nprocess: {slide_id}')
        wsi = openslide.open_slide(f'{datadir}/{slide_id}.svs')
        wsi_width, wsi_height = wsi.level_dimensions[0]

        nu_file_path = f'{segdir}/{slide_id}/{slide_id}_merged.geojson'
        if not os.path.exists(nu_file_path):
            print(f'skipped:{slide_id}\n')
            continue

        with open(nu_file_path) as f:
            nu_geojson = json.load(f)
        
        if os.path.exists(f'{segdir}/{slide_id}/nuclei_feat.csv'):
            print(f'load existing features: {slide_id}')
            nu_id_li = [elem['properties']['nuclei_id'] for elem in nu_geojson]
            nu_id_li_ttp = nu_id_li
            feat_exists = pd.read_csv(f'{segdir}/{slide_id}/nuclei_feat.csv')
            nu_id_exists = feat_exists['nuclei_id'].values
            
            if len(set(nu_id_exists) ^ set(nu_id_li_ttp)) == 0:
                print(f'skipped:{slide_id}\n')
                continue
            else:
                print(f'skipped {len(nu_id_exists)} nuclei')
            nu_geojson = [elem for elem in nu_geojson if elem['properties']['nuclei_id'] not in nu_id_exists]
            ifeatures = feat_exists
        else:
            os.makedirs(f'{datadir}/nuclei_feat/{slide_id}', exist_ok=True)
            ifeatures = None

        bs_len = (len(nu_geojson) + bs_size - 1) // bs_size  # Calculate the number of batches
        for bs_idx in tqdm(range(bs_len)):
            batches = nu_geojson[bs_idx:bs_idx+bs_size]
            nu_img_li = []
            nu_mask_li = []
            nu_geojson_li = []
            for nu_idx in range(len(batches)):
                nu_info = batches[nu_idx]
                coords = np.array(nu_info['geometry']['coordinates'][0])
                x_min = int(np.min(coords[:, 0]))
                x_max = int(np.max(coords[:, 0]))
                y_min = int(np.min(coords[:, 1]))
                y_max = int(np.max(coords[:, 1]))
                w = x_max - x_min
                h = y_max - y_min
                nu_info['x_min'] = x_min
                nu_info['y_min'] = y_min
                nu_info['x_max'] = x_max
                nu_info['y_max'] = y_max

                x_start = max(x_min-5, 0)
                x_end = min(x_max+5, wsi_width)
                y_start = max(y_min-5, 0)
                y_end = min(y_max+5, wsi_height)
                patch_w, patch_h = x_end - x_start, y_end - y_start

                coord_map = coords - np.array([x_start, y_start])
                if mag != 40:
                    ratio = 40 / mag
                    coord_map = coord_map * ratio
                    rgb_img = wsi.read_region((x_start, y_start), 0, (x_end-x_start, y_end-y_start)).convert("RGB")
                    rgb_img = rgb_img.resize((int(patch_w*ratio), int(patch_h*ratio)))
                    rgb_img = np.array(rgb_img)
                else:
                    rgb_img = wsi.read_region((x_start, y_start), 0, (x_end-x_start, y_end-y_start)).convert("RGB")
                    rgb_img = np.array(rgb_img)

                nu_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
                nu_mask = cv2.fillPoly(nu_mask, [coord_map.astype(np.int32)], 1)
                nu_img_li.append(rgb_img)
                nu_mask_li.append(nu_mask)
                nu_geojson_li.append(nu_info)

            with Pool(num_workers) as pool:
                results = pool.starmap(process_nuclei_batch, [(nu_img_li[i], nu_mask_li[i], nu_geojson_li[i]) for i in range(len(batches))])
            
            fdf_li = [res for res in results if res is not None]
            if len(fdf_li) == 0:
                print(f'skipped batch {bs_idx} of {bs_len}')
                continue
            fdf = pd.concat(fdf_li, axis=0)
            if ifeatures is None:
                ifeatures = fdf
            else:
                ifeatures = pd.concat([ifeatures, fdf], axis=0)

            if (bs_idx + 1)*bs_size % 10000 == 0 or bs_idx + 1 == len(batches):
                ifeatures.to_csv(f'{segdir}/{slide_id}/nuclei_feat.csv', mode='w', index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="path to folder containing raw wsi image files")
    parser.add_argument("--segdir", help="path to folder containing segmentation files")
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--mag', type=int, default=40, help='magnification of the slide')
    parser.add_argument('--reverse', action='store_true', default=False, help='reverse the order of slide ids')
    parser.add_argument('--bs_size', type=int, default=512, help='batch size for nuclei feature extraction')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for parallel processing')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    extract_feat(args.datadir, args.segdir, args.start, args.end, 
                mag=args.mag,
                bs_size=args.bs_size,
                num_workers=args.num_workers,
                reverse=args.reverse)
