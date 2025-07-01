import glob, sys, os, time, json
import cv2
import openslide
import sqlite3
import warnings
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from histomicstk.features.compute_nuclei_features import compute_nuclei_features
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine
from multiprocessing import Pool

warnings.filterwarnings('ignore')

class WSI_Crop(Dataset):
    def __init__(self, wsi, coord_li, mag=40, savedir=None):
        self.mag = mag
        self.wsi = wsi
        self.coord_li = coord_li
        wsi_width, wsi_height = wsi.level_dimensions[0]
        self.wsi_width = wsi_width
        self.wsi_height = wsi_height
        self.savedir = savedir

    def __len__(self):
        return len(self.coord_li)

    def __getitem__(self, idx):
        coords = self.coord_li[idx]
        x_min = int(np.min(coords[:, 0]))
        x_max = int(np.max(coords[:, 0]))
        y_min = int(np.min(coords[:, 1]))
        y_max = int(np.max(coords[:, 1]))
        w = x_max - x_min
        h = y_max - y_min

        x_start = max(x_min-5, 0)
        x_end = min(x_max+5, self.wsi_width)
        y_start = max(y_min-5, 0)
        y_end = min(y_max+5, self.wsi_height)
        patch_w, patch_h = x_end - x_start, y_end - y_start

        coord_map = coords - np.array([x_start, y_start])
        if self.mag != 40:
            ratio = 40 / self.mag
            coord_map = coord_map * ratio
            rgb_img = self.wsi.read_region((x_start, y_start), 0, (x_end-x_start, y_end-y_start)).convert("RGB")
            rgb_img = rgb_img.resize((int(patch_w*ratio), int(patch_h*ratio)))
            rgb_img = np.array(rgb_img)
        else:
            rgb_img = self.wsi.read_region((x_start, y_start), 0, (x_end-x_start, y_end-y_start)).convert("RGB")
            rgb_img = np.array(rgb_img)

        nu_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        nu_mask = cv2.fillPoly(nu_mask, [coord_map.astype(np.int32)], 1)
        
        # For debugging purposes
        if self.savedir is not None:
            os.makedirs(self.savedir, exist_ok=True)
            nu_img = rgb_img.copy()
            contour = coord_map.astype(np.int32).reshape((-1, 1, 2))
            # Draw the contour on the image
            cv2.polylines(nu_img, [contour], isClosed=True, color=(0, 255, 0), thickness=1)
            Image.fromarray(nu_img).save(f'{self.savedir}/{idx}.png')

        return {
            'image': rgb_img,
            'mask': nu_mask,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
        }

def collate_fn(batch):
    return {
        'image': [item['image'] for item in batch],
        'mask': [item['mask'] for item in batch],
        'x_min': [item['x_min'] for item in batch],
        'x_max': [item['x_max'] for item in batch],
        'y_min': [item['y_min'] for item in batch],
        'y_max': [item['y_max'] for item in batch],
    }
    
# ignore histomicstk/features/compute_morphometry_features.py:147: RankWarning: Polyfit may be poorly conditioned

def process_nuclei_batch(rgb_img, nu_mask, nu_info):
    try:
        stains, _, _ = color_deconvolution_routine(rgb_img)
        htx = 255 - stains[..., 0]
        fdf = compute_nuclei_features(im_label=nu_mask.astype(np.uint8), im_nuclei=htx)
    except:
        return None
    fdf['score'] = nu_info['properties']['score']
    fdf['type'] = nu_info['properties']['classification']['name']
    fdf['class_id'] = nu_info['properties']['label']
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
    """
    Extracts nuclei features from whole slide images (WSI) and stores them in a SQLite database.

    Parameters:
    - datadir (str): Path to the directory containing raw WSI image files.
    - segdir (str): Path to the directory containing segmentation files.
    - start (int, optional): Start index for processing slides. Defaults to 0.
    - end (int, optional): End index for processing slides. If -1, processes all slides from the start index. If None, processes all slides starting from the start index. Defaults to -1.
    - mag (int, optional): Magnification level of the slide. Defaults to 40.
    - bs_size (int, optional): Batch size for nuclei feature extraction. Defaults to 128.
    - num_workers (int, optional): Number of workers for parallel processing. Defaults to 16.
    - min_num (int, optional): Minimum number of nuclei required for processing. Defaults to 16.
    - reverse (bool, optional): If True, processes slides in reverse order. Defaults to False.

    Returns:
    None
    """
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
        nu_file_path = f'{segdir}/{slide_id}/{slide_id}_merged.geojson'
        if not os.path.exists(nu_file_path):
            print(f'not found {slide_id}, skipped\n')
            continue

        with open(nu_file_path) as f:
            nu_geojson = json.load(f)

        if os.path.exists(f'{segdir}/{slide_id}/nuclei_feat.db'):
            # Check the file size
            file_size = os.path.getsize(f'{segdir}/{slide_id}/nuclei_feat.db')
            if file_size < 1 * 1024 * 1024:  # 1 MB in bytes
                print(f"File {segdir}/{slide_id}/nuclei_feat.db is less than 1MB. Removing it.")
                os.remove(f'{segdir}/{slide_id}/nuclei_feat.db')
                table_created = False
                os.makedirs(f'{segdir}/{slide_id}', exist_ok=True)
                conn = sqlite3.connect(f'{segdir}/{slide_id}/nuclei_feat.db')
                cursor = conn.cursor()
            else:
                table_created = True
                conn = sqlite3.connect(f'{segdir}/{slide_id}/nuclei_feat.db')
                cursor = conn.cursor()
                # Get the list of nuclei_id from the database
                query = "SELECT nuclei_id FROM nuclei_features"
                cursor.execute(query)

                nu_id_li = [elem['properties']['nuclei_id'] for elem in nu_geojson]
                nu_id_exists = set(row[0] for row in cursor.fetchall())  # Convert to a set for faster lookup
                # Filter out elements from nu_geojson that already exist in the database
                if len(set(nu_id_exists) ^ set(nu_id_li)) == 0:
                    print(f'skipped:{slide_id}\n')
                    continue
                else:
                    print(f'skipped {len(nu_id_exists)} nuclei')
                nu_geojson = [elem for elem in nu_geojson if elem['properties']['nuclei_id'] not in nu_id_exists]
        else:
            table_created = False
            os.makedirs(f'{segdir}/{slide_id}', exist_ok=True)
            conn = sqlite3.connect(f'{segdir}/{slide_id}/nuclei_feat.db')
            cursor = conn.cursor()

        coord_li = [np.array(elem['geometry']['coordinates'][0]) for elem in nu_geojson]

        wsi = openslide.open_slide(f'{datadir}/{slide_id}.svs')
        nu_dataset = WSI_Crop(wsi, coord_li, mag=mag)
        nu_dataloader = DataLoader(nu_dataset, batch_size=bs_size, 
                                num_workers=num_workers,
                                shuffle=False, 
                                collate_fn=collate_fn, 
                                drop_last=False)

        # bs_len = (len(nu_geojson) + bs_size - 1) // bs_size  # Calculate the number of batches
        bs_len = len(nu_dataloader)
        for bs_idx, nu_batches in enumerate(tqdm(nu_dataloader)):
            nu_img_li = nu_batches['image']
            nu_mask_li = nu_batches['mask']
            nu_geojson_li = nu_geojson[bs_idx*bs_size:(bs_idx+1)*bs_size]
            for nu_idx in range(len(nu_geojson_li)):
                nu_info = nu_geojson_li[nu_idx]
                nu_info['x_min'] = nu_batches['x_min'][nu_idx]
                nu_info['y_min'] = nu_batches['y_min'][nu_idx]
                nu_info['x_max'] = nu_batches['x_max'][nu_idx]
                nu_info['y_max'] = nu_batches['y_max'][nu_idx]

            with Pool(num_workers) as pool:
                results = pool.starmap(process_nuclei_batch, [(nu_img_li[i], nu_mask_li[i], nu_geojson_li[i]) for i in range(len(nu_img_li))])

            fdf_li = [res for res in results if res is not None]
            if len(fdf_li) == 0:
                print(f'skipped batch {bs_idx} of {bs_len}')
                continue
            fdf = pd.concat(fdf_li, axis=0)
            fdf.columns = [col.replace('.', '_') for col in fdf.columns]

            if not table_created:
                columns = fdf.columns
                # Infer column types from DataFrame dtypes
                dtype_mapping = {
                    'int64': 'INTEGER',
                    'float64': 'REAL',
                    'object': 'TEXT',
                    'bool': 'INTEGER',
                    'datetime64[ns]': 'TEXT'
                }
                column_definitions = ", ".join([
                    f"{col} {dtype_mapping.get(str(fdf[col].dtype), 'TEXT')}" for col in fdf.columns
                ])
                cursor.execute(f"CREATE TABLE IF NOT EXISTS nuclei_features ({column_definitions})")
                conn.commit()
                table_created = True
            else:
                fdf.to_sql('nuclei_features', conn, if_exists='append', index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="path to folder containing raw wsi image files")
    parser.add_argument("--segdir", help="path to folder containing segmentation files")
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--mag', type=int, default=40, help='magnification of the slide')
    parser.add_argument('--reverse', action='store_true', default=False, help='reverse the order of slide ids')
    parser.add_argument('--bs_size', type=int, default=1024, help='batch size for nuclei feature extraction')
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
