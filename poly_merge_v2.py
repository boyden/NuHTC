import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import io

from tqdm import tqdm
from pycocotools import mask as maskUtils
from argparse import ArgumentParser
# from line_profiler import LineProfiler

# Improve Speed
# Faster than V1 but will be memory consumptive
# Total time: 315.672 s

def mask_nms(masks, pred_scores, thr=0.9, min_area=None):
    """https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L98

    Returns:
        array: not kept indice.
    """
    sort_idx = np.argsort(pred_scores)[::-1]
    mask_len = len(masks)
    tmp_masks = np.array(masks)[sort_idx]
    mask_iou = maskUtils.iou(tmp_masks.tolist(), tmp_masks.tolist(), [0] * mask_len)

    keep_idx = np.ones(mask_len, dtype=np.uint8)
    for i in range(mask_len):
        if not keep_idx[i]:
            continue
        for j in range(i + 1, mask_len):
            if not keep_idx[j]:
                continue
            tmp_iou = mask_iou[i, j]
            # tmp_iou = maskUtils.iou([tmp_masks[i]], [tmp_masks[j]], [0])[0][0]
            if tmp_iou > thr:
                keep_idx[j] = 0
    return sort_idx[keep_idx==0]

def main():
    args = parse_args()
    datadir = args.datadir
    contour_name = args.contour_name
    conn = sqlite3.connect(f'{datadir}/{contour_name}.db')
    c = conn.cursor()
    c.execute('''SELECT id, xmin, xmax, ymin, ymax FROM rtree;''')
    item_li = c.fetchall()
    item_id_li = np.array([int(elem[0]) for elem in item_li])
    reverse_item_id_li = np.zeros(max(item_id_li)+1, dtype=np.int32)
    for i, item in enumerate(item_id_li):
        reverse_item_id_li[item] = i
    keep_idx = np.ones(len(item_li), dtype=np.uint8)
    contour_df = pd.read_sql_query(f'''
                            SELECT id, xmin, xmax, ymin, ymax, coords_x, coords_y, score, keep FROM contour;''', conn)
    contour_df['x'] = contour_df['coords_x'].apply(lambda x:[int(elem.strip()) for elem in x.strip().split(',')])
    contour_df['y'] = contour_df['coords_y'].apply(lambda x:[int(elem.strip())  for elem in x.strip().split(',')])

    for item_idx, item in enumerate(tqdm(item_li)):
        if keep_idx[item_idx] == 0:
            continue
        idx, xmin, xmax, ymin, ymax = item
        c.execute(f'''
                      SELECT id FROM rtree
                      WHERE xmin<{xmax} AND xmax>{xmin}
                      AND ymin<{ymax}  AND ymax>{ymin};''') # This is if all 4 corners of BB are inside another nucleus
        overlap_ids = c.fetchall()
        if len(overlap_ids) <= 1:
            continue
        overlap_ids = [elem[0] for elem in overlap_ids]

        seg_df = contour_df.loc[contour_df['id'].isin(overlap_ids)]
        y_offset, x_offset = seg_df['ymin'].values.min(), seg_df['xmin'].values.min()
        h, w = seg_df['ymax'].values.max() - y_offset, seg_df['xmax'].values.max() - x_offset

        mask_len = len(seg_df)
        masks_li = [np.concatenate([[seg_df.iloc[k]['x']-x_offset], [seg_df.iloc[k]['y']]-y_offset], axis=0).transpose((1, 0)).flatten()
                     for k in range(mask_len)]
        masks_li = maskUtils.frPyObjects(masks_li, h, w)
        nms_idx = mask_nms(masks_li, seg_df['score'].values, thr=0.05, min_area=None)
        if len(nms_idx) == 0:
            continue
        keep_idx[reverse_item_id_li[seg_df['id'].values[nms_idx]]] = 0

    keep_data = tuple(item_id_li[keep_idx==0].tolist())
    c.execute(f'''DELETE FROM contour WHERE id IN {keep_data};''',)
    seg_df = pd.read_sql_query(f'''SELECT * FROM contour;''', conn)
    seg_df.to_csv(f'{datadir}/{contour_name}.csv', index=False)
    conn.commit()
    conn.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="path to folder containing raw wsi image files")
    parser.add_argument("contour_name", help="contour name")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
    # contour_name = 'contour_384'
    # datadir = '/home/bao/code/NuHTC/demo/wsi_infer/TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724'

    # python poly_merge_v2.py /home/bao/code/NuHTC/demo/wsi_infer/TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724 contour_256


