'''
Note: Boundary/margin cells already removed, input to merge_overlap() is cleaned cells list / df

@input --datadir: Path to folder containing geojson file
@input --geojson_name: Geojson file name
@input --overlap_threshold: Area overlap percentage threshold to be removed
@input --merge_strategy: Whether to keep the cell with highest probability or largest area, specify 'probability' or 'area' 
@input --uniform_classification: Whether to classify all cells uniformly and represent as the same color (yellow)

    Example call:
    python tools/nuclei_merge.py \
    --datadir demo/wsi_infer/TCGA-AC-A2FK \
    --geojson_name TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724 \
    --overlap_threshold 0.05 \
    --merge_strategy probability

@output processed geojson file
'''

import os
import sys
import time
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from shapely import strtree
from shapely.geometry import MultiPolygon, Polygon
from shapely.errors import ShapelyDeprecationWarning
from collections import deque

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


## Base code: CellViT++, adapted for NuHTC
def merge_overlap(cleaned_edge_cells: pd.DataFrame, overlap_threshold=0.01, merge_strategy='probability', uniform_classification=False) -> pd.DataFrame:
    
    """
    Removes overlapping cells from provided DataFrame.
    
    Args
    ----------
        cleaned_edge_cells (pd.DataFrame) : DataFrame that should be cleaned
        overlap_threshold (float) : Area overlap percentage threshold to be removed, default 0.01
        merge_strategy (str): Whether to keep the cell with highest probability or largest area, specify 'probability' or 'area'
        uniform_classification (store_true bool) : Whether to classify all cells uniformly and represent as the same color (yellow). Default (if unspecified) False
    Returns
    ----------
        pd.DataFrame : Cleaned DataFrame
        saved geojson file
    """
    
    start_time = time.time()
    cleaned_edge_cells = pd.DataFrame(cleaned_edge_cells)
    merged_cells = cleaned_edge_cells.copy()
    merged_cells['score'] = merged_cells['properties'].apply(lambda x: x.get('score', 0))
    merged_cells = merged_cells.sort_values(by='score', ascending=False).reset_index(drop=True)

    print('Starting overlap removal...')

    for iteration in range(20):
        print(f'Iteration {iteration}')
        poly_list = []
        poly_uid_map = {} # key is poly, value is uid
        uid_poly_map = {} # key is uid, value is poly

        for idx, cell_info in tqdm(merged_cells.iterrows(), total=len(merged_cells)):
            
            poly = Polygon(cell_info['geometry']['coordinates'][0])
            
            if not poly.is_valid:
                # print("Found invalid polygon - Fixing with buffer 0")
                multi = poly.buffer(0)
                if isinstance(multi, MultiPolygon):
                    multi_polys = list(multi.geoms)  # Convert to iterable list
                    if len(multi_polys) > 1:
                        poly_idx = np.argmax([p.area for p in multi_polys])
                        poly = multi_polys[poly_idx]
                        poly = Polygon(poly)
                    else:
                        poly = multi_polys[0]
                        poly = Polygon(poly)
                else:
                    poly = Polygon(multi)
                    
            poly_list.append(poly)
            poly_uid_map[poly] = idx  # store uid 
            uid_poly_map[idx] = poly

        # Use an strtree for fast querying
        tree = strtree.STRtree(poly_list)

        merged_idx = deque()
        iterated_cells = set()
        overlaps = 0

        print(f'len of poly_uid_map: {len(poly_uid_map)}')
        for query_poly in tqdm(poly_list):
            query_uid = poly_uid_map[query_poly]
            
            if query_uid not in iterated_cells:
                intersected_polygons = tree.query(query_poly)  # Contains a self-intersection

                if (len(intersected_polygons) > 1):  # We have more at least one intersection with another cell
                    submergers = []  # All cells that overlap with query
                   
                    for inter_uid in intersected_polygons:
                        if inter_uid not in uid_poly_map:
                            continue  # This polygon was removed or not tracked — skip it
                        inter_poly = uid_poly_map[inter_uid]
                        # print(f'inter_poly: {inter_poly}')
                        if (
                            inter_uid != query_uid
                            and inter_uid not in iterated_cells
                        ):
                            inst_intersect = query_poly.intersection(inter_poly).area
                            inst_iou = inst_intersect / (query_poly.area + inter_poly.area - inst_intersect)
                            if inst_iou > overlap_threshold:
                                overlaps = overlaps + 1
                                submergers.append(inter_poly)
                                iterated_cells.add(inter_uid)
                    # Catch block: empty list -> some cells are touching, but not overlapping strongly enough
                    if len(submergers) == 0:
                        merged_idx.append(query_uid)
                    # Merging strategy
                    else:
                        if merge_strategy == 'probability':
                            # scores = []
                            # for i, p in enumerate(submergers):
                            #     uid = poly_uid_map[p]
                            #     props = merged_cells.loc[uid, 'properties']
                            #     score = props.get('score', 0)
                            #     scores.append(score)
                            #     print(f'Submerger {i}: UID = {uid}, Score = {score:.4f}')
                            # selected_poly_index = int(np.argmax(scores))
                            # selected_poly = submergers[selected_poly_index]
                            # selected_uid = poly_uid_map[selected_poly]
                            merged_idx.append(query_uid)

                        elif merge_strategy == 'area':
                            selected_poly_index = np.argmax(np.array([p.area for p in submergers]))
                            selected_poly = submergers[selected_poly_index]
                            selected_uid = poly_uid_map[selected_poly]
                            merged_idx.append(selected_uid)
                        else:
                            raise ValueError(f"Invalid merge strategy: {merge_strategy}. Use 'probability' or 'area'.")
                else:
                    # No intersection, just add
                    merged_idx.append(query_uid)
                iterated_cells.add(query_uid)

        print(f"Iteration {iteration}: Found overlap of # cells: {overlaps}")
        if overlaps == 0:
            print("Found all overlapping cells")
            break
        elif iteration == 20:
            print(f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, we stop iterations now. Please raise an issue in git or increase number of iterations.")

        ## This does not reset index (keeps OG row indices e.g. 1,3,5), can help track original rows by index.
        ## In later iterations, any use of .loc or indexing using idx still corresponds to the original rows in cleaned_edge_cells
        merged_cells = merged_cells.loc[
            merged_cells.index.isin(merged_idx)
        ].reset_index(drop=True).copy()

    end_time = time.time()
    print(f"Cell overlap removal elapsed time: {end_time - start_time:.4f} seconds")
    
    return merged_cells.sort_index()


def main():
    
    args = parse_args()
    datadir = args.datadir
    geojson_name = args.geojson_name
    with open(f'{datadir}/{geojson_name}.geojson', 'r') as f:
        data = json.load(f) # List

    # Function call
    data_processed = merge_overlap(data, 
                            overlap_threshold=args.overlap_threshold, 
                            merge_strategy=args.merge_strategy, 
                            uniform_classification=args.uniform_classification)
    if args.output_name:
        output_path = f"{datadir}/{args.output_name}.geojson"
    else:
        output_path = f"{datadir}/{geojson_name}_merged.geojson"

    # Convert each row into a GeoJSON Feature
    print('Converting to geojson...')
    features = []
    for _, row in data_processed.iterrows():
        geometry = row["geometry"]
        props = row["properties"]
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": props
        }
        if args.uniform_classification:
            row["properties"]["classification"]["name"] = "uniform"
            row["properties"]["classification"]["color"] = [255, 255, 0] # Yellow

        features.append(feature)
        
    print('Saving as flat list of geojson features...')
    with open(output_path, "w") as f:
        json.dump(features, f)
    
    print('Merge and save complete.')

    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--datadir", help="path to folder containing geojson file")
    parser.add_argument("--geojson_name", help="geojson file name")
    parser.add_argument("--output_name", default=None, type=str, help="output geojson file name")
    parser.add_argument("--overlap_threshold", type=float, default=0.01, help="area overlap percentage threshold to be removed")
    parser.add_argument("--merge_strategy", default="probability", help="whether to keep the cell with highest probability or largest area, specify 'probability' or 'area'")
    parser.add_argument("--uniform_classification", action="store_true", help="whether to classify all cells uniformly and represent as the same color (yellow)")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

    
