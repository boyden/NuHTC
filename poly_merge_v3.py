'''
Note: Boundary/margin cells already removed, input to _remove_overlap() is cleaned cells list / df

@input datadir: Path to folder containing geojson file
@input geojson_name: Geojson file name
@input overlap_threshold: Area overlap percentage threshold to be removed

    Example call:
    !python poly_merge_v3.py \
    /home/sul084/immune-decoder/segmentation/NuHTC/demo/wsi_infer/BL-13 \
    BL-13-K40642

@output processed geojson file

Last updated: 5/20/2025

TODO: 
    - Add cell properties to saved geojson (including cell type, cell probabilities)
    - Check 0.01 vs 0.05 threshold
'''

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import io

from tqdm import tqdm
from pycocotools import mask as maskUtils
from argparse import ArgumentParser
from shapely import strtree
from shapely.geometry import MultiPolygon, Polygon
import warnings
from shapely.errors import ShapelyDeprecationWarning
from typing import List
import time
import json
from collections import deque

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


## Base code: CellViT++, adapted for NuHTC
def _remove_overlap(cleaned_edge_cells: pd.DataFrame, overlap_threshold) -> pd.DataFrame:
    
    """
    Removes overlapping cells from provided DataFrame.
    
    Args
    ----------
        cleaned_edge_cells (pd.DataFrame) : DataFrame that should be cleaned
        overlap_threshold (float) : Area overlap percentage threshold to be removed, default 0.01
    Returns
    ----------
        pd.DataFrame : Cleaned DataFrame
        saved geojson file
    """
    
    start_time = time.time()
    cleaned_edge_cells = pd.DataFrame(cleaned_edge_cells)
    merged_cells = cleaned_edge_cells

    print('Starting overlap removal...')

    for iteration in range(20):
        print(f'Iteration {iteration}')
        poly_list = []
        poly_uid_map = {} # key is poly, value is uid
        uid_poly_map = {} # key is uid, value is poly
        
        for idx, cell_info in tqdm(merged_cells.iterrows()):
            
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
                            # print(f'This polygon with uid {inter_uid} was removed — skip.') # Maybe can log this!
                            continue  # This polygon was removed or not tracked — skip it
                        inter_poly = uid_poly_map[inter_uid]
                        # print(f'inter_poly: {inter_poly}')
                        if (
                            inter_uid != query_uid
                            and inter_uid not in iterated_cells
                        ):
                            if (
                                query_poly.intersection(inter_poly).area
                                / query_poly.area
                                > overlap_threshold
                                or query_poly.intersection(inter_poly).area
                                / inter_poly.area
                                > overlap_threshold
                            ):
                                overlaps = overlaps + 1
                                submergers.append(inter_poly)
                                iterated_cells.add(inter_uid)
                    # Catch block: empty list -> some cells are touching, but not overlapping strongly enough
                    if len(submergers) == 0:
                        merged_idx.append(query_uid)
                    else:  # Merging strategy: take the biggest cell, other merging strategies needs to get implemented
                        selected_poly_index = np.argmax(np.array([p.area for p in submergers]))
                        selected_poly = submergers[selected_poly_index]
                        selected_uid = poly_uid_map[selected_poly]
                        merged_idx.append(selected_uid)
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
        merged_cells = cleaned_edge_cells.loc[
            cleaned_edge_cells.index.isin(merged_idx)
        ].sort_index()

        ## Alternatively (5/10) can reset index each time (if want fresh 0-based index using reset_index(drop=True)
        ## But this might not work with the logic of using cleaned_edge_cells!
        # merged_cells = cleaned_edge_cells.loc[
        #     cleaned_edge_cells.index.isin(merged_idx)
        # ].sort_index().reset_index(drop=True)

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
    data_processed = _remove_overlap(data, args.overlap_threshold)
    output_path = f"{datadir}/processed_{geojson_name}.geojson"

    print('Converting to geojson...')

    # Save as geojson
    # Convert each row into a GeoJSON Feature
    features = []
    for _, row in data_processed.iterrows():
        geometry = row["geometry"]
        props = row["properties"]
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": props
        }
        features.append(feature)

    print('Saving as flat list of geojson features...')

    with open(output_path, "w") as f:
        json.dump(features, f)
    
    print('Merge and save complete.')

    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="path to folder containing geojson file")
    parser.add_argument("geojson_name", help="geojson file name")
    parser.add_argument("overlap_threshold", type=float, default=0.01, help="area overlap percentage threshold to be removed")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

    