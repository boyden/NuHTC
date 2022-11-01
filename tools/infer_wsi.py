# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import asyncio
import glob, sys, os, time, cv2, json
import mmcv
import torch
import openslide
import colorsys
import random
import numpy as np
import pandas as pd
from pycocotools import coco
from argparse import ArgumentParser
from tqdm import tqdm
from itertools import starmap
proj_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, f'{proj_path}/thirdparty/mmdetection')
sys.path.insert(0, proj_path)
from torch.utils.data import DataLoader
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot

from nuhtc.apis.inference import init_detector, save_result
from nuhtc.utils import patch_config
from wsi_core.WholeSlideImage import WholeSlideImage, Dataset_All_Bags, Whole_Slide_Bag_FP
from wsi_core.wsi_utils import StitchCoords, save_hdf5, save_pkl
from wsi_core.batch_process_utils import initialize_df

def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def mask_map(mask, coord, ori_shape):
    coord_x, coord_y = coord
    mask_shape = mask.shape
    wsi_mask = np.zeros(ori_shape, dtype=np.uint8)
    wsi_mask[coord_y:coord_y+mask_shape[0], coord_x:coord_x+mask_shape[1]] = mask
    wsi_mask = coco.maskUtils.encode(np.asfortranarray(wsi_mask))
    return wsi_mask

def mask2inst(inst_map):
    inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    inst_contour = np.concatenate([inst_contour, inst_contour[[0]]], axis=0)
    return inst_contour

def contour_map(contour, coord):
    map_contour = contour + coord
    return map_contour

def stitching(file_path, wsi_object, downscale = 64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)


    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size = 256, step_size = 256,
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
                  vis_params = {'vis_level': -1, 'line_thickness': 500},
                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level = 0,
                  use_default_params = False,
                  seg = False, save_mask = True,
                  stitch= False,
                  patch = False, auto_skip=True, process_list = None):



    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
        'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
        'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}


            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times

# python ./tools/infer_wsi.py demo/wsi models/htc_lite_PanNuke_infer.py models/pannuke.pth --patch --seg --stitch --patch_size 256 --step_size 224 --save_dir demo/wsi_res --det
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("source", help="path to folder containing raw wsi image files")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.35, help="score threshold"
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
        help="specify the file to skip.",
    )
    parser.add_argument('--step_size', type = int, default=256,
                        help='step_size')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')
    parser.add_argument('--patch', default=False, action='store_true')
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--stitch', default=False, action='store_true')
    parser.add_argument('--no_auto_skip', default=True, action='store_false')
    parser.add_argument('--save_dir', type = str,
                        help='directory to save processed data')
    parser.add_argument('--preset', default=None, type=str,
                        help='predefined profile of default segmentation and filter parameters (.csv)')
    parser.add_argument('--patch_level', type=int, default=0,
                        help='downsample level at which to patch')
    parser.add_argument('--process_list',  type = str, default=None,
                        help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--slide_ext',  type = str, default='.svs',
                        help='ext name of wsi')
    parser.add_argument('--mode',  type = str, default='qupath',
                        help='mode of save format')
    parser.add_argument('--det', default=False, action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir' : mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.CLASSES = ('T', 'I', 'C', 'D', 'E')
    model.inst_rng_colors = [[255, 0, 0],[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    model.inst_fillColor = ["rgba(255, 0, 0, 0)","rgba(0, 255, 0, 0)", "rgba(0, 0, 255, 0)", "rgba(255, 255, 0, 0)", "rgba(255, 0, 255, 0)"]
    model.inst_lineColor = ["rgb(255, 0, 0)","rgb(0, 255, 0)", "rgb(0, 0, 255)", "rgb(255, 255, 0)", "rgb(255, 0, 255)"]
    model.inst_rand_colors = random_colors(100)
    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                            patch_size=args.patch_size, step_size=args.step_size,
                                            seg=args.seg, use_default_params=False, save_mask=True,
                                            stitch=args.stitch,
                                            patch_level=args.patch_level, patch=args.patch,
                                            process_list=process_list, auto_skip=args.no_auto_skip)

    csv_path = f'{args.save_dir}/process_list_autogen.csv'
    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.save_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.source, slide_id+args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in args.output:
            print('skipped {}'.format(slide_id))
            continue

        wsi = openslide.open_slide(slide_file_path)
        wsi_shape = wsi.level_dimensions[0]
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=False, transform=False,
            custom_downsample=1, target_patch_size=-1)
        coords = []
        # os.makedirs(f'{args.save_dir}/{slide_id}/img', exist_ok=True)
        os.makedirs(f'{args.save_dir}/{slide_id}/infer', exist_ok=True)
        infer_dataloader = DataLoader(dataset, batch_size=16, num_workers=8)
        geojson_li = []
        dsajson_li = []
        for idx, data in enumerate(tqdm(infer_dataloader)):
            img = data[0]
            img = [elem.numpy() for elem in img]
            coord = data[1].numpy()
            # img.save(f'{args.save_dir}/{slide_id}/img/img_{coord[0]}_{coord[1]}.png')
            coords.append(coord)
            # img = glob.glob(f'{args.save_dir}/{slide_id}/img/img*.png')
            # img.sort()
            img_shape = img[0].shape
            result = inference_detector(model, img)
            seg_mask = [np.array(mmcv.concat_list(tmp_mask[1]), dtype=np.uint8)
                        for tmp_mask in result]

            bbox_results = [np.concatenate(tmp_res[0])[:, :4] for tmp_res in result]
            fg_scores = [np.concatenate(tmp_res[0])[:, 4] for tmp_res in result]
            labels = [np.concatenate([np.full(bbox.shape[0], i, dtype=np.int32)
                                      for i, bbox in enumerate(tmp_res[0])])
                      for tmp_res in result]
            for mask_id in range(len(seg_mask)):
                if len(seg_mask[mask_id]) == 0:
                    continue
                if args.det:
                    save_result(
                        model,
                        img[mask_id],
                        result[mask_id],
                        score_thr=args.score_thr,
                        out_file=f'{args.save_dir}/{slide_id}/infer/img_{coord[mask_id][0]}_{coord[mask_id][1]}.jpg',
                        font_size=6, thickness=1
                    )
                seg_area = (seg_mask[mask_id] == 1).sum(axis=(1, 2))
                select_id = (bbox_results[mask_id][:, 0] >= 1) & (bbox_results[mask_id][:, 1] >=1) & (bbox_results[mask_id][:, 2] <= img_shape[1]-1) & (bbox_results[mask_id][:, 3] <= img_shape[0]-1)
                select_id = select_id & (seg_area>10)
                bbox_results[mask_id] = bbox_results[mask_id][select_id]
                seg_mask[mask_id] = seg_mask[mask_id][select_id]
                labels[mask_id] = labels[mask_id][select_id]
                fg_scores[mask_id] = fg_scores[mask_id][select_id]
                if len(seg_mask[mask_id]) == 0:
                    continue
                tmp_mask = [m for m in seg_mask[mask_id]]
                seg_contour = list(map(mask2inst, tmp_mask))
                select_id = np.array([True if len(con) >=3 else False for con in seg_contour])
                seg_contour = [seg_contour[i].reshape(1, -1, 2) + coord[mask_id]
                               for i in range(len(select_id))
                               if select_id[i]]
                seg_mask[mask_id] = seg_contour
                if len(seg_mask[mask_id]) == 0:
                    continue
                bbox_results[mask_id] = bbox_results[mask_id][select_id] + np.tile(coord[mask_id], 2)
                labels[mask_id] = labels[mask_id][select_id]
                fg_scores[mask_id] = fg_scores[mask_id][select_id]
                if args.mode == 'qupath' or args.mode == 'all':
                    geojson = [{
                        "type": "Feature",
                        "geometry": {
                          "type": "Polygon",
                          "coordinates": seg_mask[mask_id][i].tolist()
                        },
                        "properties": {
                          "object_type": "annotation",
                          "color": model.inst_rng_colors[labels[mask_id][i]],
                          "label": int(labels[mask_id][i]),
                          "score": float(fg_scores[mask_id][i]),
                          "classification": {
                            "name": model.CLASSES[labels[mask_id][i]],
                          },
                          "isLocked": False
                        }
                    } for i in range(len(seg_mask[mask_id]))]
                    geojson_li += geojson
                if args.mode == 'dsa' or args.mode == 'all':
                    dsajson = [{
                        "fillColor": model.inst_fillColor[labels[mask_id][i]],
                        "lineColor": model.inst_lineColor[labels[mask_id][i]],
                        "lineWidth": 2,
                        "type": "polyline",
                        "closed": True,
                        "points": np.pad(seg_mask[mask_id][i][0], ((0, 0), (0, 1),),  'constant', constant_values=0).tolist(),
                        "label": {
                          "value": model.CLASSES[labels[mask_id][i]]
                        }
                    } for i in range(len(seg_mask[mask_id]))]
                    dsajson_li += dsajson
            if idx % 1000 == 0 or idx == len(infer_dataloader) - 1:
                if args.mode == 'qupath' or args.mode == 'all':
                    with open(f'{args.save_dir}/{slide_id}/{slide_id}.geojson', 'w') as f:
                        json.dump(geojson_li, f)
                if args.mode == 'dsa' or args.mode == 'all':
                    with open(f'{args.save_dir}/{slide_id}/{slide_id}_dsa.json', 'w') as f:
                        dsajson_file = {
                            'description': 'Seg with NuHTC automatically',
                            'elements': dsajson_li,
                            'name': 'NuHTC'
                                        }
                        json.dump(dsajson_file, f)

if __name__ == '__main__':
    main()
    # convert qupath geojson to dsa_json
    # dsa qupath-polygon WSI/infer/FUSCCTNBC466/FUSCCTNBC466.geojson --output_dir . --annotation_name quppath --image_filename FUSCCTNBC466.svs --classes_to_include T,I,C,D,E --line_colors '{"T": "rgb(255, 0, 0)", "I": "rgb(0, 255, 0)", "C": "rgb(0, 0, 255)", "D": "rgb(255, 255, 0)", "E": "rgb(255, 0, 255)"}' --fill_colors '{"T": "rgba(255, 0, 0, 0)", "I": "rgba(0, 255, 0, 0)", "C": "rgba(0, 0, 255, 0)", "D": "rgba(255, 255, 0, 0)", "E": "rgba(255, 0, 255, 0)"}'
    # upload annt
    # dsa_upload http://172.18.20.8:8080/api/v1 --collection_name FUDAN --image_filename FUSCCTNBC466.ndpi --annotation_filepath ./quppath_FUSCCTNBC466_partial.json