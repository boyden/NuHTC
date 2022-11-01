"""run.

Usage:
  viz_utils.py --name=<n> --img_file=<path> --mask_file=<n> [--savedir=<path>] [--num_classes=<n>]
  viz_utils.py (-h | --help)
  viz_utils.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --name=<n>    Dataset name.
  --img_file=<n>    Image file.
  --mask_file=<n>    Mask file.
  --savedir=<path>    Path to save.
  --num_classes=<n>   The number of the classes. [default: 4].
"""

import os.path
import docopt
import cv2
import math
import random
import colorsys
import glob
import numpy as np
import scipy.io as sio
import itertools
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def colorize(ch, vmin, vmax):
    """Will clamp value value outside the provided range to vmax and vmin."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


####
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


####
def visualize_instances_map_pannuke(
    img_file, mask_file, save_path=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.

    Args:
        img_file: input file path
        mask_file: mask file path
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    # if not inst_rng_colors:
    #     inst_rng_colors = random_colors(len(inst_list))
    #     inst_rng_colors = np.array(inst_rng_colors) * 255
    #     inst_rng_colors = inst_rng_colors.astype(np.uint8)
    inst_rng_colors = [[255, 0, 0],[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    line_thickness=line_thickness
    print('Loading Data\n')
    img_data = np.load(img_file)
    mask_data = np.load(mask_file)
    print('Done\n')
    if save_path is None:
        save_path = os.path.dirname(img_file)
    os.makedirs(f'{save_path}/overlay', exist_ok=True)

    for i in tqdm(range(img_data.shape[0])):
        img_overlay = img_data[i]
        img_mask = mask_data[i]
        for idx in range(len(inst_rng_colors)):
            inst_colour = inst_rng_colors[idx]
            inst_cls_map = img_mask[:, :, idx]
            inst_id_li = np.unique(inst_cls_map)
            inst_id_li = np.delete(inst_id_li, 0)
            for inst_id in inst_id_li:
                inst_map = np.array(inst_cls_map==inst_id, dtype=np.uint8)
                inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(inst_contour[0]) == 0:
                    continue
                inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
                if inst_contour.shape[0] < 3:
                    continue
                if len(inst_contour.shape) != 2:
                    continue # ! check for trickery shape
                cv2.drawContours(img_overlay, [inst_contour], -1, inst_colour, line_thickness)

        Image.fromarray(np.array(img_overlay, dtype=np.uint8)).convert('RGB').save(f'{save_path}/overlay/{i}_overlay.png')
    return None

def visualize_instances_map_conic(
    img_file, mask_file, save_path=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.

    Args:
        img_file: input file path
        mask_file: mask file path
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    # if not inst_rng_colors:
    #     inst_rng_colors = random_colors(len(inst_list))
    #     inst_rng_colors = np.array(inst_rng_colors) * 255
    #     inst_rng_colors = inst_rng_colors.astype(np.uint8)
    inst_rng_colors = [[238, 120, 56], [33, 226, 35], [251, 0, 26], [44, 191, 253], [26, 80, 198], [253, 255, 51]]
    inst_class = ('Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective tissue')
    line_thickness=line_thickness


    print('Loading Data\n')
    img_data = np.load(img_file)
    mask_data = np.load(mask_file)
    print('Done\n')
    if save_path is None:
        save_path = os.path.dirname(img_file)
    os.makedirs(f'{save_path}/overlay', exist_ok=True)

    for i in tqdm(range(img_data.shape[0])):
        img_overlay = img_data[i]
        img_mask = mask_data[i]
        inst_data = img_mask[:, :, 0]
        cls_data = img_mask[:, :, 1]
        for idx in range(len(inst_rng_colors)):
            catg_id = idx+1
            annt_id_li = np.unique(inst_data[cls_data==catg_id])
            if len(annt_id_li) != 0:
                annt_id_li = np.delete(annt_id_li, 0)
                for inst_id in annt_id_li:
                    inst_map = np.array((inst_data==inst_id)&(cls_data==catg_id), dtype=np.uint8)
                    inst_colour = inst_rng_colors[idx]

                    inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(inst_contour[0]) == 0:
                        continue
                    inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
                    if inst_contour.shape[0] < 3:
                        continue
                    if len(inst_contour.shape) != 2:
                        continue # ! check for trickery shape
                    cv2.drawContours(img_overlay, [inst_contour], -1, inst_colour, line_thickness)

        Image.fromarray(np.array(img_overlay, dtype=np.uint8)).convert('RGB').save(f'{save_path}/overlay/{i}_overlay.png')
    return None

def visualize_instances_map_consep(
    img_file, mask_file, save_path=None, line_thickness=2, class_num=4
):
    """Overlays segmentation results on image as contours.

    Args:
        img_file: input file path
        mask_file: mask file path
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    # if not inst_rng_colors:
    #     inst_rng_colors = random_colors(len(inst_list))
    #     inst_rng_colors = np.array(inst_rng_colors) * 255
    #     inst_rng_colors = inst_rng_colors.astype(np.uint8)
    inst_rng_colors = [[255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255]]
    if int(class_num) != 4:
        inst_rng_colors = [[255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255]]

    print('Loading Data\n')
    print('Done\n')
    if save_path is None:
        save_path = os.path.dirname(img_file)
    os.makedirs(f'{save_path}/overlay', exist_ok=True)

    img_li = glob.glob(f'{img_file}/*png')

    for img in tqdm(img_li):
        imgname = os.path.basename(img).split('.')[0]
        img_overlay = np.array(Image.open(img))
        img_mask = sio.loadmat(f'{mask_file}/{imgname}.mat')
        img_inst_map = img_mask['inst_map']
        img_inst_type = img_mask['inst_type']
        for i in range(int(img_inst_map.max())):
            inst_type = int(img_inst_type[i][0])
            inst_map = np.array(img_inst_map==i+1, dtype=np.uint8)
            inst_colour = inst_rng_colors[inst_type-1]
            inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(inst_contour[0]) == 0:
                continue
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            cv2.drawContours(img_overlay, [inst_contour], -1, inst_colour, line_thickness)
        Image.fromarray(img_overlay).convert('RGB').save(f'{save_path}/overlay/{imgname}_overlay.png')

    return None

####
def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} , 
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        inst_colour = (int(inst_colour[0]), int(inst_colour[1]), int(inst_colour[2]))
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


####
def gen_figure(
    imgs_list,
    titles,
    fig_inch,
    shape=None,
    share_ax="all",
    show=False,
    colormap=plt.get_cmap("jet"),
):
    """Generate figure."""
    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig

def main(args):
    name = args['--name']
    img_file = args['--img_file']
    mask_file = args['--mask_file']
    savedir = args['--savedir']
    num_classes = args['--num_classes']
    # dirname = os.path.dirname(path)
    # os.makedirs(f'{dirname}/overlay', exist_ok=True)
    if name.lower() == 'pannuke':
        visualize_instances_map_pannuke(img_file, mask_file, save_path=savedir)
    elif name.lower() == 'conic':
        visualize_instances_map_conic(img_file, mask_file, save_path=savedir, line_thickness=1)
    elif name.lower() == 'consep':
        visualize_instances_map_consep(img_file, mask_file, save_path=savedir, class_num=num_classes)


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='Dataset Image Visualization v1.0')
    main(args)