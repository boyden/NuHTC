"""convert_format.py.

Used to convert output to a format that can be used for visualisaton with QuPath.
Note, this is only used for tile segmentation results; not WSI.

"""

import os
import re
import glob
import json
import pathlib
import numpy as np
import argparse
import scipy
import scipy.io as sio
import shutil

from pycocotools import mask as maskUtils
####
def to_qupath(file_path, nuc_pos_list, nuc_type_list, type_info_dict):
    """
    For QuPath v0.2.3
    """

    def rgb2int(rgb):
        r, g, b = rgb
        return (r << 16) + (g << 8) + b

    nuc_pos_list = np.array(nuc_pos_list)
    nuc_type_list = np.array(nuc_type_list)
    assert nuc_pos_list.shape[0] == nuc_type_list.shape[0]
    with open(file_path, "w") as fptr:
        fptr.write("x\ty\tclass\tname\tcolor\n")

        nr_nuc = nuc_pos_list.shape[0]
        for idx in range(nr_nuc):
            nuc_type = nuc_type_list[idx]
            nuc_pos = nuc_pos_list[idx]
            type_name = type_info_dict[nuc_type][0]
            type_color = type_info_dict[nuc_type][1]
            type_color = rgb2int(type_color)  # color in qupath format
            fptr.write(
                "{x}\t{y}\t{type_class}\t{type_name}\t{type_color}\n".format(
                    x=nuc_pos[0],
                    y=nuc_pos[1],
                    type_class="",
                    type_name=type_name,
                    type_color=type_color,
                )
            )
    return

def conic2pannuke(file_path):
    data = np.load(file_path)
    class_num = data[:, :, :, 1].max()
    data_out = []
    for idx in range(data.shape[0]):
        img_data = data[idx]
        img_target = np.zeros((img_data.shape[0], img_data.shape[1], class_num+1))
        img_inst = img_data[:, :, 0]
        img_type = img_data[:, :, 1]
        for i in range(class_num):
            img_target[:, :, i][img_type==i+1] = img_inst[img_type==i+1]
        img_target[:, :, -1][img_inst==0] = 1
        data_out.append(img_target)
    return data_out

def conic2consep(file_path):
    data_name = os.path.basename(file_path)
    data_name, ext = os.path.splitext(data_name)
    data_dir = os.path.dirname(file_path)
    data = np.load(file_path)
    data_out = []
    os.makedirs(f'{data_dir}/mat/{data_name}', exist_ok=True)
    for idx in range(data.shape[0]):
        img_data = data[idx]
        img_inst = img_data[:, :, 0]
        # img_type = img_data[:, :, 1]
        # inst_type = []
        # inst_uid = np.delete(np.unique(img_inst), 0)
        # img_bboxes = np.zeros((len(inst_uid), 4))
        # img_centroids = np.zeros((len(inst_uid), 2))
        #
        # for i, uid in enumerate(inst_uid):
        #     nuclei_inst = np.zeros(img_inst.shape, dtype=np.uint8)
        #     nuclei_inst[img_inst==uid] == 1
        #     inst_type.append(scipy.stats.mode(img_type[img_inst==uid])[0][0])
        #     img_bboxes[i] = maskUtils.toBbox(maskUtils.encode(np.asfortranarray(nuclei_inst)))
        #
        # img_centroids[:, 0] = img_bboxes[:, 0] + img_bboxes[:, 2]/2
        # img_centroids[:, 1] = img_bboxes[:, 1] + img_bboxes[:, 3]/2
        # img_mat = {
        #     'inst_map': img_inst,
        #     'inst_type': np.reshape(inst_type, (len(inst_type), 1)),
        #     'inst_centroid': img_centroids,
        #     'inst_uid': np.array(range(1, img_bboxes.shape[0]+1))
        # }
        img_mat = {
            'inst_map': img_inst,
            'inst_uid': np.array(range(1, int(img_inst.max())))
        }

        sio.savemat(f'{data_dir}/mat/{data_name}/{data_name}_{idx+1}.mat', img_mat)
    return data_out

def pannuke2conic(file_path):
    data = np.load(file_path)
    class_num = data.shape[-1] - 1
    data_out = []
    for idx in range(data.shape[0]):
        uid = 1
        tmp_data = np.zeros((data.shape[1], data.shape[2], 2))
        img_data = data[idx]
        for i in range(class_num):
            inst_uid_arr = np.delete(np.unique(img_data[:, :, i]), 0)
            for inst_uid in inst_uid_arr:
                tmp_data[:, :, 0][img_data[:, :, i]==inst_uid] = uid
                tmp_data[:, :, 1][img_data[:, :, i]==inst_uid] = i+1
                uid += 1
        data_out.append(tmp_data)
    return  data_out

def pannuke2consep(file_path):
    data_name = os.path.basename(file_path)
    data_name, ext = os.path.splitext(data_name)
    data_dir = os.path.dirname(file_path)
    data = np.load(file_path)
    class_num = data.shape[-1] - 1
    data_out = []
    os.makedirs(f'{data_dir}/mat/{data_name}', exist_ok=True)
    for idx in range(data.shape[0]):
        img_data = data[idx]
        uid = 1
        img_inst = np.zeros((img_data.shape[0], img_data.shape[1]))
        for cls_id in range(class_num):
            inst_uid_arr = np.delete(np.unique(img_data[:, :, cls_id]), 0)
            for inst_uid in inst_uid_arr:
                img_inst[img_data[:, :, cls_id]==inst_uid] = uid
                uid += 1
        img_mat = {
            'inst_map': img_inst,
            'inst_uid': np.array(range(1, int(img_inst.max())))
        }
        sio.savemat(f'{data_dir}/mat/{data_name}/{data_name}_{idx+1}.mat', img_mat)
    return data_out

def consep2conic(file_path):
    img_mat_li = glob.glob(f'{file_path}/*mat')
    img_mat_li.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    data_out = []
    # label_dict = {
    #     1: 1,
    #     2: 2,
    #     3: 3,
    #     4: 3,
    #     5: 4,
    #     6: 4,
    #     7: 4
    # }
    for img_mat_path in img_mat_li:
        img_mat = sio.loadmat(img_mat_path)
        inst_map = img_mat['inst_map']
        inst_type = img_mat['inst_type']
        inst_uid = np.delete(np.unique(inst_map), 0)
        img_data = np.zeros((inst_map.shape[0], inst_map.shape[1], 2))
        img_data[:, :, 0] = inst_map
        for i, uid in enumerate(inst_uid):
            # img_data[:, :, 1][inst_map==uid] = label_dict[int(inst_type[i, 0])]
            img_data[:, :, 1][inst_map==uid] = int(inst_type[i, 0])
        data_out.append(img_data)
    return data_out

def consep2pannuke(file_path, class_num=5):
    img_mat_li = glob.glob(f'{file_path}/*mat')
    img_mat_li.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    inst_cls_num = class_num
    data_out = []
    # label_dict = {
    #     1: 1,
    #     2: 2,
    #     3: 3,
    #     4: 3,
    #     5: 4,
    #     6: 4,
    #     7: 4
    # }
    for img_mat_path in img_mat_li:
        img_mat = sio.loadmat(img_mat_path)
        inst_map = img_mat['inst_map']
        inst_type = img_mat['inst_type']
        inst_uid = np.delete(np.unique(inst_map), 0)
        img_data = np.zeros((inst_map.shape[0], inst_map.shape[1], inst_cls_num+1))
        for i, uid in enumerate(inst_uid):
            # cls_id = label_dict[int(inst_type[i, 0])]
            cls_id = int(inst_type[i, 0]-1)
            img_data[:, :, cls_id][inst_map==uid] = img_data[:, :, cls_id].max()+1
        img_data[:, :, -1] = 1- np.max(img_data[:, :, :-1], axis=-1)
        data_out.append(img_data)
    return data_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--format", type=str)
    parser.add_argument("--to", type=str)
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    assert args.format in ['consep', 'conic', 'pannuke']
    assert args.to in ['consep', 'conic', 'pannuke']
    data_name = os.path.basename(args.data_path)
    data_name, ext = os.path.splitext(data_name)
    data_dir = os.path.dirname(args.data_path)
    func_name = f'{args.format}2{args.to}'
    if func_name == 'conic2pannuke':
        data_out = conic2pannuke(args.data_path)
        if args.name:
            np.save(f'{data_dir}/{args.name}.npy', data_out)
        else:
            np.save(f'{data_dir}/{data_name}_{args.to}.npy', data_out)
    elif func_name == 'pannuke2conic':
        data_out = pannuke2conic(args.data_path)
        if args.name:
            np.save(f'{data_dir}/{args.name}.npy', data_out)
        else:
            np.save(f'{data_dir}/{data_name}_{args.to}.npy', data_out)
    elif func_name == 'consep2conic':
        data_out = consep2conic(args.data_path)
        if args.name:
            np.save(f'{data_dir}/{args.name}.npy', data_out)
        else:
            np.save(f'{data_dir}/{data_name}_{args.to}.npy', data_out)
    elif func_name == 'consep2pannuke':
        data_out = consep2pannuke(args.data_path, class_num=args.class_num)
        if args.name:
            np.save(f'{data_dir}/{args.name}.npy', data_out)
        else:
            np.save(f'{data_dir}/{data_name}_{args.to}.npy', data_out)
    elif func_name == 'conic2consep':
        conic2consep(args.data_path)
    elif func_name == 'pannuke2consep':
        pannuke2consep(args.data_path)
    else:
        raise NotImplementedError


