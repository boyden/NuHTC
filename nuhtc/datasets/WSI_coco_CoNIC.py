import os

import numpy as np
import scipy.io as sio
from PIL import Image
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO
from pycocotools import mask as maskUtils

from . import WSICocoDataset

@DATASETS.register_module()
class CoNICCocoDataset(WSICocoDataset):
    CLASSES = ('Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective tissue')
    PALETTE = [[238, 120, 56], [33, 226, 35], [251, 0, 26], [44, 191, 253], [26, 80, 198], [253, 255, 51]]
    ori_shape = (256, 256)
    img_size = (256, 256)
    img_stride = 93

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.cat2label = {cat_id: 0 for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_img_inst_map(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Labels'
        label_li = os.listdir(label_path)
        ori_img_dict = {}
        for label in label_li:
            name, ext = os.path.splitext(label)
            ori_img_dict[name] = []
            inst_map = sio.loadmat(f'{label_path}/{name}.mat')['inst_map']
            inst_map = inst_map.astype(np.int)
            inst_map = np.eye(inst_map.max() + 1, k=-1, dtype=np.uint8)[inst_map][:, :, :-1]
            for i in range(inst_map.shape[-1]):
                RLE_inst = maskUtils.encode(np.asfortranarray(inst_map[:, :, i]))
                # RLE_inst['counts'] = RLE_inst['counts'].decode('ascii')
                ori_img_dict[name].append(RLE_inst)
            del inst_map
        return ori_img_dict

    def get_img(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Images'
        label_li = os.listdir(label_path)
        ori_img_dict = {}
        for label in label_li:
            name, ext = os.path.splitext(label)
            inst_map = np.array(Image.open(f'{label_path}/{name}.png'))
            ori_img_dict[name] = inst_map
            del inst_map
        return ori_img_dict

    def get_img_name(self):
        base_path = '/'.join(self.img_prefix.split('/')[:-1])
        label_path = f'{base_path}/Images'
        label_li = os.listdir(label_path)
        name_li = []
        for label in label_li:
            name, ext = os.path.splitext(label)
            name_li.append(name)
        return name_li

    @property
    def debug(self):
        return True

    def mask2RLE(self, args):
        pred_mask, shift_x, shift_y = args
        tmp_pred_mask = np.zeros(self.ori_shape, dtype=np.uint8)
        tmp_pred_mask[shift_y:shift_y + self.img_size[1],
        shift_x:shift_x + self.img_size[0]] = pred_mask
        return maskUtils.encode(np.asfortranarray(tmp_pred_mask))
