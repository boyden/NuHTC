from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
from . import WSICocoDataset

@DATASETS.register_module()
class PanNukeCocoDataset(WSICocoDataset):
    CLASSES = ('Neoplastic cells', 'Inflammatory', 'Connective/Soft tissue cells', 'Dead Cells', 'Epithelial')
    # PALETTE = [[251, 0 ,26],[34, 221, 76], [35, 92, 236], [225, 225, 0], [255, 158, 69]]
    PALETTE = [[255, 0, 0],[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
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