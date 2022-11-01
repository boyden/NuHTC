from mmdet.datasets import DATASETS, ConcatDataset, ClassBalancedDataset, build_dataset
from collections import defaultdict
import numpy as np
import math

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]

@DATASETS.register_module()
class CASDataset(ClassBalancedDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, dataset, filter_empty_gt=False, repeat_factors=1):
        self.dataset = build_dataset(dataset)
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.dataset.CLASSES
        self.cat_ids = self.dataset.cat_ids
        self.repeat_factors = repeat_factors
        self.cls_indices = list(range(len(self.CLASSES)))
        self.category_indices = self._get_category_indices(self.dataset)
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag

    def __getitem__(self, idx):
        cls_idx = np.random.choice(self.cls_indices)
        ori_index = np.random.choice(self.category_indices[cls_idx])
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.dataset)*self.repeat_factors

    def _get_category_indices(self, dataset):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        category_indices = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = self.cat_ids
            for cat_id in cat_ids:
                label_id = self.dataset.cat2label[cat_id]
                if category_indices.get(label_id, None):
                    category_indices[label_id].append(idx)
                else:
                    category_indices[label_id] = [idx]
        return category_indices