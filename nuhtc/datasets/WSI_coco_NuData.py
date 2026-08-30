from mmdet.datasets import DATASETS
from . import WSICocoDataset


@DATASETS.register_module()
class NuDataCocoDataset(WSICocoDataset):
    # single class; the name must match `categories` in the merged NuData json
    CLASSES = ('Nucleus',)
    PALETTE = [[255, 0, 0]]
