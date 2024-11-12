# NuHTC: A Hybrid Task Cascade for Nuclei Instance Segmentation and Classification

>  [Bao Li](https://boyden.github.io), et al.

![](./resources/nuhtc.jpg)
This repo is the official implementation of NuHTC.

## Overlaid Segmentation and Classification Prediction
The demo may take around 10s to load. 
![](./resources/instance_demo.gif)

## 👉 Setup Environment
Setup the Python environment

```shell script
# Note, please follow the env.
conda create -n nuhtc -y python=3.6 
conda activate nuhtc
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
python -m pip install histomicstk==1.1.0 --find-links https://girder.github.io/large_image_wheels -i https://pypi.org/simple
```

## 👉 Preporcessing data
First please download and unzip the files from [PanNuke dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) where the folder structure should look this:

```
NuHTC
├── ...
├── datasets
│   ├── PanNuke
│   │   ├── images
│   │   │   ├── fold1
│   │   │   │   ├── images.npy
│   │   │   ├── fold2
│   │   │   │   ├── images.npy
│   │   │   ├── fold3
│   │   │   │   ├── images.npy
│   │   ├── masks
│   │   │   ├── fold1
│   │   │   │   ├── masks.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold2
│   │   │   │   ├── masks.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold3
│   │   │   │   ├── masks.npy
│   │   │   │   ├── types.npy
├── ...
```
For the coco format annotation, please download the `coco` folder json file from [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing)
```
NuHTC
├── ...
├── coco
│   ├── PanNuke
│   │   ├── PanNuke_annt_RLE_fold1.json
│   │   ├── PanNuke_annt_RLE_fold2.json
│   │   ├── PanNuke_annt_RLE_fold3.json
├── ...
```
Then generating `png` files for training and test.
```python
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

basedir = './datasets/PanNuke'
for fold in range(3):
    print(f'Preprocessing images: fold{fold+1}')
    imgdir = f'{basedir}/images/fold{fold+1}'
    img_data = np.load(f'{imgdir}/images.npy', mmap_mode='r')
    for i in tqdm(range(img_data.shape[0])):
        img = Image.fromarray(img_data[i].astype(np.uint8))
        os.makedirs(f'{basedir}/rgb', exist_ok=True)
        if not os.path.exists(f'{basedir}/rgb/fold{fold+1}_{i+1}.png'):
            img.convert('RGB').save(f'{basedir}/rgb/fold{fold+1}_{i+1}.png')

for fold in range(3):
    print(f'Preprocessing masks: fold{fold+1}')
    imgdir = f'{basedir}/masks/fold{fold+1}'
    img_data = np.load(f'{imgdir}/masks.npy', mmap_mode='r')
    for i in tqdm(range(img_data.shape[0])):
        img = 1 - img_data[i, :, :, 5]
        img = Image.fromarray(img.astype(np.uint8))
        os.makedirs(f'{basedir}/rgb_seg', exist_ok=True)
        if not os.path.exists(f'{basedir}/rgb_seg/fold{fold+1}_{i+1}.png'):
            img.save(f'{basedir}/rgb_seg/fold{fold+1}_{i+1}.png')
```

## 👉 Train
```shell script
# Please modify the `fold = 1` content to change the fold.
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py --no-validate
```

Note, recent update (~May 2024, driver version 555.85, 555.99, 556.12) of Nvidia driver may lead to `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 0: invalid start byte` in init wandb package. If your nvidia driver version is greater than `552.44`, please downgrade to the `Nvidia 552.44 studio driver` or update to the version greater than `560.70` for successfully training the models. For more details, please refer to [wandb issue](https://github.com/wandb/wandb/issues/7683).

## 👉 Test
``` shell script
CONFIG_NAME=htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
WEIGHT_BASE_PATH=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1

# predict nuclei from images
CUDA_VISIBLE_DEVICES=0 python tools/test.py $WEIGHT_BASE_PATH/$CONFIG_NAME $WEIGHT_BASE_PATH/latest.pth \
--eval bbox --samples_per_gpu 16 \
--eval-options save=True format=pannuke save_path=$WEIGHT_BASE_PATH overlay=False

# calculate the metric
python tools/analysis_tools/pannuke/compute_stats.py --true_path=datasets/PanNuke/masks/fold3/masks.npy --type_path=datasets/PanNuke/masks/fold3/types.npy \
--pred_path=$WEIGHT_BASE_PATH/PanNukeCocoDataset/preds_pannuke.npy --save_path=$WEIGHT_BASE_PATH
```

## 👉 Infer
Our trained checkpoint can be downloaded from the `models` folder in the [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing).
```shell script
# Segment image by image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py demo/imgs configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py models/pannuke.pth --out demo/imgs_infer
```

## 🚀 Segment the Whole Slide Image
Segment for the WSI with support output version: qupath, sql, dsa. Do not automatically support various magnifications. (Default: 40X).
```shell script
CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py demo/wsi configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py models/pannuke.pth \
--out demo/wsi_res --patch --seg --stitch --space 256 --step_size 192 --target_spacing 0.25 --margin 2 --min_area 10 \
--save_dir demo/wsi_infer --mode qupath --no_auto_skip
```

## 🗓️ Ongoing
- [ ] Support Python 3.9 or higher
- [ ] Merge overlap nuclei when segmenting the WSI
