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
First please download and unzip the files from [PanNuke dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke).

```
NuHTC
├── ...
├── datasets
│   ├── PanNuke
│   │   ├── images
│   │   │   ├── fold1
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold2
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold3
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   ├── masks
│   │   │   ├── fold1
│   │   │   │   ├── masks.npy
│   │   │   ├── fold2
│   │   │   │   ├── masks.npy
│   │   │   ├── fold3
│   │   │   │   ├── masks.npy
├── ...
```
For the coco format annotation, please download the `coco` folder json file from [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing).

```shell script
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
```
## 👉 Train

```shell script
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
```
## 👉 Test
Our trained checkpoint can be downloaded from the `models` folder in the [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing).
```shell script
# Segment image by image
CUDA_VISIBLE_DEVICES=0 python ./tools/infer.py demo/imgs models/htc_lite_PanNuke_infer.py models/pannuke.pth --out demo/imgs_infer
```
Note: Due to different implementation of calculating bPQ and mPQ, the implementation of PQ in our codes are just reference. For reproduced results, please refer to PanNuke implementation https://github.com/TissueImageAnalytics/PanNuke-metrics.

## 🚀 Segment the Whole Slide Image
Segment for the WSI with support output version: qupath, sql, dsa. Do not automatically support various magnifications. (Default: 40x).
```shell script
CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py demo/wsi configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py models/pannuke.pth --out demo/wsi_res --patch --seg --stitch --space 256 --step_size 192 --target_spacing 0.25 --margin 2 --min_area 10 --save_dir demo/wsi_infer  --mode qupath --no_auto_skip
```

## 🗓️ Ongoing
- [ ] Support Python 3.9 or higher
- [ ] Merge overlap nuclei when segmenting the WSI
