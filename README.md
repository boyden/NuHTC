# NuHTC: A Hybrid Task Cascade for Nuclei Instance Segmentation and Classification

>  [Bao Li](https://boyden.github.io), Zhenyu Liu, Song Zhang, Xiangyu Liu, Jiangang Liu, Bensheng Qiu, Jie Tian.

![](./resources/nuhtc.jpg)
This repo is the official implementation of NuHTC.

## Overlaid Segmentation and Classification Prediction
The demo may take around 10s to load. 
![](./resources/instance_demo.gif)

## 👉 Setup Environment
Setup the python environment

```shell script
# Note, please follow the env.
conda create -n nuhtc -y python=3.6 
conda activate nuhtc
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
python -m pip install histomicstk==1.1.0 --find-links https://girder.github.io/large_image_wheels
pip install -r requirements.txt
```
## 👉 Train

```shell script
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
```
## 👉 Test

```shell script
# Segment image by image
CUDA_VISIBLE_DEVICES=0 python ./tools/infer.py demo/imgs models/htc_lite_PanNuke_infer.py models/pannuke.pth --out demo/imgs_infer
```

## 🚀 Segment the Whole Slide Image
Segment for the WSI with support output version: qupath, sql, dsa. Do not automatic support various magnification. (Default: 40x).
```shell script
CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py demo/wsi configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py models/pannuke.pth --out demo/wsi_res --patch --seg --stitch --space 256 --step_size 192 --target_spacing 0.25 --margin 2 --min_area 10 --save_dir demo/wsi_infer  --mode qupath --no_auto_skip
```

## 🗓️ Ongoing
- [ ] Support python 3.9 or higher
- [ ] Merge overlap nuclei when segment the WSI
