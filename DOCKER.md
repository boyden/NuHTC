# Docker Usage Guide — NuHTC

This guide covers building the NuHTC Docker image and running training, inference, testing, WSI segmentation, and feature extraction inside a container.

---

## Prerequisites

- Docker >= 20.10 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- `docker compose` (v2 plugin) or `docker-compose` (v1)
- A local clone of this repository at the repo root
- Host NVIDIA driver >= 510 (CUDA 11.6 runtime)
- At least 30 GB free disk on the Docker data root

---

## File Overview

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the `nuhtc:latest` image |
| `.dockerignore` | Excludes unnecessary files from the build context |
| `docker-compose.yml` | Defines the `nuhtc` service with GPU access and volume mounts |

---

## Building the Image

Run this once from the repository root:

```shell
docker build -t nuhtc:latest .
```

The image sets `WORKDIR /workspace` and uses `/bin/bash` as its entrypoint. First build pulls ~6 GB of base image and runs heavy pip installs (`mmcv-full`, `histomicstk`); budget 20–30 minutes depending on network.

---

## Running Containers

### Quick one-off shell

```shell
docker run --rm -it --gpus all \
  --shm-size=16gb \
  -v ./datasets:/workspace/datasets \
  -v ./coco:/workspace/coco \
  -v ./models:/workspace/models \
  -v ./work_dirs:/workspace/work_dirs \
  -v ./demo:/workspace/demo \
  nuhtc:latest
```

### Via Docker Compose (recommended)

Start an interactive session:

```shell
docker compose run --rm nuhtc
```

Or run a single command directly:

```shell
docker compose run --rm nuhtc python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py --no-validate
```

---

## Volume Mounts

The compose service and the example `docker run` command above both mount:

| Host path | Container path | Contents |
|---|---|---|
| `./datasets` | `/workspace/datasets` | PanNuke raw numpy files |
| `./coco` | `/workspace/coco` | COCO-format annotation JSONs |
| `./models` | `/workspace/models` | Pre-trained checkpoints |
| `./work_dirs` | `/workspace/work_dirs` | Training outputs and logs |
| `./demo` | `/workspace/demo` | Demo images and WSI files |

Create the host directories before the first run:

```shell
mkdir -p datasets coco models work_dirs demo/imgs
```

> **Demo-data shadowing:** the compose mount `./demo:/workspace/demo` replaces the baked-in sample images from the repo. If you want to use the bundled samples, copy them out of the image first:
> ```shell
> docker create --name nuhtc-tmp nuhtc:latest && \
>   docker cp nuhtc-tmp:/workspace/demo/imgs ./demo/ && \
>   docker rm nuhtc-tmp
> ```
> Or drop the `./demo` volume from `docker-compose.yml` to use the baked-in samples directly.

---

## Environment Variables

The image sets these defaults; override them on the command line as needed:

| Variable | Default | Notes |
|---|---|---|
| `WANDB_MODE` | `offline` | Avoids wandb network errors during training |
| `MPLBACKEND` | `Agg` | Non-interactive matplotlib backend for headless runs |

Example override:

```shell
docker compose run --rm -e WANDB_MODE=online nuhtc python tools/train.py ...
```

---

## Training

Train NuHTC on fold 1 (update `fold = 1` in the config to change folds):

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    --no-validate"
```

> **Note:** If you see `UnicodeDecodeError` during wandb init, it is caused by certain NVIDIA driver versions (555.x–556.x). Downgrade to driver 552.44 or upgrade past 560.70. Setting `WANDB_MODE=offline` (the default) avoids this entirely.

---

## Testing

```shell
docker compose run --rm nuhtc bash -c "
CONFIG_NAME=htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
WEIGHT_BASE_PATH=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  \$WEIGHT_BASE_PATH/\$CONFIG_NAME \
  \$WEIGHT_BASE_PATH/latest.pth \
  --eval bbox --samples_per_gpu 16 \
  --eval-options save=True format=pannuke save_path=\$WEIGHT_BASE_PATH overlay=False
"
```

Calculate metrics after testing:

```shell
docker compose run --rm nuhtc \
  python tools/analysis_tools/pannuke/compute_stats.py \
    --true_path=datasets/PanNuke/masks/fold3/masks.npy \
    --type_path=datasets/PanNuke/masks/fold3/types.npy \
    --pred_path=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1/PanNukeCocoDataset/preds_pannuke.npy \
    --save_path=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1
```

---

## Inference

Download the checkpoint from the `models/` folder of the [project Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing) (file: `pannuke.pth`) into `./models/pannuke.pth` on the host. Then:

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/infer.py \
    demo/imgs \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth \
    --output demo/imgs_infer"
```

---

## WSI Segmentation

### 1. Segment

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py \
    demo/wsi \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth \
    --patch --seg --stitch \
    --patch_size 256 --step_size 192 --margin 1 --min_area 10 \
    --batch_size 32 --save_dir demo/wsi_infer --mode qupath"
```

For 512×512 patches (stronger overlap performance):

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py \
    demo/wsi \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth \
    --patch --seg --stitch \
    --patch_size 512 --step_size 448 --margin 1 --min_area 10 \
    --batch_size 32 --save_dir demo/wsi_infer --mode qupath"
```

### 2. Merge overlapping nuclei

```shell
docker compose run --rm nuhtc \
  python tools/nuclei_merge.py \
    --geojson demo/wsi_res/TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724.geojson \
    --overlap_threshold 0.05 --merge_strategy probability
```

---

## Feature Extraction

### Recommended: patch-centred extraction

```shell
docker compose run --rm nuhtc \
  python tools/wsi_feat_extract.py demo/wsi --segdir demo/wsi_res --mag 40
```

### Alternative: tile-based extraction

First run WSI inference with `--mode coco` or `--mode all`, then:

```shell
docker compose run --rm nuhtc \
  python tools/nuclei_feat_extract.py demo/wsi_res
```

Key options for `nuclei_feat_extract.py`:

| Flag | Default | Description |
|---|---|---|
| `--start` | `0` | Starting slide index |
| `--end` | `None` | Ending slide index (exclusive) |
| `--min_num` | `8` | Minimum nuclei per patch |
| `--patch_size` | `512` | Patch size in pixels |
| `--reverse` | `False` | Process slides in reverse order |

---

## Troubleshooting

### GPU not visible inside the container

Verify the NVIDIA Container Toolkit is installed and the Docker daemon is restarted:

```shell
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

If this fails, follow the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Shared memory errors (`RuntimeError: DataLoader worker ... died`)

The compose service sets `shm_size: 16gb`. If you run with plain `docker run`, pass `--shm-size=16gb`.

### `UnicodeDecodeError` from wandb during training

Set `WANDB_MODE=offline` (already the container default) or upgrade your NVIDIA driver past 560.70.

### `ModuleNotFoundError` for `histomicstk`

`histomicstk` must be installed at image build time. Verify it is present in the `Dockerfile`. If it was omitted, rebuild the image.

### Out-of-memory during training

Reduce `--samples_per_gpu` or use a smaller batch size. The default config targets a single GPU with at least 24 GB VRAM.

### Docker data root fills `/`

If you see `ENOSPC` or "no space left on device" during build, the Docker data root (`/var/lib/docker` by default) is out of space. Check with `docker system df` and prune unused cache:

```shell
docker builder prune -a -f
docker image prune -a -f   # removes all unused images
```

For a longer-term fix, move the data root to a larger partition by editing `/etc/docker/daemon.json` to set `"data-root": "/path/on/bigger/disk"` and restarting the daemon.
