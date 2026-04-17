# Docker Usage Guide — NuHTC

This guide covers pulling / building the NuHTC Docker image and running training, inference, testing, WSI segmentation, and feature extraction inside a container.

---

## Quick start — pull the prebuilt image

The fastest path: pull the image we publish to GHCR. It ships with `pannuke.pth` already baked in at `/workspace/models/pannuke.pth`, so inference runs with no weight download and no Python setup.

```shell
docker pull ghcr.io/kaneyxx/nuhtc:latest
```

Everywhere below you can substitute `nuhtc:latest` (local build) with `ghcr.io/kaneyxx/nuhtc:latest` (prebuilt). Commands in this guide use the short `nuhtc:latest` tag for brevity; re-tag locally if you prefer:

```shell
docker tag ghcr.io/kaneyxx/nuhtc:latest nuhtc:latest
```

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

## Building the Image (optional — only if you prefer to build locally)

If you want to modify the image or build without the GHCR dependency, run from the repository root:

```shell
docker build -t nuhtc:latest .
```

If `./models/pannuke.pth` (or another checkpoint) exists in the build context, it is baked into the image at `/workspace/models/`; otherwise the image builds weight-less and you mount weights at run time.

The image sets `WORKDIR /workspace` and uses `/bin/bash` as its entrypoint. First build pulls ~6 GB of base image and runs heavy pip installs (`mmcv-full`, `histomicstk`); budget 20–30 minutes depending on network.

---

## Running Containers

### Quick one-off shell

The baked PanNuke weight is already at `/workspace/models/pannuke.pth`, so the `./models` mount below is only needed if you want to override with a different checkpoint or use CoNSeP / NuCLS / CoNIC weights. Same for `./demo` (baked sample images).

```shell
docker run --rm -it --gpus all \
  --shm-size=16gb \
  -v ./datasets:/workspace/datasets \
  -v ./coco:/workspace/coco \
  -v ./work_dirs:/workspace/work_dirs \
  # Optional overrides (remove the `#` to enable):
  # -v ./models:/workspace/models    # override baked pannuke.pth
  # -v ./demo:/workspace/demo        # override baked demo/imgs
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

| Host path | Container path | Contents | When to mount |
|---|---|---|---|
| `./datasets` | `/workspace/datasets` | PanNuke raw numpy files | Training / testing |
| `./coco` | `/workspace/coco` | COCO-format annotation JSONs | Training / testing |
| `./work_dirs` | `/workspace/work_dirs` | Training outputs and logs | Training |
| `./models` | `/workspace/models` | Pre-trained checkpoints | **Optional** — shadows the baked `pannuke.pth`; mount only to swap in a different weight |
| `./demo` | `/workspace/demo` | Demo images and WSI files | **Optional** — shadows the baked `demo/imgs` samples; mount only to use your own inputs |

Create the host directories before the first training / testing run:

```shell
mkdir -p datasets coco work_dirs
```

> **Override vs baked content.** The image bundles `pannuke.pth` at `/workspace/models/pannuke.pth` and a small set of demo PNGs at `/workspace/demo/imgs/`. Mounting `./models` or `./demo` REPLACES the entire directory view — you lose the baked content while the mount is active. That is usually what you want (you are bringing your own data), but if you want both the baked demo samples AND your own data, copy the baked samples out first:
> ```shell
> docker create --name nuhtc-tmp nuhtc:latest && \
>   docker cp nuhtc-tmp:/workspace/demo/imgs ./demo/ && \
>   docker rm nuhtc-tmp
> ```

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

The PanNuke checkpoint is already baked into the image at `/workspace/models/pannuke.pth`, so nothing to download or mount:

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/infer.py \
    demo/imgs \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth \
    --output demo/imgs_infer"
```

To use a different checkpoint (CoNSeP / NuCLS / CoNIC), download it from the [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing) and mount its host directory over `/workspace/models`:

```shell
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/your/weights:/workspace/models \
  nuhtc:latest -c "python tools/infer.py demo/imgs \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_CoNSeP_seasaw_CAS.py \
    models/consep.pth --output /tmp/out"
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

For 512×512 patches (stronger edge performance, but heavier on VRAM — **`--batch_size 32` OOMs on a 24 GB GPU, drop to `--batch_size 8` first**):

```shell
docker compose run --rm nuhtc \
  bash -c "CUDA_VISIBLE_DEVICES=0 python tools/infer_wsi.py \
    demo/wsi \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth \
    --patch --seg --stitch \
    --patch_size 512 --step_size 448 --margin 1 --min_area 10 \
    --batch_size 8 --save_dir demo/wsi_infer --mode qupath"
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
