# Docker Usage Guide — NuHTC

This guide covers pulling or building the NuHTC Docker image and running training, inference, testing, WSI segmentation, and feature extraction inside a container.

---

## Quick start

Pull the prebuilt image from GHCR. 

### Pull
```shell
docker pull ghcr.io/boyden/nuhtc:latest
```

### Run a demo
Copy the commands below into a terminal and run them to try inference. The image already includes `pannuke.pth` under `/workspace/models/` and sample images under `demo/imgs`.

```shell
mkdir -p demo/imgs_infer

docker run --gpus all --rm --shm-size=16g \
  -v "$PWD/demo/imgs_infer:/workspace/demo/imgs_infer" \
  ghcr.io/boyden/nuhtc:latest -c "python tools/infer.py /workspace/demo/imgs \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth --output /workspace/demo/imgs_infer"
```

When the run finishes, open `demo/imgs_infer` in your file browser to view the results.

---

## Prerequisites

- Docker ≥ 20.10 and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Repo clone at project root; host NVIDIA driver ≥ 510
- ~30 GB free on the Docker data root for build/pull
- (Optional) `docker compose` (v2) or `docker-compose` (v1)

---

## Files

| File | Role |
|---|---|
| `Dockerfile` | Image build |
| `.dockerignore` | Shrinks build context (large dirs like `datasets/`, `work_dirs/`, `demo/wsi/` are excluded) |
| `docker-compose.yml` | Optional; only if you clone the repo and prefer Compose over typing `docker run` |

---

## Build (optional)

Only if you want to build on your machine from this `Dockerfile` (e.g. local changes or baking weights):

```shell
docker build -t nuhtc:latest .
```

If `./models/pannuke.pth` exists in the build context it is copied into the image; otherwise mount weights at run time. First build is slow (~20–30 min) due to base image and pip.

---

## Training

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/coco:/workspace/coco \
  -v /path/to/work_dirs:/workspace/work_dirs \
  ghcr.io/boyden/nuhtc:latest -c "CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py --no-validate"
```

Some NVIDIA drivers (555.x–556.x) can trigger wandb `UnicodeDecodeError`; `WANDB_MODE=offline` avoids it, or upgrade the driver past 560.70.

---

## Testing

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/coco:/workspace/coco \
  -v /path/to/work_dirs:/workspace/work_dirs \
  ghcr.io/boyden/nuhtc:latest -c "
CONFIG_NAME=htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py
WEIGHT_BASE_PATH=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  \$WEIGHT_BASE_PATH/\$CONFIG_NAME \$WEIGHT_BASE_PATH/latest.pth \
  --eval bbox --samples_per_gpu 16 \
  --eval-options save=True format=pannuke save_path=\$WEIGHT_BASE_PATH overlay=False
"
```

Metrics example:

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/work_dirs:/workspace/work_dirs \
  ghcr.io/boyden/nuhtc:latest -c "python tools/analysis_tools/pannuke/compute_stats.py \
    --true_path=datasets/PanNuke/masks/fold3/masks.npy \
    --type_path=datasets/PanNuke/masks/fold3/types.npy \
    --pred_path=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1/PanNukeCocoDataset/preds_pannuke.npy \
    --save_path=work_dirs/htc_lite_swin_pytorch_seasaw_FPN_AttenROI_thres_96_base_aug_cas_PanNuke_full_epoch_200_fold1"
```


---

## Inference

This runs the model on every image in a folder you choose and writes overlay images to an output folder.

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/your/pngs:/data/imgs \
  -v /path/to/output:/data/imgs_infer \
  ghcr.io/boyden/nuhtc:latest -c "python tools/infer.py /data/imgs \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py \
    models/pannuke.pth --output /data/imgs_infer"
```

Other checkpoints: mount weights and point `tools/infer.py` at the matching config (see [Google Drive](https://drive.google.com/drive/folders/1MezZrVwx7S6MNYkpMO5ja2D6KcZkRvYo?usp=sharing)).

---

## WSI

Whole slide images (WSIs) in one input directory (e.g. `.svs`) are handled with the following pipeline.

- **Segmentation**: patch-based nucleus instance prediction and structured annotations for downstream visualization and analysis, including QuPath-compatible exports.
- **Merge**: Merge **deduplicates** nuclei in the GeoJSON so each nucleus appears **once**. **the same nucleus** can be predicted **twice near tile edges**.
- **Feature extraction** (optional): nucleus-centred ROIs on the full slide, or tile-based features on saved inference patches.

#### Segment

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/your/slides:/data/wsi \
  -v /path/to/wsi_output:/data/wsi_infer \
  ghcr.io/boyden/nuhtc:latest -c "python tools/infer_wsi.py /data/wsi \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py models/pannuke.pth \
    --patch --seg --stitch --patch_size 256 --step_size 192 --margin 1 --min_area 10 \
    --batch_size 32 --save_dir /data/wsi_infer --mode qupath"
```

#### Merge overlaps

```shell
docker run --gpus all --rm \
  -v /path/to/wsi_output:/data/wsi_infer \
  ghcr.io/boyden/nuhtc:latest -c "python tools/nuclei_merge.py \
    --geojson /data/wsi_infer/nuclei/<SLIDE_ID>/<SLIDE_ID>.geojson \
    --overlap_threshold 0.05 --merge_strategy probability"
```

#### Nucleus-centred ROI features

Reads the full WSI, crops a compact image around each nucleus from its segmentation polygon, and writes one feature vector per nucleus.

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/your/slides:/data/wsi \
  -v /path/to/wsi_output:/data/wsi_infer \
  ghcr.io/boyden/nuhtc:latest -c "python tools/wsi_feat_extract.py /data/wsi \
    --segdir /data/wsi_infer/nuclei --mag 40"
```

#### Tile-based nucleus features

Uses saved inference tiles: each fixed patch image holds many instances; features are computed for every nucleus in that tile from the patch RGB and instance masks. Run WSI inference with `--mode coco` or `--mode all` first so `coco_nuclei.json` and `imgs/` are present under the output tree.

```shell
docker run --gpus all --rm --shm-size=32g \
  -v /path/to/wsi_output:/data/wsi_infer \
  ghcr.io/boyden/nuhtc:latest -c "python tools/nuclei_feat_extract.py /data/wsi_infer"
```

`nuclei_feat_extract.py` flags: `--start`, `--end`, `--min_num`, `--patch_size`, `--reverse` (defaults in `--help`).

## Running containers

### Docker Compose (Optional)

If you built locally ([Build optional](#build-optional)), use `nuhtc:latest` wherever the examples use `ghcr.io/boyden/nuhtc:latest`.

Training, testing, and custom WSIs live on the host: pass `-v` (see table). `docker-compose.yml` does not mount them by default—add volumes in the file or on the command line.

Example training with host dirs:

```shell
docker compose run --rm \
  -v ./datasets:/workspace/datasets \
  -v ./coco:/workspace/coco \
  -v ./work_dirs:/workspace/work_dirs \
  nuhtc bash -c "python tools/train.py \
    configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py --no-validate"
```

### Typical mounts

| Host | Container | Use |
|---|---|---|
| `./datasets` | `/workspace/datasets` | PanNuke numpy |
| `./coco` | `/workspace/coco` | COCO JSON |
| `./work_dirs` | `/workspace/work_dirs` | Checkpoints / logs |
| `./models` | `/workspace/models` | Optional: override baked `pannuke.pth` |
| `./demo` | `/workspace/demo` | Optional: your images / WSI / outputs |

---

## Troubleshooting

- GPU: `docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi` — if it fails, reinstall the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- DataLoader / SHM: raise `--shm-size` on `docker run` (this guide uses `32g`; `16g` is often enough if the host is tight on RAM).
- `histomicstk` missing: must be present at image build; rebuild from this `Dockerfile`.
- Training OOM: lower `--samples_per_gpu` or batch size in config.
- Disk full on build: `docker system df`; `docker builder prune -a -f` (and image prune if appropriate).
