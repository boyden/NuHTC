FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/thirdparty/mmdetection:/workspace
ENV WANDB_MODE=offline
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide0 \
    libopenjp2-7 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgeos-dev \
    libjpeg-dev \
    libtiff5-dev \
    libhdf5-dev \
    pkg-config \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# Filter requirements.txt to drop packages that are listed but never imported by the
# NuHTC runtime (verified via grep). Keeping them causes resolver failures against
# the pinned PyTorch 1.13.1 / Python 3.10 environment:
#   mmtrack                 — heavy tracking toolbox, zero imports in this repo
#   ts, model_archiver      — TorchServe extras, zero imports
#   pytorch_sphinx_theme    — docs-only, zero imports
# We also upgrade scikit-image past 0.18 so it is compatible with numpy 1.26.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.26.4 "h5py==3.11.0" \
    && grep -Ev '^(h5py|mmtrack|ts|model_archiver|pytorch_sphinx_theme)([<>=!].*)?$' \
        /workspace/requirements.txt > /tmp/requirements-docker.txt \
    && sed -i 's/^scikit-image==0\.18\.3$/scikit-image>=0.22,<0.25/' \
        /tmp/requirements-docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && pip install --no-cache-dir mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html \
    # histomicstk 1.2.10 declares `large-image[sources]` as a transitive dep,
    # which on Linux pulls in openslide-bin + GDAL + large-image-source-gdal.
    # Those all require glibc / libgdal versions that the PyTorch 1.13.1
    # Ubuntu base image does not ship, and source builds are explicitly
    # rejected by their setup tooling. NuHTC code never imports `large_image`
    # — it uses histomicstk submodules (features, preprocessing,
    # annotations_and_masks) on numpy arrays, plus openslide via its own
    # tools/wsi_core/. So we install histomicstk with --no-deps and supply
    # only its pure-Python runtime deps manually.
    && pip install --no-cache-dir \
        'girder-client' 'nimfa' 'slicer-package-manager-client' \
        'girder-slicer-cli-web' 'ctk-cli' \
        'large-image' 'shapely' 'dask' 'distributed' 'pyvips' \
    && pip install --no-cache-dir --no-deps histomicstk==1.2.10

COPY . /workspace

RUN python -c "\
import sys; \
sys.path.insert(0, '/workspace/thirdparty/mmdetection'); \
sys.path.insert(0, '/workspace'); \
import mmdet; \
import nuhtc; \
print(f'mmdet={mmdet.__version__}')"

ENTRYPOINT ["/bin/bash"]
