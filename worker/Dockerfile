# syntax=docker/dockerfile:1.7
#
# Stage 1 fetches the Wan2.2 source. Stage 2 is a CUDA runtime image (no nvcc).
# flash-attn is installed from the official prebuilt wheel that matches the
# image's torch / CUDA / Python / C++ ABI, so nothing is compiled.
# Model weights are NOT baked in. worker.py downloads them into the
# `wan_models` volume on first start.

ARG TORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ---------------------------------------------------------------------------
FROM alpine:3.20 AS source
RUN apk add --no-cache git ca-certificates
# Pin WAN_REF to a commit SHA for reproducible builds.
ARG WAN_REPO=https://github.com/Wan-Video/Wan2.2.git
ARG WAN_REF=main
RUN git init /src \
 && git -C /src remote add origin "$WAN_REPO" \
 && git -C /src fetch --depth 1 origin "$WAN_REF" \
 && git -C /src checkout --detach FETCH_HEAD \
 && git -C /src rev-parse HEAD > /src/.wan-commit \
 && rm -rf /src/.git

# ---------------------------------------------------------------------------
FROM ${TORCH_IMAGE} AS runtime

ARG FLASH_ATTN_VERSION=2.7.4.post1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WAN_REPO_DIR=/opt/Wan2.2

# libgl1 / libglib2.0-0 are needed by opencv-python. ffmpeg is bundled by imageio-ffmpeg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsndfile1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=source /src/ /opt/Wan2.2/
COPY requirements.txt /tmp/worker-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -eux
cd /opt/Wan2.2

# Keep the CUDA build of torch that ships with the base image.
TORCH_VER=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
TV_VER=$(python -c 'import torchvision; print(torchvision.__version__.split("+")[0])')
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VER" "$TV_VER" > /tmp/constraints.txt

# Wan2.2 requirements without torch* (already present) and flash_attn (installed below as a wheel).
grep -viE '^(torch|torchvision|torchaudio|flash[-_]attn)([<>=~!; ]|$)' requirements.txt > /tmp/wan-requirements.txt
pip install -c /tmp/constraints.txt -r /tmp/wan-requirements.txt -r /tmp/worker-requirements.txt

# Choose the flash-attn wheel that matches this torch/Python/ABI.
PY_TAG=cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
TORCH_MM=$(echo "$TORCH_VER" | cut -d. -f1,2)
ABI=$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_MM}cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL}"

# Fail the build now rather than on the first job.
python -c "import torch, flash_attn, wan; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| wan OK')"
command -v hf
EOF

WORKDIR /app
COPY worker.py /app/worker.py

CMD ["python", "/app/worker.py"]
