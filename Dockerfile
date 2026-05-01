# ComfyUI Refinement Worker for RunPod Serverless
# CUDA 12.4 (compatible with RunPod us-il-1 driver 550.127.05)
# Single venv approach — no comfy-cli, no dual venv conflict

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# ComfyUI startup wait defaults
#
# handler.py's check_server() falls back to a bounded retry loop when it
# can't determine whether ComfyUI is still alive from the PID file. With the
# upstream defaults (50ms interval × 500 attempts = 25s) flaky RunPod workers
# with slow cold starts routinely miss the window and fail with "ComfyUI
# server (127.0.0.1:8188) not reachable after multiple retries."
#
# Bumping the interval to 100ms and the retry cap to 2000 gives a ~200s
# fallback window, which covers every real-world cold start we've observed
# while still failing fast on a truly broken worker. Operators can override
# either value at the RunPod endpoint level without rebuilding this image.
# ---------------------------------------------------------------------------
ENV COMFY_API_AVAILABLE_INTERVAL_MS=100
ENV COMFY_API_AVAILABLE_MAX_RETRIES=2000

# Prevent apt post-install hooks from trying to start services during the
# Docker build. Without this, packages like dbus/packagekit/networkd-dispatcher
# hang on `Processing triggers for dbus` waiting on a system bus socket that
# doesn't exist in the build container. policy-rc.d returning exit 101 tells
# `invoke-rc.d` "deny start, but exit success" so apt continues.
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod +x /usr/sbin/policy-rc.d

# Add deadsnakes PPA for Python 3.12 (not available natively on Ubuntu 22.04).
# Two non-obvious choices baked in here:
#
# 1. We avoid `software-properties-common` (which provides add-apt-repository)
#    because it pulls in dbus/packagekit/networkd-dispatcher whose post-install
#    machinery breaks non-interactive Docker builds.
#
# 2. We use `[trusted=yes]` instead of fetching the GPG key, because RunPod's
#    build environment can't reach keyserver.ubuntu.com (connection times out
#    after 3.5 min). Transport is still HTTPS to Launchpad; we just skip the
#    cryptographic verification of the package signatures from the PPA. For
#    a private worker image this is an acceptable trade.
RUN echo "deb [trusted=yes] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
       > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Single virtual environment for everything
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI via git clone (not comfy-cli)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && pip install -r requirements.txt

# Install ComfyUI-Manager
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
    && cd ComfyUI-Manager && pip install -r requirements.txt || true

# Install Impact-Pack for FaceDetailer/DetailerForEach
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    && cd ComfyUI-Impact-Pack && pip install -r requirements.txt || true

# Install Impact-Subpack — provides UltralyticsDetectorProvider (moved out of main pack in V8+)
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
    && cd ComfyUI-Impact-Subpack && pip install -r requirements.txt || true

# Install WanVideoWrapper — Wan2.2 video generation nodes
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    && cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt || true

# Install VideoHelperSuite — video I/O utilities (combine frames, load video, etc.)
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt || true

# Ensure ultralytics is installed (required for UltralyticsDetectorProvider YOLO loading)
RUN pip install ultralytics

# Install handler dependencies
RUN pip install runpod requests websocket-client

# Download YOLO detection models (bbox + segmentation)
RUN mkdir -p /comfyui/models/ultralytics/bbox /comfyui/models/ultralytics/segm \
    && wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8n.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
    && wget -q -O /comfyui/models/ultralytics/bbox/hand_yolov8s.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"

# Copy nipple segmentation model (ADetailer Nipples v2.0 YOLO11s-seg, from CivitAI #490259)
COPY assets/nipples_v2_yolov11s-seg.pt /comfyui/models/ultralytics/segm/nipples_v2_yolov11s-seg.pt

# Download upscale models
RUN mkdir -p /comfyui/models/upscale_models \
    && wget -q -O /comfyui/models/upscale_models/4x-UltraSharp.pth \
       "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
    && wget -q -O /comfyui/models/upscale_models/4x_foolhardy_Remacri.pth \
       "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" \
    && wget -q -O /comfyui/models/upscale_models/RealESRGAN_x2plus.pth \
       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

# Download SAM model for precise segmentation masking in FaceDetailer
RUN mkdir -p /comfyui/models/sams \
    && wget -q -O /comfyui/models/sams/sam_vit_b_01ec64.pth \
       "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Download CodeFormer face restoration model (final face cleanup after upscaling)
RUN mkdir -p /comfyui/models/facerestore_models \
    && wget -q -O /comfyui/models/facerestore_models/codeformer-v0.1.0.pth \
       "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

# Add extra model paths for network volume
WORKDIR /comfyui
COPY src/extra_model_paths.yaml ./

# Add handler and startup scripts
WORKDIR /
COPY src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode || true

# Prevent pip from asking for confirmation
ENV PIP_NO_INPUT=1

WORKDIR /comfyui
CMD ["/start.sh"]
