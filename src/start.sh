#!/usr/bin/env bash

# Cold-start timing markers — the handler reads /tmp/boot-timing and returns it
# in the job output so the download-vs-boot split is observable via the status
# API (RunPod exposes no serverless container-log API).
BOOT_TIMING=/tmp/boot-timing
bt() { echo "$1=$(date +%s.%N)" >> "$BOOT_TIMING"; }
: > "$BOOT_TIMING"; bt container_start

# Mirror ALL of start.sh's own output to /tmp/boot.log (shipped to R2 below)
# while keeping it on container stdout for the dashboard.
exec > >(tee -a /tmp/boot.log) 2>&1

# RunPod exposes NO serverless log API, so ship our own logs to R2 from the
# very first moment of boot: every 20s upload boot.log + boot-timing +
# comfyui.log (once it exists) to logs/serverless/<endpoint>/<pod-id>/.
# Runs whenever R2 creds are on the endpoint env; failures never break boot.
if [ -n "${R2_ENDPOINT:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
    (
        while true; do
            python -u - <<'PYEOF' >/tmp/log-ship.err 2>&1 || true
import os
import boto3
from botocore.client import Config as BotoConfig
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=BotoConfig(signature_version="s3v4"),
)
pod = os.environ.get("RUNPOD_POD_ID", "unknown-pod")
endpoint = os.environ.get("RUNPOD_ENDPOINT_ID", "unknown-endpoint")
prefix = f"logs/serverless/{endpoint}/{pod}"
for local, key in (
    ("/tmp/boot.log", "boot.log"),
    ("/tmp/boot-timing", "boot-timing.txt"),
    ("/tmp/comfyui.log", "comfyui.log"),
):
    if os.path.exists(local):
        client.upload_file(local, os.environ["R2_BUCKET"], f"{prefix}/{key}")
PYEOF
            sleep 20
        done
    ) &
    echo "worker-comfyui: R2 log shipper started (logs/serverless/${RUNPOD_ENDPOINT_ID:-unknown-endpoint}/${RUNPOD_POD_ID:-unknown-pod}/)"
fi

# Start SSH server if PUBLIC_KEY is set (enables remote access and dev-sync.sh)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if they don't exist (removed during image build for security)
    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "worker-comfyui: SSH server started" || echo "worker-comfyui: SSH server could not be started" >&2
fi

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available. PyTorch CUDA init failed:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: This usually means the GPU on this machine is not properly initialized."
    echo "worker-comfyui: Please contact RunPod support and report this machine."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# ---------------------------------------------------------------------------
# Ephemeral-pod weight bootstrap.
# If MANIFEST_URL is set, this container is being used as an ephemeral
# Wan-batch pod (no network volume mounted). Download the aria2 manifest
# from R2, install aria2, run the download, signal completion via a
# sentinel file. Then ComfyUI is started normally below with the freshly
# downloaded weights visible in /comfyui/models/.
#
# We do this in start.sh — not via SSH-exec from the orchestrator — because
# RunPod's SSH relay (ssh.runpod.io) is unreliable for multi-second commands
# and large outputs (apt-get update + install drops the session ~half the
# time). Doing it here keeps the orchestrator's SSH usage to tiny polls.
# ---------------------------------------------------------------------------
if [ -n "$MANIFEST_URL" ]; then
    echo "worker-comfyui: MANIFEST_URL set — bootstrapping ephemeral weights"
    mkdir -p /workspace /comfyui/models/diffusion_models /comfyui/models/text_encoders /comfyui/models/vae /comfyui/models/loras

    echo "worker-comfyui: Installing aria2 + curl…"
    # Grouped with braces so the redirect captures both commands' output and
    # `!` negates the combined exit status. Without the braces, bash parses
    # `! cmd1 && cmd2` as `(!cmd1) && cmd2` — when cmd1 succeeds, the `&&`
    # short-circuits and cmd2 (the actual install) never runs.
    # curl is required by the orchestrator to POST workflows to ComfyUI and
    # poll /history over the SSH relay (WanPodBatchOrchestrator.runBatch).
    if ! { apt-get update -qq && apt-get install -y aria2 curl; } > /tmp/apt-install.log 2>&1; then
        echo "worker-comfyui: aria2/curl install FAILED — see /tmp/apt-install.log"
        cat /tmp/apt-install.log
        exit 1
    fi
    if ! command -v aria2c > /dev/null; then
        echo "worker-comfyui: aria2c missing after install — aborting"
        cat /tmp/apt-install.log
        exit 1
    fi
    if ! command -v curl > /dev/null; then
        echo "worker-comfyui: curl missing after install — aborting"
        cat /tmp/apt-install.log
        exit 1
    fi

    echo "worker-comfyui: Downloading manifest from R2…"
    if ! wget --quiet -O /workspace/manifest.aria2 "$MANIFEST_URL"; then
        echo "worker-comfyui: manifest wget FAILED"
        exit 1
    fi

    echo "worker-comfyui: Running aria2c (5 files × 5 connections)…"
    # -d /  — aria2 treats `out=/abs/path` as relative to its working dir even
    # when out starts with `/`. Without -d, cwd at exec time (typically /) was
    # OK by luck, but inside the container start.sh runs from `/` so out=/comfyui/...
    # produced /comfyui/comfyui/... (double-prefix). Pin -d / so absolute paths
    # in the manifest land where they say they do.
    if ! aria2c -i /workspace/manifest.aria2 \
        -d / \
        -j 5 -x 5 -s 5 \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --console-log-level=warn \
        --summary-interval=15 > /tmp/aria2c.log 2>&1; then
        echo "worker-comfyui: aria2c download FAILED — see /tmp/aria2c.log"
        tail -50 /tmp/aria2c.log
        exit 1
    fi

    # Sentinel — the orchestrator polls for this file via a tiny SSH command.
    date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/download.done
    echo "worker-comfyui: Weight bootstrap complete"
fi

# ---------------------------------------------------------------------------
# LTX serverless lazy-weights bootstrap.
# The LTX serverless image ships WITHOUT baked weights (a baked 22B checkpoint
# makes the image ~62 GB — unbuildable/unpushable on our CI/registry/pods). When
# LTX_BOOTSTRAP is set, download this worker's variant weights into container
# disk now, then fall through to the NORMAL serverless handler below (unlike the
# MANIFEST_URL/ephemeral path, which sleeps forever). See download-ltx-weights.sh.
# ---------------------------------------------------------------------------
if [ -n "$LTX_BOOTSTRAP" ]; then
    echo "worker-comfyui: LTX_BOOTSTRAP set — downloading LTX weights (variant=${LTX_VARIANT:-unset})"
    bt download_start
    if ! /download-ltx-weights.sh; then
        echo "worker-comfyui: LTX weight download FAILED — aborting boot" >&2
        exit 1
    fi
    bt download_end
fi

# ---------------------------------------------------------------------------
# Krea 2 serverless lazy-weights bootstrap — same shape as LTX_BOOTSTRAP above.
# ---------------------------------------------------------------------------
if [ -n "${KREA_BOOTSTRAP:-}" ]; then
    echo "worker-comfyui: KREA_BOOTSTRAP set — downloading Krea 2 weights"
    bt download_start
    if ! /download-krea-weights.sh; then
        echo "worker-comfyui: Krea weight download FAILED — aborting boot" >&2
        exit 1
    fi
    bt download_end
fi

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

bt comfy_launch
echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

COMFY_PID_FILE="/tmp/comfyui.pid"
COMFY_PID_FILE_TMP="${COMFY_PID_FILE}.tmp"
rm -f "$COMFY_PID_FILE" "$COMFY_PID_FILE_TMP"

write_comfy_pid_file() {
    local pid="$1"
    echo "$pid" > "$COMFY_PID_FILE_TMP"
    mv "$COMFY_PID_FILE_TMP" "$COMFY_PID_FILE"
}

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout --extra-model-paths-config /comfyui/extra_model_paths.yaml &
    write_comfy_pid_file "$!"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    # Ephemeral mode: bind ComfyUI to 0.0.0.0 so the orchestrator can probe
    # /system_stats over the SSH relay (which exec-runs `curl localhost:8188/...`).
    if [ -n "$MANIFEST_URL" ]; then
        python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen 0.0.0.0 --port 8188 --verbose "${COMFY_LOG_LEVEL}" --log-stdout --extra-model-paths-config /comfyui/extra_model_paths.yaml &
        write_comfy_pid_file "$!"
        echo "worker-comfyui: Ephemeral mode — sleeping forever (no serverless handler)"
        # Don't launch handler.py — there's no serverless queue feeding this pod.
        # Sleep keeps the container alive so the orchestrator can dispatch via SSH.
        sleep infinity
    else
        # ComfyUI output goes to /tmp/comfyui.log (handler embeds the tail in
        # error payloads; the R2 shipper below uploads it) AND is tailed back
        # to container stdout for the dashboard.
        python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout --extra-model-paths-config /comfyui/extra_model_paths.yaml >> /tmp/comfyui.log 2>&1 &
        write_comfy_pid_file "$!"
        tail -n +1 -f /tmp/comfyui.log &

        echo "worker-comfyui: Starting RunPod Handler"
        python -u /handler.py
    fi
fi
