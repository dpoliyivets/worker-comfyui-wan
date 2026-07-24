#!/usr/bin/env bash
# Krea 2 serverless lazy-weights bootstrap (mirrors download-ltx-weights.sh).
#
# The krea-render image ships WITHOUT baked weights (~10 GB image, builds and
# pushes anywhere — same setup as the LTX lazy image). Each worker downloads
# the Krea 2 set (~17.4 GB) into container disk on cold start:
#   diffusion_models/krea2_turbo_fp8_scaled.safetensors  (12.24 GiB)
#   text_encoders/qwen3vl_4b_fp8_scaled.safetensors      ( 4.88 GiB)
#   vae/qwen_image_vae.safetensors                       ( 0.24 GiB)
# All three come from the PUBLIC Comfy-Org/Krea-2 HF repack. HF has migrated
# these repos to Xet storage where aria2/wget byte-range requests 403
# ([[project_hf_xet_aria2_fails]]), so we use the Xet-native hf client with
# hf_transfer. LoRAs are NOT downloaded here — the handler pulls them from R2
# per job (see src/krea_workflow.py).
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS_DIR=/comfyui/models
REPO=Comfy-Org/Krea-2

download() {
    local rel_path="$1"
    local dest="$MODELS_DIR/$rel_path"
    if [ -s "$dest" ]; then
        echo "krea-weights: cached $rel_path"
        return 0
    fi
    echo "krea-weights: downloading $rel_path"
    hf download "$REPO" "$rel_path" --local-dir /tmp/hf-krea
    mkdir -p "$(dirname "$dest")"
    mv "/tmp/hf-krea/$rel_path" "$dest"
    rm -rf /tmp/hf-krea
}

start_ts=$(date +%s)
download diffusion_models/krea2_turbo_fp8_scaled.safetensors
download text_encoders/qwen3vl_4b_fp8_scaled.safetensors
download vae/qwen_image_vae.safetensors
echo "krea-weights: complete in $(( $(date +%s) - start_ts ))s"
