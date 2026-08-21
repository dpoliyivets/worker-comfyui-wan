#!/usr/bin/env bash
# MiniMax H3 serverless lazy-weights bootstrap (mirrors download-ltx-weights.sh
# and download-krea-weights.sh).
#
# The H3 image ships WITHOUT baked weights (the LTX experience: a baked image is
# pull-bandwidth-bound at cold start, so a bigger image is strictly SLOWER — see
# docs/video-generation-canonical.md §12.4). Each worker downloads its variant's
# weights into container disk on cold start, ~44 GB for the default ref2va set:
#
#   diffusion_models/minimax_h3_<variant>_pruned_int8_convrot.safetensors  20.97 GB
#   text_encoders/qwen3vl_32b_minimax_h3_<enc>.safetensors                 15.69 GB (nvfp4)
#   vae/minimax_h3_video_vae_fp16.safetensors                               5.21 GB
#   vae/minimax_h3_audio_vae_fp32.safetensors                               0.61 GB
#   loras/minimax_h3_<ref2v|fl2v>_turbo_*step_*.safetensors                 1.96 GB
#
# Everything comes from the PUBLIC Comfy-Org/MiniMax-H3 repack. HF serves these
# through Xet, where aria2/wget byte-range GETs 403 ([[project_hf_xet_aria2_fails]]),
# so we use the Xet-native hf client with hf_transfer.
#
# INT8-convrot is deliberate, not a size compromise: every community NSFW H3 LoRA
# is trained against the pruned int8 build, and LoRAs bind only partially to other
# quants (working doc M-54).
#
# Required env: H3_VARIANT=ref2va|fl2va
# Optional env: H3_TEXT_ENCODER=nvfp4|int8|bf16 (default nvfp4 — Blackwell only;
#               use int8 on Hopper/Ada), H3_SKIP_TURBO_LORA=1
set -euo pipefail

: "${H3_VARIANT:?H3_VARIANT must be ref2va|fl2va}"
: "${H3_TEXT_ENCODER:=nvfp4}"

# HF_XET_HIGH_PERFORMANCE is the current switch; the old HF_TRANSFER var stays for
# hub versions that still read it (newer ones just FutureWarn on it).
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS_DIR=/comfyui/models
REPO=Comfy-Org/MiniMax-H3

case "$H3_VARIANT" in
    ref2va) TURBO_LORA=loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors ;;
    fl2va)  TURBO_LORA=loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors ;;
    *) echo "h3-weights: unknown H3_VARIANT=$H3_VARIANT" >&2; exit 1 ;;
esac

case "$H3_TEXT_ENCODER" in
    nvfp4) ENCODER=text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors ;;
    int8)  ENCODER=text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ;;
    bf16)  ENCODER=text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors ;;
    *) echo "h3-weights: unknown H3_TEXT_ENCODER=$H3_TEXT_ENCODER" >&2; exit 1 ;;
esac

download() {
    local rel_path="$1"
    local dest="$MODELS_DIR/$rel_path"
    if [ -s "$dest" ]; then
        echo "h3-weights: cached $rel_path"
        return 0
    fi
    echo "h3-weights: downloading $rel_path"
    hf download "$REPO" "$rel_path" --local-dir /tmp/hf-h3
    mkdir -p "$(dirname "$dest")"
    mv "/tmp/hf-h3/$rel_path" "$dest"
    rm -rf /tmp/hf-h3
}

start_ts=$(date +%s)
echo "h3-weights: bootstrapping variant=$H3_VARIANT encoder=$H3_TEXT_ENCODER"
download "diffusion_models/minimax_h3_${H3_VARIANT}_pruned_int8_convrot.safetensors"
download "$ENCODER"
download vae/minimax_h3_video_vae_fp16.safetensors
download vae/minimax_h3_audio_vae_fp32.safetensors
if [ -z "${H3_SKIP_TURBO_LORA:-}" ]; then
    download "$TURBO_LORA"
fi
echo "h3-weights: complete in $(( $(date +%s) - start_ts ))s"
