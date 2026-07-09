#!/usr/bin/env bash
# LTX-2.3 serverless lazy-weights bootstrap.
#
# The LTX serverless image ships WITHOUT baked weights (a baked 22B checkpoint
# makes the image ~62GB, which no CI/registry/pod we have can build+push).
# Each worker downloads its variant's weights into container disk on cold start.
# File set mirrors ltx-base-weights.json + collectAllLtxLoras.ts +
# collectAllLtxCheckpoints.ts (the canonical pod recipe).
#
# Download method matters: the public-HF weights live on repos HuggingFace has
# migrated to **Xet** (resolve URLs redirect to us.aws.cdn.hf.co/xet-bridge-us/
# with byte-range-scoped signed URLs). aria2's multi-connection byte-range
# requests get HTTP 403 there ([[project_hf_xet_aria2_fails]]), so HF files go
# through the **hf client** (Xet-native + hf_transfer → fast, no 403). The
# R2-mirrored 4 NSFW LoRAs (small) download cleanly with `aws s3 cp`. Split by
# LTX_VARIANT: both variants pull the shared base weights + 4 LoRAs; the 22B
# checkpoint differs (sfw → official dev-fp8 on HF, nsfw → 10Eros on a PRIVATE
# HF mirror `dpoliyivets/ltx-2.3-10eros-fp8`). The 10Eros checkpoint was moved
# off R2 → HF because R2 `aws s3 cp` ran ~17 MB/s single-stream (a 3477-part
# multipart object), a ~28 min blocking boot that starved the cold-start window;
# HF/Xet + hf_transfer restores parity with the fast SFW path. A sentinel skips
# re-download on a warm container. `hf download` auths via HF_TOKEN (endpoint env).
#
# Required env: LTX_VARIANT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
#               R2_ENDPOINT, R2_BUCKET
set -euo pipefail

MODELS=/comfyui/models
SENTINEL="$MODELS/.ltx_weights_done"
TMP=/tmp/hfdl

: "${LTX_VARIANT:?LTX_VARIANT must be sfw|nsfw}"
: "${R2_ACCESS_KEY_ID:?}"; : "${R2_SECRET_ACCESS_KEY:?}"; : "${R2_ENDPOINT:?}"; : "${R2_BUCKET:?}"

if [ -f "$SENTINEL" ]; then
  echo "ltx-weights: sentinel present — weights already downloaded, skipping"
  exit 0
fi

echo "ltx-weights: bootstrapping variant=$LTX_VARIANT (hf client for HF, aws for R2)"
mkdir -p "$MODELS"/checkpoints "$MODELS"/text_encoders "$MODELS"/loras "$MODELS"/latent_upscale_models "$TMP"
export HF_HUB_ENABLE_HF_TRANSFER=1

# hf_get <repo> <path-in-repo> <dest> — Xet-aware, follows repo subpath then moves flat.
hf_get() {
  hf download "$1" "$2" --local-dir "$TMP" >/dev/null
  mv -f "$TMP/$2" "$3"
}
# r2_get <key> <dest> — R2 via aws (creds from env; region auto keeps R2 happy).
r2_get() {
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  AWS_DEFAULT_REGION=auto aws s3 cp "s3://$R2_BUCKET/$1" "$2" --endpoint-url "$R2_ENDPOINT" --quiet
}

# All downloads run concurrently (each tool is internally multi-threaded); a
# failed background job is caught by the non-empty verification below, not $?.
# 1) Shared public-HF base weights (both variants).
hf_get Comfy-Org/ltx-2   split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
       "$MODELS/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" &
hf_get Comfy-Org/ltx-2.3 split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors \
       "$MODELS/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" &
hf_get Comfy-Org/ltx-2   split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors \
       "$MODELS/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" &
hf_get Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
       "$MODELS/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" &
# 2) R2-mirrored NSFW LoRAs (both variants).
for f in DR34ML4Y_LTXXX_V2 Penile_Praxis_V4 VBVR_I2V_V3 DT_K3NK_V3; do
  r2_get "ltx-loras/${f}.safetensors" "$MODELS/loras/${f}.safetensors" &
done
# 3) Variant 22B checkpoint.
if [ "$LTX_VARIANT" = "sfw" ]; then
  hf_get Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors \
         "$MODELS/checkpoints/ltx-2.3-22b-dev-fp8.safetensors" &
elif [ "$LTX_VARIANT" = "nsfw" ]; then
  # 10Eros mirrored to a PRIVATE HF repo so it downloads via the fast Xet/hf_transfer path
  # (~hundreds of MB/s), same as the SFW dev-fp8 checkpoint — R2 `aws s3 cp` was ~17 MB/s
  # single-stream (a 3477-part multipart object), a ~28 min blocking boot that starved the
  # cold-start window. `hf download` authenticates via the HF_TOKEN env set on the endpoint.
  hf_get dpoliyivets/ltx-2.3-10eros-fp8 10Eros_v1-fp8mixed_learned.safetensors \
         "$MODELS/checkpoints/10Eros_v1-fp8mixed_learned.safetensors" &
else
  echo "ltx-weights: LTX_VARIANT must be sfw|nsfw, got '$LTX_VARIANT'" >&2; exit 1
fi
wait

# Fail loud on any missing/empty file (a background download that died shows up here).
check() { test -s "$MODELS/$1" || { echo "ltx-weights: MISSING/EMPTY $1" >&2; exit 1; }; }
check text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
check loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
check loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
check latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
for f in DR34ML4Y_LTXXX_V2 Penile_Praxis_V4 VBVR_I2V_V3 DT_K3NK_V3; do check "loras/${f}.safetensors"; done
if [ "$LTX_VARIANT" = "sfw" ]; then check checkpoints/ltx-2.3-22b-dev-fp8.safetensors
else check checkpoints/10Eros_v1-fp8mixed_learned.safetensors; fi

rm -rf "$TMP"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SENTINEL"
echo "ltx-weights: bootstrap complete"
