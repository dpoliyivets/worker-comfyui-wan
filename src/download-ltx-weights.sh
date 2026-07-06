#!/usr/bin/env bash
# LTX-2.3 serverless lazy-weights bootstrap.
#
# The LTX serverless image ships WITHOUT baked weights (a baked 22B checkpoint
# makes the image ~62 GB, which no CI/registry/pod we have can build+push).
# Instead each worker downloads its variant's weights into container disk on
# cold start — exactly what the on-demand LtxPodService already does, just here
# in the serverless worker. The file set mirrors ltx-base-weights.json +
# collectAllLtxLoras.ts + collectAllLtxCheckpoints.ts (the canonical pod recipe).
#
# Split by LTX_VARIANT (sfw|nsfw): both variants pull the shared base weights +
# 4 NSFW LoRAs; the 22B checkpoint differs (sfw → official dev-fp8 on public HF,
# nsfw → 10Eros mirrored to R2). A worker only ever downloads ONE checkpoint.
#
# Download strategy: aria2c over a generated manifest (public HF direct URLs +
# R2 objects presigned on the fly with the worker's R2 creds). Same aria2 tuning
# the pod uses. A sentinel skips re-download when a warm container is reused.
#
# Required env: LTX_VARIANT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
#               R2_ENDPOINT, R2_BUCKET
set -euo pipefail

MODELS_DIR=/comfyui/models
SENTINEL="$MODELS_DIR/.ltx_weights_done"
MANIFEST=/tmp/ltx-weights.aria2
PRESIGN_EXPIRY=604800   # 7 days — long enough that a manifest built at boot never expires mid-download

: "${LTX_VARIANT:?LTX_VARIANT must be sfw|nsfw}"
: "${R2_ACCESS_KEY_ID:?}"; : "${R2_SECRET_ACCESS_KEY:?}"; : "${R2_ENDPOINT:?}"; : "${R2_BUCKET:?}"

if [ -f "$SENTINEL" ]; then
  echo "ltx-weights: sentinel present — weights already downloaded, skipping"
  exit 0
fi

echo "ltx-weights: bootstrapping variant=$LTX_VARIANT"
mkdir -p "$MODELS_DIR"/checkpoints "$MODELS_DIR"/text_encoders "$MODELS_DIR"/loras "$MODELS_DIR"/latent_upscale_models

# aria2 lines are `<url>\n  out=<path-relative-to--d>`. -d /comfyui below, so
# out=models/... lands in /comfyui/models/... (matches ComfyUI model dirs).
: > "$MANIFEST"
add() { printf '%s\n  out=%s\n\n' "$1" "$2" >> "$MANIFEST"; }

# R2 presign helper (awscli v1). --region auto keeps R2 happy; --endpoint-url
# targets the R2 account endpoint.
presign() {
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  AWS_DEFAULT_REGION=auto aws s3 presign "s3://$R2_BUCKET/$1" \
    --endpoint-url "$R2_ENDPOINT" --expires-in "$PRESIGN_EXPIRY"
}

# 1) Shared public-HF base weights (both variants) — text encoder, 2 base LoRAs,
#    spatial upscaler. Plain-LFS direct URLs (aria2-safe, NOT Xet).
HF=https://huggingface.co
add "$HF/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
    models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
add "$HF/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" \
    models/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
add "$HF/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
    models/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
add "$HF/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# 2) The 4 R2-mirrored NSFW LoRAs (both variants) — presigned.
for f in DR34ML4Y_LTXXX_V2 Penile_Praxis_V4 VBVR_I2V_V3 DT_K3NK_V3; do
  add "$(presign "ltx-loras/${f}.safetensors")" "models/loras/${f}.safetensors"
done

# 3) The VARIANT 22B checkpoint — sfw → official dev-fp8 (public HF), nsfw →
#    10Eros (R2, presigned). Exactly one per worker.
if [ "$LTX_VARIANT" = "sfw" ]; then
  add "$HF/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
      models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors
elif [ "$LTX_VARIANT" = "nsfw" ]; then
  add "$(presign "ltx-checkpoints/10Eros_v1-fp8mixed_learned.safetensors")" \
      models/checkpoints/10Eros_v1-fp8mixed_learned.safetensors
else
  echo "ltx-weights: LTX_VARIANT must be sfw|nsfw, got '$LTX_VARIANT'" >&2; exit 1
fi

echo "ltx-weights: downloading $(grep -c '^  out=' "$MANIFEST") files with aria2c…"
aria2c -i "$MANIFEST" -d /comfyui \
  -j 6 -x 8 -s 8 --auto-file-renaming=false --allow-overwrite=true \
  --console-log-level=warn --summary-interval=20

# Fail loud on any missing/empty file (a silent 404 here = a broken first render).
check() { test -s "$MODELS_DIR/$1" || { echo "ltx-weights: MISSING/EMPTY $1" >&2; exit 1; }; }
check text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
check loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
check loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
check latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
for f in DR34ML4Y_LTXXX_V2 Penile_Praxis_V4 VBVR_I2V_V3 DT_K3NK_V3; do check "loras/${f}.safetensors"; done
if [ "$LTX_VARIANT" = "sfw" ]; then check checkpoints/ltx-2.3-22b-dev-fp8.safetensors
else check checkpoints/10Eros_v1-fp8mixed_learned.safetensors; fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$SENTINEL"
echo "ltx-weights: bootstrap complete"
