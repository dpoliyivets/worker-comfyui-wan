# worker-comfyui-wan

RunPod serverless worker for **Wan2.2 Image-to-Video** generation. Forked from
`dpoliyivets/worker-comfyui`.

## What this worker does

- Receives a still image + text prompt + ComfyUI workflow JSON.
- Runs Wan2.2-I2V-A14B (MoE expert pair) via Kijai's `ComfyUI-WanVideoWrapper`.
- Combines frames into an mp4 via `ComfyUI-VideoHelperSuite`.
- Uploads the mp4 to RunPod's network-volume S3 bucket.
- Returns a presigned download URL in the job output.

## Network volume layout (read at runtime)

Models live on the shared network volume `5k8e4cnkpv` mounted at `/runpod-vol`:

- `models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8.safetensors`
- `models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8.safetensors`
- `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `models/vae/wan_2.1_vae.safetensors`
- `models/clip_vision/clip_vision_h.safetensors`

## Required env vars (set in RunPod endpoint dashboard)

| Name | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Network-volume S3 creds |
| `AWS_S3_BUCKET` | `5k8e4cnkpv` |
| `AWS_S3_ENDPOINT_URL` | `https://s3api-us-il-1.runpod.io` |
| `AWS_REGION` | `us-il-1` |
| `S3_OUTPUT_PREFIX` | `video-outputs/` |
