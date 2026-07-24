"""Krea 2 flat-params job expansion for the krea-render serverless endpoint.

The scene-worker submits `{"input": {"krea": {...}}}` with the flat contract
(prompt/negatives/sampler/scheduler/steps/cfg/resolution/seed/loras[]) instead
of a raw ComfyUI workflow. This module downloads the requested LoRAs from R2
into the local lora dir (cached across warm jobs) and expands the params into
the ComfyUI graph validated by the 2026-07-23 backbone bench (UNETLoader →
LoraLoaderModelOnly chain → KSampler → VAEDecode → SaveImage).

Checkpoint/encoder/VAE files are baked into the image; LoRAs are never baked —
bundle edits in Studio need no image rebuild.
"""

import logging
import os

logger = logging.getLogger(__name__)

KREA_UNET_NAME = os.environ.get("KREA_UNET_NAME", "krea2_turbo_fp8_scaled.safetensors")
KREA_CLIP_NAME = os.environ.get("KREA_CLIP_NAME", "qwen3vl_4b_fp8_scaled.safetensors")
KREA_VAE_NAME = os.environ.get("KREA_VAE_NAME", "qwen_image_vae.safetensors")
LORA_DIR = "/comfyui/models/loras"

REQUIRED_FIELDS = (
    "cfg",
    "loras",
    "negatives",
    "prompt",
    "resolution",
    "sampler",
    "scheduler",
    "seed",
    "steps",
)


def _default_s3_client():
    """R2 client from endpoint env (R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)."""
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )


def ensure_loras(loras, lora_dir=LORA_DIR, s3_client=None, bucket=None):
    """Download each r2:// LoRA into lora_dir unless already cached.

    Returns the local filenames in request order. Only r2:// sources are
    accepted — arbitrary URLs would let a compromised job pull attacker
    weights.
    """
    filenames = []
    for lora in loras:
        source = lora["source"]
        if not source.startswith("r2://"):
            raise ValueError(f"Unsupported lora source (only r2:// allowed): {source}")
        key = source[len("r2://") :]
        filename = os.path.basename(key)
        if not filename.endswith(".safetensors"):
            raise ValueError(f"Lora source must be a .safetensors object: {source}")
        local_path = os.path.join(lora_dir, filename)
        if os.path.exists(local_path):
            logger.info("krea: lora cached: %s", filename)
        else:
            if s3_client is None:
                s3_client = _default_s3_client()
            if bucket is None:
                bucket = os.environ["R2_BUCKET"]
            logger.info("krea: downloading lora %s from r2://%s", filename, key)
            os.makedirs(lora_dir, exist_ok=True)
            tmp_path = f"{local_path}.part"
            s3_client.download_file(bucket, key, tmp_path)
            os.replace(tmp_path, local_path)
        filenames.append(filename)
    return filenames


def build_krea_workflow(params, lora_filenames):
    """Expand flat params + resolved lora filenames into the bench ComfyUI graph."""
    resolution = params["resolution"]
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": KREA_UNET_NAME, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": KREA_CLIP_NAME, "type": "krea2"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA_VAE_NAME}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": params["prompt"]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": params["negatives"]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": resolution["width"],
                "height": resolution["height"],
                "batch_size": 1,
            },
        },
    }

    model_link = ["1", 0]
    for index, (lora, filename) in enumerate(zip(params["loras"], lora_filenames)):
        node_id = str(10 + index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_link,
                "lora_name": filename,
                "strength_model": lora["strength"],
            },
        }
        model_link = [node_id, 0]

    workflow["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_link,
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": params["seed"],
            "steps": params["steps"],
            "cfg": params["cfg"],
            "sampler_name": params["sampler"],
            "scheduler": params["scheduler"],
            "denoise": 1.0,
        },
    }
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "krea"},
    }
    return workflow


def expand_krea_input(krea_params, ensure=ensure_loras):
    """Validate flat params, fetch LoRAs, and return a standard workflow job input."""
    if not isinstance(krea_params, dict):
        raise ValueError("'krea' input must be an object")
    missing = [field for field in REQUIRED_FIELDS if field not in krea_params]
    if missing:
        raise ValueError(f"'krea' input missing fields: {', '.join(missing)}")
    resolution = krea_params["resolution"]
    if not isinstance(resolution, dict) or "width" not in resolution or "height" not in resolution:
        raise ValueError("'krea' resolution must be {width, height}")
    lora_filenames = ensure(krea_params["loras"])
    return {"workflow": build_krea_workflow(krea_params, lora_filenames)}
