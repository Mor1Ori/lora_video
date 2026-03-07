#!/usr/bin/env python3
"""Generate sample images with a trained SD LoRA adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set local cache paths before importing transformers/diffusers.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT.resolve()))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((_CACHE_ROOT / "hub").resolve()))

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with SD LoRA.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/infer_lora"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, watermark, text",
    )
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(args.cache_dir.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache_dir / "hub").resolve()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=args.cache_dir,
    )
    pipe.load_lora_weights(args.lora_path)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device.type).manual_seed(args.seed)

    result = pipe(
        prompt=[args.prompt] * args.num_images,
        negative_prompt=[args.negative_prompt] * args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    for i, img in enumerate(result.images):
        out_path = args.output_dir / f"sample_{i:02d}.png"
        img.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
