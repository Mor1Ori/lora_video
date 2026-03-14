#!/usr/bin/env python3
"""Generate a short style video clip (frame sequence + GIF) with SD1.5 and optional LoRA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Set cache paths before importing diffusers/transformers.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT.resolve()))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((_CACHE_ROOT / "hub").resolve()))

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short style video clip.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/video_gen"))
    parser.add_argument("--clip-id", type=str, default="clip_0001")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, watermark, text",
    )
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument(
        "--weight-name",
        type=str,
        default="pytorch_lora_weights.safetensors",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        return
    duration_ms = max(1, int(1000 / max(1, fps)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    args = parse_args()
    if args.num_frames < 2:
        raise SystemExit("--num-frames must be >= 2")
    if not (0.0 < args.strength < 1.0):
        raise SystemExit("--strength must be between 0 and 1")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_dir.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache_dir / "hub").resolve()))

    out_root = args.output_dir / args.clip_id
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    txt2img = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=args.cache_dir,
    ).to(device)
    txt2img.set_progress_bar_config(disable=False)

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=args.cache_dir,
    ).to(device)
    img2img.set_progress_bar_config(disable=False)

    if args.lora_path is not None:
        txt2img.load_lora_weights(args.lora_path, weight_name=args.weight_name)
        img2img.load_lora_weights(args.lora_path, weight_name=args.weight_name)

    frames: list[Image.Image] = []
    frame_paths: list[str] = []

    # Frame 0 from text-to-image.
    gen0 = torch.Generator(device=device.type).manual_seed(args.seed)
    current = txt2img(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=gen0,
    ).images[0]
    frames.append(current)

    # Subsequent frames via img2img chain for temporal continuity.
    for idx in range(1, args.num_frames):
        gen = torch.Generator(device=device.type).manual_seed(args.seed + idx)
        current = img2img(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=current,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        frames.append(current)

    for idx, frame in enumerate(frames):
        path = out_root / f"{idx:06d}.png"
        frame.save(path)
        frame_paths.append(str(path).replace("\\", "/"))
        print(f"saved frame: {path}")

    gif_path = args.output_dir / f"{args.clip_id}.gif"
    save_gif(frames, gif_path, fps=args.fps)
    print(f"saved gif: {gif_path}")

    manifest = {
        "clip_id": args.clip_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "lora_path": None if args.lora_path is None else str(args.lora_path).replace("\\", "/"),
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "fps": args.fps,
        "frames": frame_paths,
        "gif": str(gif_path).replace("\\", "/"),
    }
    with open(args.output_dir / f"{args.clip_id}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved manifest: {args.output_dir / f'{args.clip_id}_manifest.json'}")


if __name__ == "__main__":
    main()
