#!/usr/bin/env python3
"""Batch-generate baseline and LoRA style clips for a prompt set."""

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


DEFAULT_PROMPTS = [
    "a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement",
    "a small bakery street in spring, whimsical anime look, drifting camera",
    "a countryside train crossing green fields, cinematic animation frame sequence",
    "a quiet bedroom with morning sunlight, watercolor anime style, slow panning shot",
    "an old stone bridge over clear water, storybook animation, subtle motion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate baseline and LoRA clips for prompt comparison."
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument(
        "--baseline-output-dir",
        type=Path,
        default=Path("artifacts/video_gen_baseline_batch"),
    )
    parser.add_argument(
        "--lora-output-dir",
        type=Path,
        default=Path("artifacts/video_gen_lora_batch"),
    )
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument(
        "--weight-name",
        type=str,
        default="pytorch_lora_weights.safetensors",
    )
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, watermark, text",
    )
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def load_prompts(prompt_file: Path | None) -> list[str]:
    if prompt_file is None:
        return DEFAULT_PROMPTS
    if not prompt_file.exists():
        raise SystemExit(f"prompt file not found: {prompt_file}")
    prompts = []
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            prompts.append(line)
    if not prompts:
        raise SystemExit("prompt file is empty")
    return prompts


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


def build_pipelines(
    model_name: str,
    cache_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    lora_path: Path | None,
    weight_name: str,
) -> tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
    txt2img = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=cache_dir,
    ).to(device)
    txt2img.set_progress_bar_config(disable=False)

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=cache_dir,
    ).to(device)
    img2img.set_progress_bar_config(disable=False)

    if lora_path is not None:
        txt2img.load_lora_weights(lora_path, weight_name=weight_name)
        img2img.load_lora_weights(lora_path, weight_name=weight_name)

    return txt2img, img2img


def generate_one_clip(
    txt2img: StableDiffusionPipeline,
    img2img: StableDiffusionImg2ImgPipeline,
    prompt: str,
    clip_seed: int,
    clip_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    frame_paths: list[str] = []

    gen0 = torch.Generator(device=txt2img.device.type).manual_seed(clip_seed)
    current = txt2img(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=gen0,
    ).images[0]
    frames.append(current)

    for idx in range(1, args.num_frames):
        gen = torch.Generator(device=txt2img.device.type).manual_seed(clip_seed + idx)
        current = img2img(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            image=current,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        frames.append(current)

    for idx, frame in enumerate(frames):
        path = clip_dir / f"{idx:06d}.png"
        frame.save(path)
        frame_paths.append(str(path).replace("\\", "/"))

    gif_path = output_dir / f"{clip_id}.gif"
    save_gif(frames, gif_path, fps=args.fps)

    return {
        "clip_id": clip_id,
        "prompt": prompt,
        "seed": clip_seed,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "frames": frame_paths,
        "gif": str(gif_path).replace("\\", "/"),
    }


def generate_set(
    prompts: list[str],
    output_dir: Path,
    txt2img: StableDiffusionPipeline,
    img2img: StableDiffusionImg2ImgPipeline,
    args: argparse.Namespace,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, prompt in enumerate(prompts):
        clip_id = f"clip_{i + 1:04d}"
        clip_seed = args.seed + i * 1000
        print(f"[generate] {output_dir.name} {clip_id}")
        row = generate_one_clip(
            txt2img=txt2img,
            img2img=img2img,
            prompt=prompt,
            clip_seed=clip_seed,
            clip_id=clip_id,
            output_dir=output_dir,
            args=args,
        )
        records.append(row)
    return records


def main() -> None:
    args = parse_args()

    if args.num_frames < 2:
        raise SystemExit("--num-frames must be >= 2")
    if not (0.0 < args.strength < 1.0):
        raise SystemExit("--strength must be between 0 and 1")
    if not args.lora_path.exists():
        raise SystemExit(f"lora path not found: {args.lora_path}")

    prompts = load_prompts(args.prompt_file)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_dir.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache_dir / "hub").resolve()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Baseline set.
    txt2img_base, img2img_base = build_pipelines(
        model_name=args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        device=device,
        lora_path=None,
        weight_name=args.weight_name,
    )
    baseline_records = generate_set(
        prompts=prompts,
        output_dir=args.baseline_output_dir,
        txt2img=txt2img_base,
        img2img=img2img_base,
        args=args,
    )
    del txt2img_base, img2img_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LoRA set.
    txt2img_lora, img2img_lora = build_pipelines(
        model_name=args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        device=device,
        lora_path=args.lora_path,
        weight_name=args.weight_name,
    )
    lora_records = generate_set(
        prompts=prompts,
        output_dir=args.lora_output_dir,
        txt2img=txt2img_lora,
        img2img=img2img_lora,
        args=args,
    )

    summary = {
        "prompt_count": len(prompts),
        "prompt_file": None if args.prompt_file is None else str(args.prompt_file).replace("\\", "/"),
        "lora_path": str(args.lora_path).replace("\\", "/"),
        "baseline_output_dir": str(args.baseline_output_dir).replace("\\", "/"),
        "lora_output_dir": str(args.lora_output_dir).replace("\\", "/"),
        "generation": {
            "num_frames": args.num_frames,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "height": args.height,
            "width": args.width,
            "seed_base": args.seed,
            "fps": args.fps,
        },
        "baseline_records": baseline_records,
        "lora_records": lora_records,
    }
    out_manifest = args.lora_output_dir.parent / "video_batch_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved manifest: {out_manifest}")


if __name__ == "__main__":
    main()
