#!/usr/bin/env python3
"""Generate baseline vs LoRA comparison images with fixed prompts and seeds."""

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
from diffusers import StableDiffusionPipeline


DEFAULT_PROMPTS = [
    "a cozy riverside town at sunset, hand-drawn animation style, warm light",
    "a small bakery street in spring, whimsical anime illustration, soft clouds",
    "a countryside train passing through green fields, cinematic animated frame",
    "a quiet bedroom with morning sunlight, watercolor anime style, detailed scene",
    "an old stone bridge over clear water, storybook animation look, gentle colors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline vs LoRA comparisons.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument(
        "--weight-name",
        type=str,
        default="pytorch_lora_weights.safetensors",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/compare_lora"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, watermark, text",
    )
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


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lora").mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(args.cache_dir.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache_dir / "hub").resolve()))

    prompts = load_prompts(args.prompt_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=args.cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    records = []

    for idx, prompt in enumerate(prompts):
        prompt_seed = args.seed + idx
        gen = torch.Generator(device=device.type).manual_seed(prompt_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]
        out_path = args.output_dir / "baseline" / f"prompt_{idx:02d}.png"
        image.save(out_path)
        records.append(
            {
                "idx": idx,
                "seed": prompt_seed,
                "prompt": prompt,
                "baseline_image": str(out_path).replace("\\", "/"),
            }
        )
        print(f"baseline saved: {out_path}")

    pipe.load_lora_weights(args.lora_path, weight_name=args.weight_name)
    for idx, prompt in enumerate(prompts):
        prompt_seed = args.seed + idx
        gen = torch.Generator(device=device.type).manual_seed(prompt_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]
        out_path = args.output_dir / "lora" / f"prompt_{idx:02d}.png"
        image.save(out_path)
        records[idx]["lora_image"] = str(out_path).replace("\\", "/")
        print(f"lora saved: {out_path}")

    with open(args.output_dir / "comparison_manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"manifest saved: {args.output_dir / 'comparison_manifest.json'}")


if __name__ == "__main__":
    main()
