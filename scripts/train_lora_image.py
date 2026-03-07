#!/usr/bin/env python3
"""
Minimal Stable Diffusion image LoRA trainer for local style verification.

Input format:
- data root contains image files referenced by metadata jsonl.
- metadata jsonl lines: {"file_name": "train/000000.jpg", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# Set local cache paths before importing transformers/diffusers.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT.resolve()))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((_CACHE_ROOT / "hub").resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class TrainState:
    global_step: int = 0
    epoch: int = 0
    best_loss: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image LoRA on SD.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--train-data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--metadata-file", type=Path, default=Path("data/processed/metadata_train.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lora_image"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--caption-dropout", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    image = image.crop((left, top, left + short, top + short))
    return image.resize((size, size), Image.Resampling.BICUBIC)


def pil_to_tensor_norm(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class JsonlImageTextDataset(Dataset):
    def __init__(
        self,
        train_data_dir: Path,
        metadata_file: Path,
        tokenizer: CLIPTokenizer,
        resolution: int,
        caption_dropout: float,
    ) -> None:
        self.train_data_dir = train_data_dir
        self.metadata = load_jsonl(metadata_file)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_dropout = caption_dropout

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata[idx]
        image_path = self.train_data_dir / row["file_name"]
        text = row.get("text", "") or ""

        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            text = ""

        image = Image.open(image_path)
        image = center_crop_resize(image, self.resolution)
        pixel_values = pil_to_tensor_norm(image)

        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def collate_fn(examples: list[dict]) -> dict:
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    input_ids = torch.stack([e["input_ids"] for e in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def add_lora_adapter(unet: UNet2DConditionModel, rank: int) -> list[torch.nn.Parameter]:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found after adapter injection.")
    return trainable_params


def save_state(
    output_dir: Path,
    unet: UNet2DConditionModel,
    args: argparse.Namespace,
    state: TrainState,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    with open(output_dir / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, default=str, indent=2)


def main() -> None:
    args = parse_args()

    if not args.metadata_file.exists():
        raise SystemExit(f"metadata file not found: {args.metadata_file}")
    if not args.train_data_dir.exists():
        raise SystemExit(f"train data dir not found: {args.train_data_dir}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_dir.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache_dir / "hub").resolve()))

    set_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.mixed_precision == "fp16":
        print("CPU detected, overriding mixed precision to 'no'.")
        args.mixed_precision = "no"
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_params = add_lora_adapter(unet, rank=args.rank)

    vae.to(device=device, dtype=weight_dtype)
    text_encoder.to(device=device, dtype=weight_dtype)
    unet.to(device=device, dtype=weight_dtype)
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = JsonlImageTextDataset(
        train_data_dir=args.train_data_dir,
        metadata_file=args.metadata_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        caption_dropout=args.caption_dropout,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    update_steps_per_epoch = math.ceil(
        len(train_dataloader) / max(1, args.gradient_accumulation_steps)
    )
    max_train_steps = min(
        args.max_train_steps, max(1, args.num_train_epochs * update_steps_per_epoch)
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    state = TrainState(global_step=0, epoch=0, best_loss=None)

    progress = tqdm(total=max_train_steps, desc="train")
    for epoch in range(args.num_train_epochs):
        state.epoch = epoch
        unet.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device=device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(args.mixed_precision == "fp16" and device.type == "cuda"),
            ):
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
                ).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            should_step = (step + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                state.global_step += 1
                step_loss = loss.item() * args.gradient_accumulation_steps
                if state.best_loss is None or step_loss < state.best_loss:
                    state.best_loss = step_loss
                progress.set_postfix({"loss": f"{step_loss:.4f}"})
                progress.update(1)

                if state.global_step % args.save_steps == 0:
                    ckpt_dir = args.output_dir / f"checkpoint-{state.global_step}"
                    save_state(ckpt_dir, unet, args, state)

                if state.global_step >= max_train_steps:
                    break

        if state.global_step >= max_train_steps:
            break

    progress.close()
    save_state(args.output_dir, unet, args, state)
    print(f"Training done. Steps={state.global_step}, best_loss={state.best_loss}")
    print(f"LoRA saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
