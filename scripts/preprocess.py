#!/usr/bin/env python3
"""
Minimal data preprocessing pipeline for local LoRA-video experiments.

Stages:
1) scan: collect image files
2) clean: basic integrity check
3) transform: optional resize and center crop
4) split: train/val split with fixed seed
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required for preprocess.py. Install with: pip install pillow"
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Stats:
    scanned: int = 0
    valid: int = 0
    invalid: int = 0
    train: int = 0
    val: int = 0


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def process_image(src: Path, dst: Path, size: int | None) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        if size:
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize((size, size))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image dataset.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512, help="Square output size")
    parser.add_argument("--copy-only", action="store_true", help="Skip resize/crop")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    if not (0 < args.val_ratio < 1):
        raise SystemExit("--val-ratio must be between 0 and 1")

    random.seed(args.seed)
    stats = Stats()

    all_imgs = list(iter_images(args.input_dir))
    stats.scanned = len(all_imgs)
    valid_imgs = [p for p in all_imgs if is_valid_image(p)]
    stats.valid = len(valid_imgs)
    stats.invalid = stats.scanned - stats.valid

    random.shuffle(valid_imgs)
    split_idx = int(len(valid_imgs) * (1 - args.val_ratio))
    train_imgs = valid_imgs[:split_idx]
    val_imgs = valid_imgs[split_idx:]

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(train_imgs):
        dst = train_dir / f"{i:06d}.jpg"
        if args.copy_only:
            shutil.copy2(src, dst)
        else:
            process_image(src, dst, args.size)
    for i, src in enumerate(val_imgs):
        dst = val_dir / f"{i:06d}.jpg"
        if args.copy_only:
            shutil.copy2(src, dst)
        else:
            process_image(src, dst, args.size)

    stats.train = len(train_imgs)
    stats.val = len(val_imgs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, ensure_ascii=False, indent=2)

    print("Preprocess done.")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
