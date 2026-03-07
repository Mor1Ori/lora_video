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
    captions_found: int = 0
    captions_missing: int = 0


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


def to_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def read_text_file(path: Path) -> str:
    # Try common encodings, then fallback with replacement.
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_caption(src_image: Path, caption_ext: str) -> tuple[str, Path | None]:
    text_path = src_image.with_suffix(caption_ext)
    if not text_path.exists():
        return "", None
    return read_text_file(text_path), text_path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_split(
    split_name: str,
    images: list[Path],
    split_dir: Path,
    args: argparse.Namespace,
    input_dir: Path,
) -> tuple[list[dict], list[dict], int, int]:
    metadata_rows: list[dict] = []
    mapping_rows: list[dict] = []
    found = 0
    missing = 0

    for i, src in enumerate(images):
        dst = split_dir / f"{i:06d}.jpg"
        dst_rel = Path(split_name) / dst.name

        if args.copy_only:
            shutil.copy2(src, dst)
        else:
            process_image(src, dst, args.size)

        caption, caption_path = read_caption(src, args.caption_ext)
        if caption_path is None:
            missing += 1
        else:
            found += 1

        metadata_rows.append(
            {
                "file_name": to_posix(dst_rel),
                "text": caption,
            }
        )

        try:
            source_image = to_posix(src.relative_to(input_dir))
        except ValueError:
            source_image = to_posix(src)

        if caption_path is None:
            source_caption = None
        else:
            try:
                source_caption = to_posix(caption_path.relative_to(input_dir))
            except ValueError:
                source_caption = to_posix(caption_path)

        mapping_rows.append(
            {
                "source_image": source_image,
                "source_caption": source_caption,
                "target_image": to_posix(dst_rel),
            }
        )

    return metadata_rows, mapping_rows, found, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image dataset.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512, help="Square output size")
    parser.add_argument("--copy-only", action="store_true", help="Skip resize/crop")
    parser.add_argument(
        "--caption-ext",
        type=str,
        default=".txt",
        help="Sidecar caption extension, e.g. .txt",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    if not (0 < args.val_ratio < 1):
        raise SystemExit("--val-ratio must be between 0 and 1")
    if not args.caption_ext.startswith("."):
        raise SystemExit("--caption-ext must start with '.' (example: .txt)")

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

    train_metadata, train_map, train_found, train_missing = export_split(
        split_name="train",
        images=train_imgs,
        split_dir=train_dir,
        args=args,
        input_dir=args.input_dir,
    )
    val_metadata, val_map, val_found, val_missing = export_split(
        split_name="val",
        images=val_imgs,
        split_dir=val_dir,
        args=args,
        input_dir=args.input_dir,
    )

    stats.train = len(train_imgs)
    stats.val = len(val_imgs)
    stats.captions_found = train_found + val_found
    stats.captions_missing = train_missing + val_missing

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, ensure_ascii=False, indent=2)
    write_jsonl(args.output_dir / "metadata_train.jsonl", train_metadata)
    write_jsonl(args.output_dir / "metadata_val.jsonl", val_metadata)
    write_jsonl(args.output_dir / "source_map_train.jsonl", train_map)
    write_jsonl(args.output_dir / "source_map_val.jsonl", val_map)

    print("Preprocess done.")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
