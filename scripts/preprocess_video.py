#!/usr/bin/env python3
"""
Preprocess clip-based video-frame datasets for local video-stage experiments.

Input layout (recommended):
  data/video_raw/<clip_dir>/{frame_*.png, clip.txt(optional)}

Output layout:
  <output_dir>/
    train/<clip_id>/000000.jpg ...
    val/<clip_id>/000000.jpg ...
    metadata_train.jsonl
    metadata_val.jsonl
    source_map_train.jsonl
    source_map_val.jsonl
    stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required for preprocess_video.py. Install with: pip install pillow"
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Stats:
    scanned_clips: int = 0
    valid_clips: int = 0
    invalid_clips: int = 0
    train_clips: int = 0
    val_clips: int = 0
    frames_exported: int = 0
    captions_found: int = 0
    captions_missing: int = 0


def to_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def iter_clip_dirs(root: Path) -> list[Path]:
    clips: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        has_frames = any(
            c.is_file() and c.suffix.lower() in IMAGE_SUFFIXES for c in p.iterdir()
        )
        if has_frames:
            clips.append(p)
    return sorted(clips)


def list_frames(clip_dir: Path) -> list[Path]:
    frames = [
        p
        for p in clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(frames)


def sample_frames(frames: list[Path], stride: int, max_frames: int) -> list[Path]:
    sampled = frames[::stride]
    if max_frames > 0 and len(sampled) > max_frames:
        # Uniformly keep max_frames from the sampled sequence.
        indices = [int(i * len(sampled) / max_frames) for i in range(max_frames)]
        sampled = [sampled[i] for i in indices]
    return sampled


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.BICUBIC)


def process_frame(src: Path, dst: Path, size: int | None, copy_only: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
        return

    with Image.open(src) as img:
        if size:
            img = center_crop_resize(img, size)
        else:
            img = img.convert("RGB")
        img.save(dst, quality=95)


def read_caption(clip_dir: Path, caption_file: str) -> str:
    cap_path = clip_dir / caption_file
    if not cap_path.exists():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return cap_path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return cap_path.read_text(encoding="utf-8", errors="replace").strip()


def clip_id_from_path(root: Path, clip_dir: Path) -> str:
    rel = clip_dir.relative_to(root)
    return "__".join(rel.parts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_split(
    split_name: str,
    clip_dirs: list[Path],
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], int, int, int]:
    metadata_rows: list[dict] = []
    mapping_rows: list[dict] = []
    captions_found = 0
    captions_missing = 0
    exported_frames = 0

    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for clip_dir in clip_dirs:
        clip_id = clip_id_from_path(input_root, clip_dir)
        frames = list_frames(clip_dir)
        frames = sample_frames(frames, args.frame_stride, args.max_frames)
        if len(frames) < args.min_frames:
            continue

        clip_out_dir = split_dir / clip_id
        frame_rel_paths: list[str] = []
        for idx, src_frame in enumerate(frames):
            dst_name = f"{idx:06d}.jpg"
            dst_path = clip_out_dir / dst_name
            process_frame(src_frame, dst_path, args.size, args.copy_only)

            rel_path = Path(split_name) / clip_id / dst_name
            frame_rel_paths.append(to_posix(rel_path))

            mapping_rows.append(
                {
                    "clip_id": clip_id,
                    "source_frame": to_posix(src_frame.relative_to(input_root)),
                    "target_frame": to_posix(rel_path),
                }
            )

        caption_text = read_caption(clip_dir, args.caption_file)
        if caption_text:
            captions_found += 1
        else:
            captions_missing += 1

        metadata_rows.append(
            {
                "clip_id": clip_id,
                "frames": frame_rel_paths,
                "num_frames": len(frame_rel_paths),
                "text": caption_text,
            }
        )
        exported_frames += len(frame_rel_paths)

    return metadata_rows, mapping_rows, captions_found, captions_missing, exported_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess clip-based video frame datasets.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512, help="Square output frame size")
    parser.add_argument("--copy-only", action="store_true", help="Skip resize/crop")
    parser.add_argument("--caption-file", type=str, default="clip.txt")
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32, help="0 means no limit")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"input dir not found: {args.input_dir}")
    if not (0 < args.val_ratio < 1):
        raise SystemExit("--val-ratio must be between 0 and 1")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    if args.min_frames < 2:
        raise SystemExit("--min-frames must be >= 2")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")

    random.seed(args.seed)
    stats = Stats()

    all_clips = iter_clip_dirs(args.input_dir)
    stats.scanned_clips = len(all_clips)

    valid_clips: list[Path] = []
    for clip_dir in all_clips:
        frames = sample_frames(list_frames(clip_dir), args.frame_stride, args.max_frames)
        if len(frames) >= args.min_frames:
            valid_clips.append(clip_dir)

    stats.valid_clips = len(valid_clips)
    stats.invalid_clips = stats.scanned_clips - stats.valid_clips

    random.shuffle(valid_clips)
    split_idx = int(len(valid_clips) * (1 - args.val_ratio))
    train_clips = valid_clips[:split_idx]
    val_clips = valid_clips[split_idx:]

    train_meta, train_map, train_caps_found, train_caps_missing, train_frames = export_split(
        split_name="train",
        clip_dirs=train_clips,
        input_root=args.input_dir,
        output_root=args.output_dir,
        args=args,
    )
    val_meta, val_map, val_caps_found, val_caps_missing, val_frames = export_split(
        split_name="val",
        clip_dirs=val_clips,
        input_root=args.input_dir,
        output_root=args.output_dir,
        args=args,
    )

    stats.train_clips = len(train_meta)
    stats.val_clips = len(val_meta)
    stats.frames_exported = train_frames + val_frames
    stats.captions_found = train_caps_found + val_caps_found
    stats.captions_missing = train_caps_missing + val_caps_missing

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, ensure_ascii=False, indent=2)

    write_jsonl(args.output_dir / "metadata_train.jsonl", train_meta)
    write_jsonl(args.output_dir / "metadata_val.jsonl", val_meta)
    write_jsonl(args.output_dir / "source_map_train.jsonl", train_map)
    write_jsonl(args.output_dir / "source_map_val.jsonl", val_map)

    print("Video preprocess done.")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
