#!/usr/bin/env python3
"""Compute simple no-reference temporal consistency metrics on generated clips."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ClipMetrics:
    clip_id: str
    num_frames: int
    mean_abs_diff: float
    mean_edge_diff: float
    luminance_std: float


def list_frames(clip_dir: Path) -> list[Path]:
    frames = [
        p
        for p in clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(frames)


def find_clip_dirs(root: Path) -> list[Path]:
    clips: list[Path] = []

    # If root itself is a clip directory, include it.
    if any(
        c.is_file() and c.suffix.lower() in IMAGE_SUFFIXES for c in root.iterdir()
    ):
        clips.append(root)

    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        has_frames = any(
            c.is_file() and c.suffix.lower() in IMAGE_SUFFIXES for c in p.iterdir()
        )
        if has_frames:
            clips.append(p)
    # De-duplicate and stable sort.
    return sorted(set(clips))


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.sqrt(gx * gx + gy * gy)


def compute_clip_metrics(clip_dir: Path, min_frames: int) -> ClipMetrics | None:
    frame_paths = list_frames(clip_dir)
    if len(frame_paths) < min_frames:
        return None

    prev_rgb: np.ndarray | None = None
    prev_edge: np.ndarray | None = None
    abs_diffs: list[float] = []
    edge_diffs: list[float] = []
    lum_values: list[float] = []

    for p in frame_paths:
        rgb = load_rgb_float(p)
        gray = np.mean(rgb, axis=2)
        edge = gradient_magnitude(gray)
        lum_values.append(float(np.mean(gray)))

        if prev_rgb is not None:
            abs_diffs.append(float(np.mean(np.abs(rgb - prev_rgb))))
        if prev_edge is not None:
            edge_diffs.append(float(np.mean(np.abs(edge - prev_edge))))

        prev_rgb = rgb
        prev_edge = edge

    clip_id = clip_dir.name
    return ClipMetrics(
        clip_id=clip_id,
        num_frames=len(frame_paths),
        mean_abs_diff=float(np.mean(abs_diffs)) if abs_diffs else 0.0,
        mean_edge_diff=float(np.mean(edge_diffs)) if edge_diffs else 0.0,
        luminance_std=float(np.std(lum_values)) if lum_values else 0.0,
    )


def summarize(metrics: list[ClipMetrics]) -> dict:
    if not metrics:
        return {
            "clip_count": 0,
            "mean_num_frames": 0.0,
            "mean_abs_diff_mean": 0.0,
            "mean_edge_diff_mean": 0.0,
            "luminance_std_mean": 0.0,
            "note": "no valid clips found",
        }

    return {
        "clip_count": len(metrics),
        "mean_num_frames": statistics.mean(m.num_frames for m in metrics),
        "mean_abs_diff_mean": statistics.mean(m.mean_abs_diff for m in metrics),
        "mean_abs_diff_median": statistics.median(m.mean_abs_diff for m in metrics),
        "mean_edge_diff_mean": statistics.mean(m.mean_edge_diff for m in metrics),
        "mean_edge_diff_median": statistics.median(m.mean_edge_diff for m in metrics),
        "luminance_std_mean": statistics.mean(m.luminance_std for m in metrics),
        "luminance_std_median": statistics.median(m.luminance_std for m in metrics),
        "interpretation": {
            "mean_abs_diff": "Lower usually means smoother frame-to-frame changes.",
            "mean_edge_diff": "Lower usually means better structural temporal stability.",
            "luminance_std": "Lower usually means less global brightness flicker.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal consistency on clip directories."
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/video_eval/consistency_report.json"),
    )
    parser.add_argument("--min-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.clips_dir.exists():
        raise SystemExit(f"clips dir not found: {args.clips_dir}")
    if args.min_frames < 2:
        raise SystemExit("--min-frames must be >= 2")

    clip_dirs = find_clip_dirs(args.clips_dir)
    metrics: list[ClipMetrics] = []
    for clip_dir in clip_dirs:
        item = compute_clip_metrics(clip_dir, min_frames=args.min_frames)
        if item is not None:
            metrics.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "clips_dir": str(args.clips_dir).replace("\\", "/"),
        "summary": summarize(metrics),
        "per_clip": [asdict(m) for m in metrics],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {args.output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
