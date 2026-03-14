# Video Stage Protocol

## Goal

Build a reproducible video-stage loop:

1. clip/frame dataset preprocessing
2. video model training/inference (next phase)
3. temporal consistency evaluation and logging

This document focuses on data and evaluation design for phase-1 video onboarding.

## 1. Data Design (Clip-Based)

Recommended raw layout:

```text
data/video_raw/
|- clip_0001/
|  |- frame_0001.png
|  |- frame_0002.png
|  `- clip.txt          # optional caption for the whole clip
|- clip_0002/
|  |- 000001.jpg
|  `- ...
`- ...
```

Design principles:

- Split by clip, not by frame (avoid train/val leakage).
- Keep temporal order by filename sorting.
- Use one caption per clip for the first phase.
- Enforce minimum frame count per clip.

## 2. Preprocessing

Script: `scripts/preprocess_video.py`

Example command:

```bash
python scripts/preprocess_video.py --input-dir data/video_raw --output-dir data/video_processed --size 512 --val-ratio 0.2 --min-frames 16 --max-frames 32 --frame-stride 1 --caption-file clip.txt
```

Main outputs:

- `data/video_processed/train/<clip_id>/000000.jpg ...`
- `data/video_processed/val/<clip_id>/000000.jpg ...`
- `data/video_processed/metadata_train.jsonl`
- `data/video_processed/metadata_val.jsonl`
- `data/video_processed/source_map_train.jsonl`
- `data/video_processed/source_map_val.jsonl`
- `data/video_processed/stats.json`

Metadata schema (jsonl row):

```json
{
  "clip_id": "clip_0001",
  "frames": [
    "train/clip_0001/000000.jpg",
    "train/clip_0001/000001.jpg"
  ],
  "num_frames": 32,
  "text": "a windy field in animation style"
}
```

## 3. Evaluation Design (No-Reference Temporal Metrics)

Script: `scripts/eval_video_consistency.py`

Example command:

```bash
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen --output artifacts/video_eval/consistency_report.json --min-frames 8
```

Current metrics:

- `mean_abs_diff`: average pixel-level frame-to-frame difference
- `mean_edge_diff`: average frame-to-frame edge-map difference
- `luminance_std`: std of clip brightness across frames

Interpretation:

- Lower `mean_abs_diff` often means smoother motion transitions.
- Lower `mean_edge_diff` often means better structural stability.
- Lower `luminance_std` often means less global flicker.

## 3.5 First Video Generation Baseline

Script: `scripts/generate_video_style.py`

Example (baseline):

```bash
python scripts/generate_video_style.py --output-dir artifacts/video_gen_baseline --clip-id clip_0001 --prompt "a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement" --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
```

Example (with LoRA style):

```bash
python scripts/generate_video_style.py --output-dir artifacts/video_gen_lora_r8_s400 --clip-id clip_0001 --prompt "a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement" --lora-path artifacts/lora_image_ghibli_r8_s400 --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
```

Batch compare on 5 prompts:

```bash
python scripts/generate_video_style_batch.py --baseline-output-dir artifacts/video_gen_baseline_batch --lora-output-dir artifacts/video_gen_lora_r8_s400_batch --lora-path artifacts/lora_image_ghibli_r8_s400 --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_baseline_batch --output artifacts/video_eval/consistency_baseline_batch.json --min-frames 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_lora_r8_s400_batch --output artifacts/video_eval/consistency_lora_batch.json --min-frames 8
```

## 4. Experiment Rules (Video Stage)

- Change one variable per run.
- Keep prompt set and seed fixed when comparing methods.
- Log each run with:
  - config
  - generated clip path
  - metric report path
  - qualitative notes

## 5. Suggested Next Run

1. Prepare 20 to 50 clips in `data/video_raw`.
2. Run `preprocess_video.py` to create `data/video_processed`.
3. Generate first batch of clips from your video model.
4. Run `eval_video_consistency.py`.
5. Add results to weekly report with same format as image stage.
