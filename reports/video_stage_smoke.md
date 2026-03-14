# Video Stage Smoke Report (2026-03-15)

## Scope

Validate the new video-stage pipeline with the current mock clip dataset:

1. clip preprocessing
2. train/val split leakage check
3. temporal consistency metric export

## Dataset Used

- Source: `data/raw_ghibli100` (converted into mock clip folders for pipeline smoke)
- Working raw clips: `data/video_raw`
- Processed output: `data/video_processed`

## Preprocess Command

```bash
python scripts/preprocess_video.py --input-dir data/video_raw --output-dir data/video_processed --size 512 --val-ratio 0.2 --min-frames 16 --max-frames 32 --frame-stride 1 --caption-file clip.txt
```

## Preprocess Result

From `data/video_processed/stats.json`:

- scanned_clips: 5
- valid_clips: 5
- invalid_clips: 0
- train_clips: 4
- val_clips: 1
- frames_exported: 100
- captions_found: 5
- captions_missing: 0

## Split Leakage Check

Output: `artifacts/video_eval/split_check.json`

- train_clip_count: 4
- val_clip_count: 1
- overlap_count: 0
- no_leakage: true

## Temporal Consistency Evaluation

Train split report: `artifacts/video_eval/consistency_train_smoke.json`

- clip_count: 4
- mean_num_frames: 20
- mean_abs_diff_mean: 0.2838259992238722
- mean_edge_diff_mean: 0.0471962588643165
- luminance_std_mean: 0.11719541472734996

Val split report: `artifacts/video_eval/consistency_val_smoke.json`

- clip_count: 1
- mean_num_frames: 20
- mean_abs_diff_mean: 0.27836646531757553
- mean_edge_diff_mean: 0.04403110770018477
- luminance_std_mean: 0.1310943058177913

## Conclusion

Video-stage preprocessing and evaluation pipeline is working end-to-end.
Current metrics are valid for process verification, but not for final temporal-quality conclusions because the smoke dataset uses mock clip construction.
