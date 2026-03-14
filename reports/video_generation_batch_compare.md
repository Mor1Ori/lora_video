# Video Generation Batch Compare (2026-03-15)

## Scope

Batch compare baseline vs LoRA style generation on 5 prompts.

Generation script:

- `scripts/generate_video_style_batch.py`

Evaluation script:

- `scripts/eval_video_consistency.py`

## Artifacts

Generation outputs:

- Baseline clips: `artifacts/video_gen_baseline_batch/`
- LoRA clips: `artifacts/video_gen_lora_r8_s400_batch/`
- Batch manifest: `artifacts/video_batch_manifest.json`

Evaluation outputs:

- Baseline metrics: `artifacts/video_eval/consistency_baseline_batch.json`
- LoRA metrics: `artifacts/video_eval/consistency_lora_batch.json`

## Summary Metrics (5 clips, 12 frames each)

Baseline:

- mean_abs_diff_mean: `0.13878863447091797`
- mean_edge_diff_mean: `0.06251895180480047`
- luminance_std_mean: `0.01675243514084985`

LoRA:

- mean_abs_diff_mean: `0.12985831438140436`
- mean_edge_diff_mean: `0.060860569300976664`
- luminance_std_mean: `0.018716047805856874`

Delta (LoRA - Baseline):

- mean_abs_diff_mean: `-0.00893032008951361`
- mean_edge_diff_mean: `-0.001658382503823802`
- luminance_std_mean: `+0.001963612665007025`

## Interpretation

- LoRA is better on temporal smoothness/structure in this batch (`mean_abs_diff`, `mean_edge_diff` lower).
- LoRA has slightly higher brightness fluctuation (`luminance_std` higher).
- This is already stronger than a single-clip smoke result, but still belongs to a small-sample validation stage.

## Next Step

Increase to 10 to 20 prompts/clips with fixed seed policy, then decide whether to lock current LoRA video settings before cloud-scale runs.
