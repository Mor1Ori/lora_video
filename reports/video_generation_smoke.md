# Video Generation Smoke Compare (2026-03-15)

## Scope

Generate and compare one baseline clip and one LoRA-style clip with the same prompt/seed.

## Generation Setup

- Prompt: `a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement`
- Seed: `42`
- Frames: `12`
- Steps: `20`
- Guidance scale: `7.5`
- Img2img chain strength: `0.30`
- Resolution: `512x512`

Baseline output:

- `artifacts/video_gen_baseline/clip_0001/`
- `artifacts/video_gen_baseline/clip_0001.gif`
- `artifacts/video_gen_baseline/clip_0001_manifest.json`

LoRA output:

- `artifacts/video_gen_lora_r8_s400/clip_0001/`
- `artifacts/video_gen_lora_r8_s400/clip_0001.gif`
- `artifacts/video_gen_lora_r8_s400/clip_0001_manifest.json`
- LoRA path: `artifacts/lora_image_ghibli_r8_s400`

## Temporal Consistency Metrics

Baseline report: `artifacts/video_eval/consistency_baseline_generated.json`

- mean_abs_diff_mean: `0.12721974064003339`
- mean_edge_diff_mean: `0.06028400022875179`
- luminance_std_mean: `0.015750190871054795`

LoRA report: `artifacts/video_eval/consistency_lora_generated.json`

- mean_abs_diff_mean: `0.11667193404652855`
- mean_edge_diff_mean: `0.055768551474267784`
- luminance_std_mean: `0.018009939353786256`

## Quick Read

- On this single-clip smoke run, LoRA clip has lower frame/edge diff (smoother structural transitions).
- LoRA clip has slightly higher luminance std (a bit more brightness variation).
- Because sample size is 1 clip, this is process validation only, not a final quality conclusion.

## Next Recommended Action

Run the same compare on 5 to 10 prompts (or clips) and use average/median metrics for a stable conclusion.
