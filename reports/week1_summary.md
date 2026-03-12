# Week 1 Summary

## Project
- Name: Style LoRA (Image-stage verification for later video extension)
- Date: 2026-03-13
- Owner: Cheng Sen

## Data Snapshot
- Source split: `data/processed/stats.json`
- Scanned: 100
- Valid: 100
- Train/Val: 80/20
- Captions found: 100

## Run Log
| Run ID | Date | Data Version | Config | Key Change | Result |
|---|---|---|---|---|---|
| lora-r8-s200 | 2026-03-08 | processed_512_v1 (80/20) | rank=8, steps=200, lr=1e-4, bs=1 | baseline rank | success |
| lora-r4-s200 | 2026-03-13 | processed_512_v1 (80/20) | rank=4, steps=200, lr=1e-4, bs=1 | lower rank | success |
| lora-r16-s200 | 2026-03-13 | processed_512_v1 (80/20) | rank=16, steps=200, lr=1e-4, bs=1 | higher rank | success |
| compare-r8 | 2026-03-08 | prompt_set_v1 | SD1.5 + LoRA(r8), steps=25, cfg=7.5 | baseline vs lora image compare | success |
| compare-r4 | 2026-03-13 | prompt_set_v1 | SD1.5 + LoRA(r4), steps=25, cfg=7.5 | baseline vs lora image compare | success |
| compare-r16 | 2026-03-13 | prompt_set_v1 | SD1.5 + LoRA(r16), steps=25, cfg=7.5 | baseline vs lora image compare | success |

## Metrics
- r4 best train loss: `0.002451554872095585`
- r8 best train loss: `0.0012350388569757342`
- r16 best train loss: `0.0028840149752795696`
- FVD: N/A (image stage)
- CLIP Score: N/A (not computed yet)

## Qualitative Check
- Prompt set: default 5 prompts in `scripts/generate_prompt_comparison.py`
- Output evidence:
  - `artifacts/compare_lora_r4_s200/`
  - `artifacts/compare_lora_r8_s200/`
  - `artifacts/compare_lora_r16_s200/`
- Initial observation:
  - rank=8 gives the best loss among tested ranks.
  - rank=4 and rank=16 both converge but currently underperform rank=8 on this 100-sample setup.

## Issues and Fixes
- Issue: compare script failed in offline mode with LoRA loading error.
- Root cause: `load_lora_weights` did not specify local `weight_name` under `HF_HUB_OFFLINE=1`.
- Fix: updated `scripts/generate_prompt_comparison.py` to pass `weight_name="pytorch_lora_weights.safetensors"` (and exposed as CLI arg).

## Next Actions (Top 3)
1. Lock `rank=8` as current baseline and run longer schedule (`steps=400/600`) to check stability.
2. Add quantitative image metrics (CLIP similarity / aesthetic proxy) for objective compare.
3. Start video-stage preparation: build frame-sequence dataset and define temporal consistency evaluation.
