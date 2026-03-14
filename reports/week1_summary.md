# Week 1 Summary

## Project
- Name: Style LoRA (Image-stage verification for later video extension)
- Date: 2026-03-15
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
| lora-r8-s400 | 2026-03-15 | processed_512_v1 (80/20) | rank=8, epochs=5, max_steps=400, lr=1e-4, bs=1 | longer schedule attempt | success (actual step=400) |
| compare-r8 | 2026-03-08 | prompt_set_v1 | SD1.5 + LoRA(r8), steps=25, cfg=7.5 | baseline vs lora image compare | success |
| compare-r4 | 2026-03-13 | prompt_set_v1 | SD1.5 + LoRA(r4), steps=25, cfg=7.5 | baseline vs lora image compare | success |
| compare-r16 | 2026-03-13 | prompt_set_v1 | SD1.5 + LoRA(r16), steps=25, cfg=7.5 | baseline vs lora image compare | success |
| compare-r8-s400 | 2026-03-15 | prompt_set_v1 | SD1.5 + LoRA(r8_s400), steps=25, cfg=7.5 | baseline vs lora image compare | success |

## Metrics
- r4 best train loss: `0.002451554872095585`
- r8-s200 best train loss: `0.0012350388569757342` (global_step=200)
- r8-s400 run best train loss: `0.0012351041659712791` (global_step=400)
- r16 best train loss: `0.0028840149752795696`
- r8-s200 vs r8-s400(delta): `+6.530899554491043e-08` (no meaningful gain)
- FVD: N/A (image stage)
- CLIP Score: N/A (not computed yet)

## Qualitative Check
- Prompt set: default 5 prompts in `scripts/generate_prompt_comparison.py`
- Output evidence:
  - `artifacts/compare_lora_r4_s200/`
  - `artifacts/compare_lora_r8_s200/`
  - `artifacts/compare_lora_r16_s200/`
  - `artifacts/compare_lora_r8_s400/`
- Updated observation:
  - rank=8 remains the best setting in current small-sample regime.
  - Extending run from s200 to true r8-s400 (actual 400 steps) still does not show clear improvement in best loss.
  - Visual quality should be judged by side-by-side comparison first; current metric change is negligible.

## Issues and Fixes
- Issue: compare script failed in offline mode with LoRA loading error.
- Root cause: `load_lora_weights` did not specify local `weight_name` under `HF_HUB_OFFLINE=1`.
- Fix: updated `scripts/generate_prompt_comparison.py` to pass `weight_name="pytorch_lora_weights.safetensors"` (and exposed as CLI arg).
- Issue: requested `max_train_steps=400` but training stopped at 240.
- Root cause: trainer caps steps by `num_train_epochs * steps_per_epoch`; with 80 train samples and current setup, 3 epochs only provide 240 steps.
- Fix: reran with `num_train_epochs=5`; now confirmed true 400-step completion.

## Next Actions (Top 3)
1. Decide whether to run `rank=8, step=600` for one more long-schedule check, or freeze current image-stage config now.
2. Add quantitative image metrics (CLIP similarity or aesthetic proxy) for objective comparison.
3. If r8 true-400 still has no gain, freeze image-stage config and move to video-stage data/pipeline preparation.
