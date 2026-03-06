# Experiment Protocol (Week 1)

## Goal
Build a reproducible local loop for:
data preprocessing -> smoke training -> qualitative check.

## Baseline Settings
- Seed: `42`
- Image size: `512`
- Train/Val split: `8:2`
- Small-sample scale: `50-200` samples

## Runbook
1. Preprocess
```powershell
python scripts/preprocess.py --input-dir data/raw --output-dir data/processed --size 512 --val-ratio 0.2
```

2. Smoke training
```powershell
python scripts/train_smoke_test.py --steps 50 --batch-size 8 --out-dir artifacts/smoke
```

3. Save evidence
- `data/processed/stats.json`
- `artifacts/smoke/smoke_result.json`
- first generated samples from your real trainer

## Rules
- Change only one variable each run.
- Log every run in `reports/week1_summary.md`.
- Keep failed runs; mark root cause and fix.
