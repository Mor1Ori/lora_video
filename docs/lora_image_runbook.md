# Image LoRA Runbook

All commands below use the `lora_video` conda env.

## 1) Train (short smoke run)

```powershell
conda --no-plugins run -n lora_video python scripts/train_lora_image.py `
  --train-data-dir data/processed `
  --metadata-file data/processed/metadata_train.jsonl `
  --output-dir artifacts/lora_image_ghibli `
  --resolution 512 `
  --train-batch-size 1 `
  --num-train-epochs 1 `
  --max-train-steps 50 `
  --save-steps 25 `
  --learning-rate 1e-4 `
  --rank 8 `
  --mixed-precision fp16
```

## 2) Inference

```powershell
conda --no-plugins run -n lora_video python scripts/infer_lora_image.py `
  --lora-path artifacts/lora_image_ghibli `
  --output-dir artifacts/infer_lora_ghibli `
  --prompt "a cozy countryside street in hand-drawn animation style, warm light, cinematic composition" `
  --num-images 4 `
  --steps 30
```

## 3) Suggested first ablation

- Keep everything fixed, change only `--rank` to `4 / 8 / 16`.
- Compare style strength, detail stability, and training speed.
