#!/usr/bin/env python3
"""
Training smoke test:
- If PyTorch exists, run a tiny CPU training loop to verify environment.
- If not, print a clear checklist for full LoRA training command wiring.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run_torch_smoke(steps: int, batch_size: int, out_dir: Path) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    start = time.time()
    losses = []
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, 64)
        y = x.sum(dim=1, keepdim=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == steps:
            print(f"step={step} loss={loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "torch_smoke",
        "steps": steps,
        "batch_size": batch_size,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "duration_sec": round(time.time() - start, 3),
    }
    with open(out_dir / "smoke_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local training smoke test.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()

    try:
        result = run_torch_smoke(args.steps, args.batch_size, args.out_dir)
        print("Smoke training succeeded.")
        print(json.dumps(result, indent=2))
    except ImportError:
        print("PyTorch not found. Install training deps first, then run your LoRA command.")
        print("Suggested next step:")
        print(
            "  python train.py --config configs/baseline.yaml "
            "--train-data data/processed/train --val-data data/processed/val"
        )


if __name__ == "__main__":
    main()
