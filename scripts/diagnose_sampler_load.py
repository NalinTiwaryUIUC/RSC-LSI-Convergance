#!/usr/bin/env python3
"""
Runs 2–4: Sampler load diagnostics (T=1, h=0, noise=0).
Load checkpoint, evaluate on train data, print ce_mean and acc only.

Run 2: model.eval(), same batch as Run 1 (from .batch.pt)
Run 3: model.train(), same batch as Run 1 (mode sensitivity)
Run 4: model.eval(), new batch with TRAIN_TRANSFORM (transform sensitivity)

Usage:
  python scripts/diagnose_sampler_load.py --run 2 --checkpoint experiments/checkpoints/pretrain_w0.1_n1024.pt
  python scripts/diagnose_sampler_load.py --run 3 --checkpoint experiments/checkpoints/pretrain_w0.1_n1024.pt
  python scripts/diagnose_sampler_load.py --run 4 --checkpoint experiments/checkpoints/pretrain_w0.1_n1024.pt

Requires: Run pretrain with --verify first to create the .batch.pt file (for runs 2 and 3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import get_train_loader
from models import create_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, required=True, choices=[2, 3, 4], help="2=eval+same batch, 3=train+same batch, 4=eval+TRAIN_TRANSFORM")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    width = ckpt.get("width", 1.0)
    model = create_model(width_multiplier=width).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    if args.run in (2, 3):
        batch_path = ckpt_path.with_suffix(".batch.pt")
        if not batch_path.exists():
            print(f"ERROR: {batch_path} not found. Run pretrain with --verify first.")
            return 1
        batch = torch.load(batch_path, map_location=device, weights_only=True)
        x, y = batch["x"].to(device), batch["y"].to(device)
    else:
        # Run 4: same loader as pretrain/chain (eval_transform=True, TEST_TRANSFORM)
        loader = get_train_loader(
            args.n_train,
            batch_size=args.n_train,
            dataset_seed=args.dataset_seed,
            data_dir=args.data_dir,
            root=args.root,
            eval_transform=True,
        )
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)

    if args.run == 3:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        logits = model(x)
        ce_mean = F.cross_entropy(logits, y, reduction="mean").item()
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100

    print(f"Run {args.run}: ce_mean = {ce_mean:.6f}, acc = {acc:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
