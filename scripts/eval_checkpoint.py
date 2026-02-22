"""
One-off eval: load a pretrained checkpoint and compute mean CE + accuracy.
Uses pure eval mode (model.eval(), dropout off, BN running stats) and
TEST_TRANSFORM (no augmentations) for deterministic, comparable metrics.

Use this to sanity-check: if pretrain reported CE ~1.x but Run 1 shows CE ~4,
there may be a pipeline mismatch (BN mode, augmentations, data split, etc.).

Usage:
  python scripts/eval_checkpoint.py experiments/checkpoints/pretrain_w0.1_n1024.pt
  python scripts/eval_checkpoint.py path/to/ckpt.pt --n_train 1024 --dataset_seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cifar import TEST_TRANSFORM
from data.indices import get_train_subset_indices, get_probe_indices
from models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10


def main() -> None:
    p = argparse.ArgumentParser(description="Eval checkpoint: mean CE + accuracy (pure eval, no aug)")
    p.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--probe_size", type=int, default=512)
    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    width = ckpt.get("width", 1.0)
    model = create_model(width_multiplier=width).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Train subset with TEST_TRANSFORM (no augmentation) for deterministic eval
    train_indices = get_train_subset_indices(args.n_train, args.dataset_seed, args.data_dir)
    train_ds = CIFAR10(root=args.root, train=True, download=True, transform=TEST_TRANSFORM)
    train_subset = Subset(train_ds, train_indices)
    train_loader = DataLoader(train_subset, batch_size=len(train_indices), shuffle=False)

    # Probe subset (test split)
    probe_indices = get_probe_indices(args.probe_size, args.dataset_seed + 1, args.data_dir)
    probe_ds = CIFAR10(root=args.root, train=False, download=True, transform=TEST_TRANSFORM)
    probe_subset = Subset(probe_ds, probe_indices)
    probe_loader = DataLoader(probe_subset, batch_size=len(probe_indices), shuffle=False)

    with torch.no_grad():
        # Train subset
        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        logits = model(x_train)
        ce_train = F.cross_entropy(logits, y_train, reduction="mean").item()
        acc_train = (logits.argmax(dim=1) == y_train).float().mean().item() * 100

        # Probe subset (test split)
        x_probe, y_probe = next(iter(probe_loader))
        x_probe, y_probe = x_probe.to(device), y_probe.to(device)
        logits_probe = model(x_probe)
        ce_probe = F.cross_entropy(logits_probe, y_probe, reduction="mean").item()
        acc_probe = (logits_probe.argmax(dim=1) == y_probe).float().mean().item() * 100

    print("=" * 50)
    print("Checkpoint eval (model.eval(), TEST_TRANSFORM, no aug)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Width: {width}")
    print(f"  Train (n={args.n_train}): mean CE = {ce_train:.4f}, accuracy = {acc_train:.2f}%")
    print(f"  Probe (n={args.probe_size}, test split): mean CE = {ce_probe:.4f}, accuracy = {acc_probe:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
