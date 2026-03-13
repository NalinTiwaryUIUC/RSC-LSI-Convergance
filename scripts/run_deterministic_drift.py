#!/usr/bin/env python3
"""
Deterministic posterior drift test for ULA chains.

Runs a single chain with noise_scale=0.0 so that the update is:
    theta_{k+1} = theta_k - h * grad U(theta_k)

Designed for small_resnet_ln, m=64 (width=1), n_train=512, alpha=0.3, h=5e-8,
T=20000, B=0, S=1, log_every=100, starting from a shared pretrained checkpoint.

Usage (from project root):

  python scripts/run_deterministic_drift.py \
    --pretrain-path experiments/checkpoints/pretrain_small_resnet_ln_w1_n512.pt

This writes a normal run directory under experiments/runs/, with full
iter_metrics.jsonl suitable for the usual diagnostics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, ensure_directories, get_device  # type: ignore
from data import get_probe_loader, get_train_loader  # type: ignore
from run.chain import run_chain  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic posterior drift test (noise_scale=0.0)")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier (default 1.0 for m=64)")
    p.add_argument("--n_train", type=int, default=512, help="Training subset size (default 512)")
    p.add_argument("--alpha", type=float, default=0.3, help="L2 prior strength (default 0.3)")
    p.add_argument("--h", type=float, default=5e-8, help="ULA step size (default 5e-8)")
    p.add_argument("--T", type=int, default=20000, help="Total steps (default 20000)")
    p.add_argument("--B", type=int, default=0, help="Burn-in steps (default 0)")
    p.add_argument("--S", type=int, default=1, help="Save stride (default 1, save every step)")
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Write iter_metrics every N steps (default 100)",
    )
    p.add_argument(
        "--pretrain-path",
        type=str,
        required=True,
        help="Path to pretrained checkpoint for initialization (e.g. m=64 shared checkpoint).",
    )
    p.add_argument(
        "--runs-dir",
        type=str,
        default="experiments/runs",
        help="Parent directory for run outputs (default experiments/runs).",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="experiments/data",
        help="Data/indices directory (default experiments/data).",
    )
    p.add_argument(
        "--root",
        type=str,
        default="./data",
        help="CIFAR-10 download root (default ./data).",
    )
    p.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Dataset seed for subset indices (default 42).",
    )
    p.add_argument(
        "--chain",
        type=int,
        default=0,
        help="Chain id (default 0; used only for naming and seeds).",
    )
    args = p.parse_args()

    ensure_directories()
    device = get_device()
    use_gpu = device.type == "cuda"

    config = RunConfig(
        n_train=args.n_train,
        probe_size=512,
        width_multiplier=args.width,
        arch="small_resnet_ln",
        num_blocks=2,
        h=args.h,
        alpha=args.alpha,
        ce_reduction="sum",
        T=args.T,
        B=args.B,
        S=args.S,
        log_every=args.log_every,
        K=1,
        noise_scale=0.0,  # deterministic drift: no Langevin noise
        pretrain_steps=0,
        pretrain_lr=0.0,
        data_dir=args.data_dir,
        bn_mode="eval",
        bn_calibration_steps=0,
        dataset_seed=args.dataset_seed,
        chain_seed=args.dataset_seed + args.chain * 1000,
        probe_projection_seed=12345,
    )

    train_loader = get_train_loader(
        config.n_train,
        batch_size=config.n_train,
        dataset_seed=config.dataset_seed,
        data_dir=config.data_dir,
        root=args.root,
        pin_memory=use_gpu,
        eval_transform=True,
    )
    probe_loader = get_probe_loader(
        config.probe_size,
        dataset_seed=config.dataset_seed + 1,
        data_dir=config.data_dir,
        root=args.root,
        pin_memory=use_gpu,
    )

    w_str = int(args.width) if args.width == int(args.width) else args.width
    alpha_str = str(args.alpha).replace("-", "m")
    run_name = f"w{w_str}_n{args.n_train}_h{args.h}_a{alpha_str}_det_chain{args.chain}"
    run_dir = Path(args.runs_dir) / run_name

    run_chain(
        config=config,
        chain_id=args.chain,
        run_dir=run_dir,
        train_loader=train_loader,
        probe_loader=probe_loader,
        device=device,
        pretrain_path=Path(args.pretrain_path),
    )


if __name__ == "__main__":
    main()

