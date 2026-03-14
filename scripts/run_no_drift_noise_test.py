#!/usr/bin/env python3
"""
No-drift, noise-only test for ULA chains.

Runs a chain with drift_scale=0 and noise_scale=1.0 so that the update is:
    theta_{k+1} = theta_k + sqrt(2*h) * noise_scale * Z_k   (pure diffusion from init)

Same setup as the stochastic run: m=64 (width=1), n_train=512, h=5e-8, T=20000,
B=0, S=1, log_every=100, from shared pretrained checkpoint. Output goes to
experiments/runs/ under a distinct name (drift0_noise1_...) so it is easy to
tell apart from normal and deterministic-drift runs.

Usage (from project root):

  python scripts/run_no_drift_noise_test.py \
    --pretrain-path experiments/checkpoints/pretrain_small_resnet_ln_w1_n512.pt

Optional: --chains 2 for two chains (default 1).
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
    p = argparse.ArgumentParser(
        description="No-drift noise-only test (drift_scale=0, noise_scale=1.0)"
    )
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier (default 1.0 for m=64)")
    p.add_argument("--n_train", type=int, default=512, help="Training subset size (default 512)")
    p.add_argument("--alpha", type=float, default=0.3, help="L2 prior strength (default 0.3)")
    p.add_argument("--h", type=float, default=5e-8, help="ULA step size (default 5e-8)")
    p.add_argument("--T", type=int, default=20000, help="Total steps (default 20000)")
    p.add_argument("--B", type=int, default=0, help="Burn-in steps (default 0)")
    p.add_argument("--S", type=int, default=1, help="Save stride (default 1)")
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
        help="Path to pretrained checkpoint (e.g. m=64 shared checkpoint).",
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
        "--chains",
        type=int,
        default=1,
        help="Number of chains to run (default 1; use 2 if affordable).",
    )
    args = p.parse_args()

    ensure_directories()
    device = get_device()
    use_gpu = device.type == "cuda"

    train_loader = get_train_loader(
        args.n_train,
        batch_size=args.n_train,
        dataset_seed=args.dataset_seed,
        data_dir=args.data_dir,
        root=args.root,
        pin_memory=use_gpu,
        eval_transform=True,
    )
    probe_loader = get_probe_loader(
        512,
        dataset_seed=args.dataset_seed + 1,
        data_dir=args.data_dir,
        root=args.root,
        pin_memory=use_gpu,
    )

    w_str = int(args.width) if args.width == int(args.width) else args.width
    alpha_str = str(args.alpha).replace("-", "m")
    base_name = f"drift0_noise1_w{w_str}_n{args.n_train}_h{args.h}_a{alpha_str}"

    for chain_id in range(args.chains):
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
            K=args.chains,
            noise_scale=1.0,
            drift_scale=0.0,
            pretrain_steps=0,
            pretrain_lr=0.0,
            data_dir=args.data_dir,
            bn_mode="eval",
            bn_calibration_steps=0,
            dataset_seed=args.dataset_seed,
            chain_seed=args.dataset_seed + chain_id * 1000,
            probe_projection_seed=12345,
        )

        run_name = f"{base_name}_chain{chain_id}"
        run_dir = Path(args.runs_dir) / run_name

        run_chain(
            config=config,
            chain_id=chain_id,
            run_dir=run_dir,
            train_loader=train_loader,
            probe_loader=probe_loader,
            device=device,
            pretrain_path=Path(args.pretrain_path),
        )
        print("Done:", run_dir)


if __name__ == "__main__":
    main()
