"""
Run a single ULA chain (for the full experiment or job arrays).
Usage:
  python scripts/run_single_chain.py --width 1 --h 1e-5 --chain 0 --n_train 1024
  python scripts/run_single_chain.py --width 1 --h 1e-5 --chain 0  # uses plan defaults
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, ensure_directories, get_device
from data import get_probe_loader, get_train_loader
from run.chain import run_chain


def main() -> None:
    p = argparse.ArgumentParser(description="Run one ULA chain")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier w (0.5, 1, 2, 4)")
    p.add_argument("--h", type=float, default=1e-5, help="Step size")
    p.add_argument("--chain", type=int, default=0, help="Chain id (0 .. K-1)")
    p.add_argument("--n_train", type=int, default=1024, help="Training subset size (512, 1024, 2048)")
    p.add_argument("--probe_size", type=int, default=512, help="Probe set size")
    p.add_argument("--T", type=int, default=200_000, help="Total steps")
    p.add_argument("--B", type=int, default=50_000, help="Burn-in steps")
    p.add_argument("--S", type=int, default=200, help="Save stride")
    p.add_argument("--pretrain-steps", type=int, default=0, help="Full-batch SGD steps before ULA")
    p.add_argument("--pretrain-lr", type=float, default=0.1, help="Learning rate for pretraining")
    p.add_argument("--data_dir", type=str, default="experiments/data", help="Indices and projections")
    p.add_argument("--runs_dir", type=str, default="experiments/runs", help="Parent dir for run dirs")
    p.add_argument("--root", type=str, default="./data", help="CIFAR-10 download root")
    p.add_argument("--seed", type=int, default=42, help="Dataset seed")
    p.add_argument("--device", type=str, default=None, help="Device: cuda, cuda:0, cpu, or empty for auto (cuda if available)")
    args = p.parse_args()

    ensure_directories()
    if args.device is not None and args.device != "":
        import torch
        device = torch.device(args.device)
    else:
        device = get_device()
    use_gpu = device.type == "cuda"
    config = RunConfig(
        n_train=args.n_train,
        probe_size=args.probe_size,
        width_multiplier=args.width,
        h=args.h,
        alpha=1e-2,
        T=args.T,
        B=args.B,
        S=args.S,
        K=4,
         pretrain_steps=args.pretrain_steps,
         pretrain_lr=args.pretrain_lr,
        chain_seed=args.seed + args.chain * 1000,
        dataset_seed=args.seed,
        data_dir=args.data_dir,
    )
    w_str = int(args.width) if args.width == int(args.width) else args.width
    run_name = f"w{w_str}_n{args.n_train}_h{args.h}_chain{args.chain}"
    run_dir = Path(args.runs_dir) / run_name

    train_loader = get_train_loader(
        config.n_train,
        batch_size=config.n_train,
        dataset_seed=config.dataset_seed,
        data_dir=config.data_dir,
        root=args.root,
        pin_memory=use_gpu,
    )
    probe_loader = get_probe_loader(
        config.probe_size,
        dataset_seed=config.dataset_seed + 1,
        data_dir=config.data_dir,
        root=args.root,
        pin_memory=use_gpu,
    )
    run_chain(
        config, chain_id=args.chain, run_dir=run_dir,
        train_loader=train_loader, probe_loader=probe_loader, device=device,
    )
    print("Done:", run_dir)


if __name__ == "__main__":
    main()
