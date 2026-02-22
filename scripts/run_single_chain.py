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

_DEFAULTS = RunConfig()


def main() -> None:
    p = argparse.ArgumentParser(description="Run one ULA chain")
    p.add_argument("--width", type=float, default=_DEFAULTS.width_multiplier, help="Width multiplier w (0.5, 1, 2, 4)")
    p.add_argument("--h", type=float, default=_DEFAULTS.h, help="Step size")
    p.add_argument("--chain", type=int, default=0, help="Chain id (0 .. K-1)")
    p.add_argument("--n_train", type=int, default=_DEFAULTS.n_train, help="Training subset size")
    p.add_argument("--probe_size", type=int, default=_DEFAULTS.probe_size, help="Probe set size")
    p.add_argument("--T", type=int, default=_DEFAULTS.T, help="Total steps")
    p.add_argument("--B", type=int, default=_DEFAULTS.B, help="Burn-in steps")
    p.add_argument("--S", type=int, default=_DEFAULTS.S, help="Save stride")
    p.add_argument("--log-every", type=int, default=_DEFAULTS.log_every, help="Write iter_metrics every N steps (default 1000)")
    p.add_argument("--pretrain-steps", type=int, default=_DEFAULTS.pretrain_steps, help="Full-batch SGD steps before ULA (ignored if --pretrain-path)")
    p.add_argument("--pretrain-lr", type=float, default=_DEFAULTS.pretrain_lr, help="Learning rate for pretraining")
    p.add_argument("--pretrain-path", type=str, default=None, help="Path to pretrained checkpoint; if set, skips per-chain pretrain")
    p.add_argument("--bn-mode", type=str, default=_DEFAULTS.bn_mode, choices=["eval", "batchstat_frozen"],
                   help="BN mode for ULA sampling: eval=frozen running stats (partition-invariant), batchstat_frozen=batch stats+frozen buffers")
    p.add_argument("--bn-calibration-steps", type=int, default=_DEFAULTS.bn_calibration_steps,
                   help="When bn_mode=eval: N forward passes (train, no grad) over subset to populate BN running stats before sampling. 0=skip")
    p.add_argument("--data_dir", type=str, default=_DEFAULTS.data_dir, help="Indices and projections")
    p.add_argument("--runs_dir", type=str, default="experiments/runs", help="Parent dir for run dirs")
    p.add_argument("--root", type=str, default="./data", help="CIFAR-10 download root")
    p.add_argument("--seed", type=int, default=_DEFAULTS.dataset_seed, help="Dataset seed")
    p.add_argument("--device", type=str, default=None, help="Device: cuda, cuda:0, cpu, or empty for auto")
    p.add_argument("--noise-scale", type=float, default=_DEFAULTS.noise_scale, help="Langevin noise scale (default 1.0; <1 = less diffusion, >1 = more)")
    p.add_argument("--alpha", type=float, default=_DEFAULTS.alpha, help="L2 prior strength (higher = stronger pull, less drift)")
    p.add_argument("--ce-reduction", type=str, default=_DEFAULTS.ce_reduction, choices=["mean", "sum"],
                   help="CE reduction in U: mean (stable at larger h) or sum")
    p.add_argument("--clip-grad-norm", type=float, default=None,
                   help="S3: Clip grad norm to this value; logs grad_norm_pre_clip, grad_norm_post_clip. Omit for no clipping.")
    p.add_argument("--microbatch-size", type=int, default=None,
                   help="Per-step batch size for gradient accumulation; effective_batch=n_train. Omit for full-batch.")
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
        alpha=args.alpha,
        ce_reduction=args.ce_reduction,
        T=args.T,
        B=args.B,
        S=args.S,
        log_every=args.log_every,
        K=_DEFAULTS.K,
        noise_scale=args.noise_scale,
        pretrain_steps=args.pretrain_steps,
        pretrain_lr=args.pretrain_lr,
        chain_seed=args.seed + args.chain * 1000,
        dataset_seed=args.seed,
        data_dir=args.data_dir,
        bn_mode=args.bn_mode,
        bn_calibration_steps=args.bn_calibration_steps,
        clip_grad_norm=args.clip_grad_norm,
    )
    if args.microbatch_size is not None:
        if args.microbatch_size <= 0 or args.n_train % args.microbatch_size != 0:
            raise ValueError("--microbatch-size must divide --n_train")
        config.microbatch_size = args.microbatch_size
        config.num_microbatches = args.n_train // args.microbatch_size
        config.effective_batch_size = args.n_train
    w_str = int(args.width) if args.width == int(args.width) else args.width
    alpha_str = str(args.alpha).replace("-", "m")  # 1e-5 -> 1em5 for filenames
    run_name = f"w{w_str}_n{args.n_train}_h{args.h}_a{alpha_str}_chain{args.chain}"
    run_dir = Path(args.runs_dir) / run_name

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
    pretrain_path = Path(args.pretrain_path) if args.pretrain_path else None
    run_chain(
        config, chain_id=args.chain, run_dir=run_dir,
        train_loader=train_loader, probe_loader=probe_loader, device=device,
        pretrain_path=pretrain_path,
    )
    print("Done:", run_dir)


if __name__ == "__main__":
    main()
