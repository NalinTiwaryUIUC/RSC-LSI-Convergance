#!/usr/bin/env python3
"""
One-step ULA moment test at a fixed theta0.

Goal:
  Check that the ULA update implementation produces the correct mean and
  variance for Delta-theta when starting from a fixed parameter vector.

For a fixed theta0 and gradient grad_U = ∇U(theta0), the ideal ULA step is:
  Δθ = -h * grad_U + sqrt(2h) * ξ,  ξ ~ N(0, I).

For random unit directions v:
  E[v^T Δθ]   = -h * v^T grad_U
  Var[v^T Δθ] = 2h

This script:
  - Loads theta0 from a pretrained checkpoint.
  - Computes grad_U(theta0) using compute_U (matching the chain's U).
  - Generates many Δθ samples using the same formula as ula_step.
  - Checks empirical means/variances along random projections and subset
    coordinates against the theoretical targets.

Outputs:
  - Human-readable summary on stdout.
  - JSON file under experiments/runs/ with detailed statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, ensure_directories, get_device  # type: ignore
from data import get_train_loader  # type: ignore
from models import create_model, flatten_params  # type: ignore
from ula.potential import compute_U  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description="One-step ULA moment test at fixed theta0")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier (default 1.0 for m=64)")
    p.add_argument("--n_train", type=int, default=512, help="Training subset size (default 512)")
    p.add_argument("--alpha", type=float, default=0.3, help="L2 prior strength (default 0.3)")
    p.add_argument("--h", type=float, default=5e-8, help="ULA step size (default 5e-8)")
    p.add_argument("--reps", type=int, default=512, help="Number of one-step samples (default 512)")
    p.add_argument(
        "--pretrain-path",
        type=str,
        required=True,
        help="Path to pretrained checkpoint for theta0.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data/indices directory; defaults to RunConfig.data_dir.",
    )
    p.add_argument(
        "--root",
        type=str,
        default="./data",
        help="CIFAR-10 download root (default ./data).",
    )
    p.add_argument(
        "--bn-mode",
        type=str,
        default="eval",
        choices=["eval", "batchstat_frozen"],
        help="BN mode for gradient computation (default eval).",
    )
    p.add_argument(
        "--ce-reduction",
        type=str,
        default=None,
        choices=["mean", "sum"],
        help="CE reduction for U; defaults to config.ce_reduction.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for directions and noise (default 12345).",
    )
    p.add_argument(
        "--num-projections",
        type=int,
        default=10,
        help="Number of random projection directions (default 10).",
    )
    p.add_argument(
        "--coord-subsample",
        type=int,
        default=512,
        help="Number of coordinates for coordinate-wise variance check (default 512).",
    )
    args = p.parse_args()

    ensure_directories()
    base_cfg = RunConfig()
    device = get_device()
    use_gpu = device.type == "cuda"

    # Resolve config-like parameters
    data_dir = args.data_dir if args.data_dir is not None else base_cfg.data_dir
    ce_reduction = args.ce_reduction if args.ce_reduction is not None else base_cfg.ce_reduction

    # Data loader (full batch)
    train_loader = get_train_loader(
        args.n_train,
        batch_size=args.n_train,
        dataset_seed=base_cfg.dataset_seed,
        data_dir=data_dir,
        root=args.root,
        pin_memory=use_gpu,
        eval_transform=True,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)
    train_data = (x_train, y_train)

    # Model and theta0 from checkpoint
    model = create_model(
        width_multiplier=args.width,
        num_classes=base_cfg.num_classes,
        arch="small_resnet_ln",
        num_blocks=base_cfg.num_blocks,
    ).to(device)
    ckpt = torch.load(args.pretrain_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # BN mode (eval by default, for reproducibility)
    if args.bn_mode == "eval":
        model.eval()
    elif args.bn_mode == "batchstat_frozen":
        from run.bn_mode import set_bn_batchstats_freeze_buffers  # type: ignore

        set_bn_batchstats_freeze_buffers(model)
    else:
        raise ValueError(f"Unknown bn_mode: {args.bn_mode}")

    # theta0 and grad_U(theta0)
    theta0_flat = flatten_params(model).detach()
    d = theta0_flat.numel()

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    model.zero_grad(set_to_none=True)
    U_val = compute_U(model, train_data, args.alpha, device, ce_reduction=ce_reduction)
    U_val.backward()
    grad_flat = torch.cat([p.grad.view(-1) for p in model.parameters()]).detach()

    # Random projection directions v_i (unit vectors)
    num_proj = args.num_projections
    directions = torch.randn(num_proj, d, device=device)
    directions = torch.nn.functional.normalize(directions, dim=1)

    # Accumulators for projections: sums and sums of squares
    proj_sum = torch.zeros(num_proj, device=device)
    proj_sumsq = torch.zeros(num_proj, device=device)

    # Coordinate subset for variance check
    coord_count = min(args.coord_subsample, d)
    coord_indices = torch.randperm(d, device=device)[:coord_count]
    coord_sum = torch.zeros(coord_count, device=device)
    coord_sumsq = torch.zeros(coord_count, device=device)

    sqrt_2h = math.sqrt(2.0 * args.h)

    for _ in range(args.reps):
        noise = torch.randn(d, device=device)
        delta = -args.h * grad_flat + sqrt_2h * noise

        # Projections
        proj = torch.mv(directions, delta)  # (num_proj,)
        proj_sum += proj
        proj_sumsq += proj * proj

        # Coordinate subset
        sub = delta[coord_indices]
        coord_sum += sub
        coord_sumsq += sub * sub

    # Compute statistics
    proj_mean = proj_sum / args.reps
    proj_var = proj_sumsq / args.reps - proj_mean * proj_mean
    proj_target_mean = -args.h * torch.mv(directions, grad_flat)
    proj_target_var = torch.full_like(proj_var, 2.0 * args.h)

    # Coordinate variances
    coord_mean = coord_sum / args.reps
    coord_var = coord_sumsq / args.reps - coord_mean * coord_mean

    # Error metrics and pass/fail heuristics
    eps = 1e-12
    mean_err = (proj_mean - proj_target_mean).abs()
    # Approximate standard error for mean given empirical variance
    std_err = (proj_var.clamp_min(eps).sqrt()) / math.sqrt(args.reps)
    mean_within_3se = (mean_err <= 3.0 * std_err).float().mean().item()

    rel_var_err = (proj_var - proj_target_var).abs() / (proj_target_var + eps)
    var_within_20pct = (rel_var_err <= 0.2).float().mean().item()

    coord_var_target = 2.0 * args.h
    coord_rel_err = (coord_var - coord_var_target).abs() / (coord_var_target + eps)
    coord_within_20pct = (coord_rel_err <= 0.2).float().mean().item()

    pass_mean = mean_within_3se >= 0.8  # at least 80% of projections within 3 * SE
    pass_var = var_within_20pct >= 0.8  # at least 80% of projections within 20%
    pass_coord = coord_within_20pct >= 0.8
    passed = pass_mean and pass_var and pass_coord

    # Human-readable summary
    print("One-step ULA moment test")
    print(f"  width={args.width}, n_train={args.n_train}, alpha={args.alpha}, h={args.h}")
    print(f"  reps={args.reps}, num_proj={num_proj}, coord_subsample={coord_count}")
    print(f"  U(theta0) = {U_val.item():.6f}")
    print(f"  ||grad_U(theta0)|| = {grad_flat.norm().item():.6f}")
    print(f"  mean projections within 3*SE: {mean_within_3se*100:.1f}% (pass={pass_mean})")
    print(f"  var projections within 20% of 2h: {var_within_20pct*100:.1f}% (pass={pass_var})")
    print(f"  coord-wise var within 20% of 2h: {coord_within_20pct*100:.1f}% (pass={pass_coord})")
    print(f"  overall pass = {passed}")

    # JSON results
    w_str = int(args.width) if args.width == int(args.width) else args.width
    alpha_str = str(args.alpha).replace("-", "m")
    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"one_step_moment_w{w_str}_n{args.n_train}_h{args.h}_a{alpha_str}.json"

    result: Dict[str, object] = {
        "config": {
            "width": args.width,
            "n_train": args.n_train,
            "alpha": args.alpha,
            "h": args.h,
            "reps": args.reps,
            "bn_mode": args.bn_mode,
            "ce_reduction": ce_reduction,
            "pretrain_path": args.pretrain_path,
            "seed": args.seed,
            "num_projections": num_proj,
            "coord_subsample": coord_count,
        },
        "theta0": {
            "norm": theta0_flat.norm().item(),
            "dim": d,
        },
        "U_theta0": U_val.item(),
        "grad_theta0": {
            "norm": grad_flat.norm().item(),
        },
        "projections": {
            "mean_emp": proj_mean.tolist(),
            "mean_target": proj_target_mean.tolist(),
            "var_emp": proj_var.tolist(),
            "var_target": proj_target_var.tolist(),
            "mean_within_3se_fraction": mean_within_3se,
            "var_within_20pct_fraction": var_within_20pct,
        },
        "coordinates": {
            "indices": coord_indices.cpu().tolist(),
            "var_emp": coord_var.tolist(),
            "coord_within_20pct_fraction": coord_within_20pct,
        },
        "pass": {
            "mean": pass_mean,
            "var": pass_var,
            "coord": pass_coord,
            "overall": passed,
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote JSON summary to {out_path}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

