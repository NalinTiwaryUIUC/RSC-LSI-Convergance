"""
Systematic ULA diagnostic: decompose gradient, compute drift vs restorative balance,
and recommend alpha/noise_scale to achieve equilibrium.

Usage:
  python scripts/diagnose_ula.py
  python scripts/diagnose_ula.py --n_train 1024 --width 1 --alpha 0.01 --noise-scale 1.0
  python scripts/diagnose_ula.py --pretrain-path experiments/checkpoints/pretrain_w1_n1024.pt

Outputs:
  - Gradient decomposition (NLL vs prior)
  - SNR, drift coefficient, restorative coefficient
  - Balance ratio (restorative/drift): <1 => drift, >1 => shrinking
  - Recommended noise_scale and alpha for balance
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import RunConfig, ensure_directories, get_device
from data import get_probe_loader, get_train_loader
from models import create_model, flatten_params, unflatten_like
from ula.potential import compute_U


def main() -> None:
    p = argparse.ArgumentParser(description="ULA drift diagnostic")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=None, help="Use config default if not set")
    p.add_argument("--noise-scale", type=float, default=None, help="Use config default if not set")
    p.add_argument("--pretrain-steps", type=int, default=2000)
    p.add_argument("--pretrain-path", type=str, default=None, help="Path to pretrained checkpoint; if set, skips pretrain")
    args = p.parse_args()

    ensure_directories()
    device = get_device()
    use_gpu = device.type == "cuda"
    config = RunConfig(
        n_train=args.n_train,
        probe_size=512,
        width_multiplier=args.width,
        pretrain_steps=args.pretrain_steps,
    )
    alpha = args.alpha if args.alpha is not None else config.alpha
    noise_scale = args.noise_scale if args.noise_scale is not None else config.noise_scale
    h = config.h

    train_loader = get_train_loader(
        config.n_train,
        batch_size=config.n_train,
        dataset_seed=config.dataset_seed,
        data_dir=config.data_dir,
        root="./data",
        pin_memory=use_gpu,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)
    train_data = (x_train, y_train)

    model = create_model(width_multiplier=config.width_multiplier).to(device)
    d = flatten_params(model).numel()

    if args.pretrain_path:
        ckpt = torch.load(args.pretrain_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    elif config.pretrain_steps > 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.pretrain_lr, momentum=0.9)
        model.train()
        for _ in range(config.pretrain_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train)
            loss = torch.nn.functional.cross_entropy(logits, y_train, reduction="mean")
            loss.backward()
            optimizer.step()

    theta = flatten_params(model).clone().detach()
    theta_norm = theta.norm().item()
    theta_sq = (theta_norm ** 2)

    # Full gradient: ∇U = ∇NLL + αθ
    model.zero_grad(set_to_none=True)
    U = compute_U(model, train_data, alpha, device)
    U.backward()
    grad_U = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_norm = grad_U.norm().item()

    prior_grad = alpha * theta  # ∇(α/2 ||θ||²) = αθ
    grad_NLL = grad_U - prior_grad
    grad_nll_norm = grad_NLL.norm().item()
    prior_grad_norm = prior_grad.norm().item()

    # SNR
    signal = h * grad_norm
    noise_std_full = math.sqrt(2.0 * h * d) * noise_scale
    snr = signal / noise_std_full if noise_std_full > 0 else float("nan")

    # Drift: E[||noise_step||²] per step. Each component gets N(0, 2h*noise_scale²).
    drift_coef = d * 2.0 * h * (noise_scale ** 2)

    # Restorative: prior gradient -αθ pulls toward origin. Change in ||θ||² from gradient alone:
    # Δθ = -h*∇U. For prior part: Δθ_prior = -h*αθ. Δ||θ||² ≈ 2θ·Δθ = -2hα||θ||².
    restorative_coef = 2.0 * h * alpha * theta_sq

    # Balance ratio: restorative / drift. >1 => prior wins (θ shrinks). <1 => drift wins (θ grows).
    balance_ratio = restorative_coef / drift_coef if drift_coef > 0 else float("inf")

    # Recommended values for balance (restorative = drift)
    noise_scale_balance = math.sqrt(alpha * theta_sq / d) if d > 0 else 0.0
    alpha_balance = d * (noise_scale ** 2) / theta_sq if theta_sq > 0 else 0.0

    print("=" * 60)
    print("ULA DIAGNOSTIC (post-pretrain)")
    print("=" * 60)
    print(f"  d (params)     = {d:,}")
    print(f"  h              = {h}")
    print(f"  alpha          = {alpha}")
    print(f"  noise_scale    = {noise_scale}")
    print(f"  theta_norm     = {theta_norm:.4f}")
    print()
    print("--- Gradient decomposition ---")
    print(f"  ||∇U||         = {grad_norm:.6f}")
    print(f"  ||∇NLL||       = {grad_nll_norm:.6f}  (data term)")
    print(f"  ||αθ||         = {prior_grad_norm:.6f}  (prior term)")
    print(f"  prior/||∇U||   = {prior_grad_norm/grad_norm:.4f}  (prior fraction)")
    print()
    print("--- Drift vs restorative ---")
    print(f"  SNR            = {snr:.2e}  (signal/noise; <1e-3 => random walk)")
    print(f"  drift_coef     = {drift_coef:.2e}  (E[||noise||²] per step)")
    print(f"  restorative    = {restorative_coef:.2e}  (prior pull per step)")
    print(f"  balance_ratio  = {balance_ratio:.4f}  (<1 => drift, >1 => shrinking)")
    print()
    print("--- Recommendations for equilibrium (balance_ratio ≈ 1) ---")
    print(f"  noise_scale    = {noise_scale_balance:.6f}  (current: {noise_scale})")
    print(f"  alpha          = {alpha_balance:.4f}  (current: {alpha})")
    print()
    if balance_ratio < 0.5:
        print("  => Drift dominates. Increase alpha or decrease noise_scale.")
    elif balance_ratio > 2.0:
        print("  => Restorative dominates. May get stuck. Decrease alpha or increase noise_scale.")
    else:
        print("  => Roughly balanced. Monitor U/theta over run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
