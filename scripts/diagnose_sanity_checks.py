#!/usr/bin/env python3
"""
Quick sanity checks for ULA setup.
Runs 6 tests: BN partition sensitivity, gradient determinism, noise scaling,
CE sum vs mean, prior correctness, BN buffer frozen.

Usage:
  python scripts/diagnose_sanity_checks.py
  python scripts/diagnose_sanity_checks.py --n_train 512 --width 0.1 --pretrain-path experiments/checkpoints/pretrain_w0.1_n1024.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Lock determinism for diagnostic runs
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, get_device
from data import get_train_loader
from models import create_model, flatten_params, unflatten_like
from run.bn_mode import max_bn_running_delta, set_bn_batchstats_freeze_buffers, snapshot_bn_running_stats
from ula.potential import compute_U


def _compute_U_and_grad(model, x, y, alpha, device, ce_reduction="sum"):
    """Compute U and return (U, grad_flat)."""
    model.zero_grad(set_to_none=True)
    logits = model(x)
    ce = F.cross_entropy(logits, y, reduction=ce_reduction)
    reg = (alpha / 2.0) * sum((p * p).sum() for p in model.parameters())
    U = ce + reg
    U.backward()
    grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return U.item(), grad


def _compute_ce_and_grad(model, x, y, device, ce_reduction):
    """Compute CE and grad of CE only. Returns (ce, grad)."""
    model.zero_grad(set_to_none=True)
    logits = model(x)
    ce = F.cross_entropy(logits, y, reduction=ce_reduction)
    ce.backward()
    grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return ce.item(), grad


def test1_bn_partition_sensitivity(model, x_full, y_full, alpha, device, n_train=1024):
    """One-shot (full batch) vs accumulated (half + half) in eval and batchstat_frozen."""
    half = n_train // 2
    x1, x2 = x_full[:half], x_full[half:]
    y1, y2 = y_full[:half], y_full[half:]
    results = {}

    for bn_mode, name in [("eval", "eval"), ("batchstat_frozen", "batchstat_frozen")]:
        if bn_mode == "eval":
            model.eval()
        else:
            set_bn_batchstats_freeze_buffers(model)

        with torch.no_grad():
            theta = flatten_params(model).clone()

        # One-shot: full batch 1024
        unflatten_like(theta.clone(), model)
        U_oneshot, grad_oneshot = _compute_U_and_grad(model, x_full, y_full, alpha, device, "sum")

        # Accumulated: two passes 512+512, sum CE and sum grads (PyTorch accumulates grads)
        unflatten_like(theta.clone(), model)
        model.zero_grad(set_to_none=True)
        logits1 = model(x1)
        ce1 = F.cross_entropy(logits1, y1, reduction="sum")
        ce1.backward()
        logits2 = model(x2)
        ce2 = F.cross_entropy(logits2, y2, reduction="sum")
        ce2.backward()
        grad_ce_acc = torch.cat([p.grad.view(-1) for p in model.parameters()])
        prior = (alpha / 2.0) * (theta * theta).sum().item()
        grad_prior = alpha * theta
        grad_acc = grad_ce_acc + grad_prior
        U_acc = ce1.item() + ce2.item() + prior

        diff_U = abs(U_oneshot - U_acc)
        diff_grad = (grad_oneshot - grad_acc).abs().max().item()
        diff_grad_rel = diff_grad / (grad_oneshot.norm().item() + 1e-12)

        results[name] = {
            "U_oneshot": U_oneshot,
            "U_acc": U_acc,
            "diff_U": diff_U,
            "diff_grad_max": diff_grad,
            "diff_grad_rel": diff_grad_rel,
            "pass": diff_U < 0.01 and diff_grad_rel < 0.01,
        }

    return results


def test2_gradient_determinism(model, train_data, alpha, device, ce_reduction="sum", n_repeat=2):
    """Compute U and grad twice; should match (deterministic)."""
    with torch.no_grad():
        theta = flatten_params(model).clone()

    us, grads = [], []
    for _ in range(n_repeat):
        unflatten_like(theta.clone(), model)
        u, g = _compute_U_and_grad(model, train_data[0], train_data[1], alpha, device, ce_reduction)
        us.append(u)
        grads.append(g.clone())

    diff_U = max(abs(us[i] - us[0]) for i in range(1, n_repeat))
    diff_grad_max = max((grads[i] - grads[0]).abs().max().item() for i in range(1, n_repeat))
    diff_grad_rel = diff_grad_max / (grads[0].norm().item() + 1e-12)

    return {
        "diff_U": diff_U,
        "diff_grad_max": diff_grad_max,
        "diff_grad_rel": diff_grad_rel,
        "pass": diff_U < 1e-4 and diff_grad_rel < 1e-4,
    }


def test3_noise_scaling(h, d, device, noise_scale=1.0, n_samples=1000, seed=42):
    """With grad=0, sample noise steps; verify var(Δθ_i) ≈ 2h, ||Δθ||^2 ≈ 2hd."""
    noise_std = (2.0 * h) ** 0.5 * noise_scale
    g = torch.Generator(device=device).manual_seed(seed)
    deltas = []
    for _ in range(n_samples):
        delta = noise_std * torch.randn(d, device=device, generator=g)
        deltas.append(delta)
    stack = torch.stack(deltas)

    var_per_coord = stack.var(dim=0).mean().item()
    expected_var = 2.0 * h * (noise_scale**2)
    var_ratio = var_per_coord / expected_var if expected_var > 0 else float("nan")

    norms_sq = (stack**2).sum(dim=1)
    mean_norm_sq = norms_sq.mean().item()
    expected_norm_sq = d * 2.0 * h * (noise_scale**2)
    norm_ratio = mean_norm_sq / expected_norm_sq if expected_norm_sq > 0 else float("nan")

    return {
        "var_per_coord": var_per_coord,
        "expected_var": expected_var,
        "var_ratio": var_ratio,
        "mean_norm_sq": mean_norm_sq,
        "expected_norm_sq": expected_norm_sq,
        "norm_ratio": norm_ratio,
        "pass": 0.9 < var_ratio < 1.1 and 0.9 < norm_ratio < 1.1,
    }


def test4_ce_sum_vs_mean(model, x, y, device, n_train):
    """CE sum vs mean: ce_sum ≈ n*ce_mean, ||grad_sum|| ≈ n*||grad_mean||."""
    with torch.no_grad():
        theta = flatten_params(model).clone()

    unflatten_like(theta.clone(), model)
    ce_sum, grad_sum = _compute_ce_and_grad(model, x, y, device, "sum")
    unflatten_like(theta.clone(), model)
    ce_mean, grad_mean = _compute_ce_and_grad(model, x, y, device, "mean")

    ce_ratio = ce_sum / (n_train * ce_mean) if ce_mean != 0 else float("nan")
    grad_ratio = grad_sum.norm().item() / (n_train * grad_mean.norm().item()) if grad_mean.norm() > 0 else float("nan")

    return {
        "ce_sum": ce_sum,
        "ce_mean": ce_mean,
        "n*ce_mean": n_train * ce_mean,
        "ce_ratio": ce_ratio,
        "||grad_sum||": grad_sum.norm().item(),
        "n*||grad_mean||": n_train * grad_mean.norm().item(),
        "grad_ratio": grad_ratio,
        "pass": 0.99 < ce_ratio < 1.01 and 0.95 < grad_ratio < 1.05,
    }


def test5_prior_correctness(model, train_data, alpha, device, ce_reduction="sum"):
    """Verify U_train ≈ U_prior + U_data, grad decomposition."""
    theta = flatten_params(model)
    theta_norm_sq = (theta * theta).sum().item()
    U_prior = 0.5 * alpha * theta_norm_sq
    grad_prior = alpha * theta

    U_train, grad_U = _compute_U_and_grad(model, train_data[0], train_data[1], alpha, device, ce_reduction)
    U_data = U_train - U_prior

    grad_prior_norm = grad_prior.norm().item()
    model.zero_grad(set_to_none=True)
    logits = model(train_data[0])
    ce = F.cross_entropy(logits, train_data[1], reduction=ce_reduction)
    ce.backward()
    grad_ce = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_recon = grad_ce + grad_prior
    residual = (grad_U - grad_recon).norm().item()

    return {
        "U_train": U_train,
        "U_prior": U_prior,
        "U_data": U_data,
        "U_prior + U_data": U_prior + U_data,
        "residual (U)": abs(U_train - (U_prior + U_data)),
        "||grad_prior||": grad_prior_norm,
        "grad_residual": residual,
        "pass": residual < 1e-5,
    }


def test6_bn_buffer_frozen(model, x, y, alpha, device):
    """In batchstat_frozen, BN running stats must not change after forward/backward."""
    set_bn_batchstats_freeze_buffers(model)
    stats_before = snapshot_bn_running_stats(model)

    model.zero_grad(set_to_none=True)
    U = compute_U(model, (x, y), alpha, device, ce_reduction="sum")
    U.backward()

    delta = max_bn_running_delta(stats_before, model)

    return {
        "max_bn_delta": delta,
        "pass": delta == 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--width", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--h", type=float, default=1e-9)
    p.add_argument("--pretrain-path", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    loader = get_train_loader(
        args.n_train,
        batch_size=args.n_train,
        dataset_seed=args.seed,
        data_dir=args.data_dir,
        eval_transform=True,
    )
    x_full, y_full = next(iter(loader))
    x_full = x_full.to(device)
    y_full = y_full.to(device)
    train_data = (x_full, y_full)

    model = create_model(width_multiplier=args.width).to(device)
    d = flatten_params(model).numel()

    if args.pretrain_path:
        ckpt = torch.load(args.pretrain_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        # Quick pretrain for non-trivial theta
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.02)
        for _ in range(50):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_full), y_full, reduction="mean")
            loss.backward()
            opt.step()

    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    print(f"n_train={args.n_train}, width={args.width}, alpha={args.alpha}, d={d}")
    print()

    # Test 1: BN partition sensitivity
    half = args.n_train // 2
    print("1) BN PARTITION SENSITIVITY")
    print(f"   One-shot: batch_size={args.n_train} (full)")
    print(f"   Accumulated: 2 passes batch_size={half}+{half}")
    print()
    r1 = test1_bn_partition_sensitivity(model, x_full, y_full, args.alpha, device, args.n_train)
    for mode, v in r1.items():
        print(f"   bn_mode={mode}:")
        print(f"      ΔU = |U_one_shot - U_accum| = {v['diff_U']:.6f}")
        print(f"      Δg = max_abs(grad_one_shot - grad_accum) = {v['diff_grad_max']:.6f}")
        print(f"      grad_diff_rel = Δg/||grad|| = {v['diff_grad_rel']:.6f}")
        status = "PASS" if v["pass"] else "FAIL"
        print(f"      -> {status}")
    print("   Decision: ", end="")
    batchstat_large = not r1.get("batchstat_frozen", {}).get("pass", True)
    eval_large = not r1.get("eval", {}).get("pass", True)
    if batchstat_large and not eval_large:
        print(
            "batchstat_frozen shows large ΔU/Δg → must force single-batch (batch_size=n_train) "
            "or switch to bn_mode=eval during sampling."
        )
    elif eval_large:
        print("Both modes show large ΔU/Δg → BN not the culprit; check determinism/noise/prior.")
    else:
        print("Both modes show small ΔU/Δg → BN not the culprit; move to other tests.")
    print()

    # Test 2: Gradient determinism
    print("2) GRADIENT DETERMINISM (same θ, two U/grad evals)")
    for bn_mode in ["eval", "batchstat_frozen"]:
        if bn_mode == "eval":
            model.eval()
        else:
            set_bn_batchstats_freeze_buffers(model)
        r2 = test2_gradient_determinism(model, train_data, args.alpha, device, "sum")
        status = "PASS" if r2["pass"] else "FAIL"
        print(f"   bn_mode={bn_mode}: diff_U={r2['diff_U']:.2e}, diff_grad_rel={r2['diff_grad_rel']:.2e} -> {status}")
    print()

    # Test 3: Noise scaling
    print("3) NOISE SCALING (grad=0, 1000 samples)")
    r3 = test3_noise_scaling(args.h, d, device, noise_scale=1.0, n_samples=1000)
    print(f"   var_per_coord / expected = {r3['var_ratio']:.4f} (expected 1.0)")
    print(f"   mean||Δθ||^2 / (2hd) = {r3['norm_ratio']:.4f} (expected 1.0)")
    print(f"   -> {'PASS' if r3['pass'] else 'FAIL'}")
    print()

    # Test 4: CE sum vs mean
    print("4) CE SUM VS MEAN SCALING")
    model.eval()  # deterministic for this test
    r4 = test4_ce_sum_vs_mean(model, x_full, y_full, device, args.n_train)
    print(f"   ce_sum / (n*ce_mean) = {r4['ce_ratio']:.6f}")
    print(f"   ||grad_sum|| / (n*||grad_mean||) = {r4['grad_ratio']:.6f}")
    print(f"   -> {'PASS' if r4['pass'] else 'FAIL'}")
    print()

    # Test 5: Prior correctness
    print("5) PRIOR TERM CORRECTNESS")
    model.eval()
    r5 = test5_prior_correctness(model, train_data, args.alpha, device, "sum")
    print(f"   U_train = {r5['U_train']:.4f}, U_prior + U_data = {r5['U_prior + U_data']:.4f}")
    print(f"   grad residual ||∇U - (∇CE + αθ)|| = {r5['grad_residual']:.2e}")
    print(f"   -> {'PASS' if r5['pass'] else 'FAIL'}")
    print()

    # Test 6: BN buffer frozen
    print("6) BN BUFFER FROZEN (batchstat_frozen)")
    r6 = test6_bn_buffer_frozen(model, x_full, y_full, args.alpha, device)
    print(f"   max |running_stat_after - running_stat_before| = {r6['max_bn_delta']:.2e}")
    print(f"   -> {'PASS' if r6['pass'] else 'FAIL'}")

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
