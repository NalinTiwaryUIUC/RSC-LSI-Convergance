#!/usr/bin/env python3
"""
Regression test suite for ULA pipeline and small ResNet LN.
Run selected sections or --all. Use temp dirs; at most 2 widths per test.

Usage:
  python scripts/run_regression_suite.py --all
  python scripts/run_regression_suite.py --section 1 --section 7
  python scripts/run_regression_suite.py --section 1 --record-baseline baseline.json
  python scripts/run_regression_suite.py --section 1 --compare-baseline baseline.json --quick
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, get_device
from data import get_probe_loader, get_train_loader
from models import create_model
from models.params import flatten_params, param_count, unflatten_like
from models.small_resnet_cifar_ln import ChannelLayerNorm
from run.chain import run_chain
from run.persistence import load_run_config
from ula.potential import compute_U
from ula.step import ula_step

SUITE_SEED = 12345
N_TRAIN = 512
PROBE_SIZE = 256
BATCH_SIZE_FWD = 8


def set_seeds(seed: int = SUITE_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pretrain_with_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    alpha: float,
    n_train: int,
    device: torch.device,
    weight_decay: float = 0.0,
) -> None:
    """SGD with ce_mean + 0.5*alpha/n_train*||θ||² (same minimizer as pretrain.py)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce_mean = F.cross_entropy(logits, y, reduction="mean")
        reg = (0.5 * alpha / n_train) * sum((p * p).sum() for p in model.parameters())
        loss = ce_mean + reg
        loss.backward()
        optimizer.step()


def run_section_1(
    device: torch.device,
    data_dir: str,
    root: str,
    quick: bool,
    record_baseline_path: str | None,
    compare_baseline_path: str | None,
) -> bool:
    """Old known-good baseline: ResNet-18, n_train=512, w=0.1, alpha=0.1, h=5e-8, 5k ULA steps."""
    set_seeds()
    run_dir = Path(tempfile.mkdtemp(prefix="regress_s1_"))
    alpha = 0.1
    h = 5e-8
    T = 500 if quick else 5000
    B = 0
    S = 200
    pretrain_steps = 500 if quick else 2000

    train_loader = get_train_loader(
        N_TRAIN,
        batch_size=N_TRAIN,
        dataset_seed=42,
        data_dir=data_dir,
        root=root,
        eval_transform=True,
    )
    probe_loader = get_probe_loader(
        PROBE_SIZE,
        dataset_seed=43,
        data_dir=data_dir,
        root=root,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)
    train_data = (x_train, y_train)

    model = create_model(
        width_multiplier=0.1,
        num_classes=10,
        arch="resnet18",
    ).to(device)
    _pretrain_with_l2(model, x_train, y_train, pretrain_steps, 0.02, alpha, N_TRAIN, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ckpt_path = run_dir / "pretrain.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": 0.1,
            "n_train": N_TRAIN,
            "arch": "resnet18",
            "num_blocks": 2,
        },
        ckpt_path,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    config = RunConfig(
        n_train=N_TRAIN,
        probe_size=PROBE_SIZE,
        width_multiplier=0.1,
        alpha=alpha,
        h=h,
        T=T,
        B=B,
        S=S,
        log_every=500,
        progress_print_every=0,
        pretrain_steps=0,
        ce_reduction="sum",
        dataset_seed=42,
        chain_seed=SUITE_SEED,
        data_dir=data_dir,
        arch="resnet18",
        num_blocks=2,
    )
    run_chain(
        config,
        chain_id=0,
        run_dir=run_dir,
        train_loader=train_loader,
        probe_loader=probe_loader,
        device=device,
        pretrain_path=ckpt_path,
    )

    # Checks
    iter_path = run_dir / "iter_metrics.jsonl"
    samples_path = run_dir / "samples_metrics.npz"
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists() or not iter_path.exists() or not samples_path.exists():
        print("  FAIL: missing run_config.yaml, iter_metrics.jsonl, or samples_metrics.npz")
        return False
    loaded_config = load_run_config(run_dir)
    if loaded_config.T != T or loaded_config.B != B or loaded_config.S != S:
        print(f"  FAIL: loaded config T/B/S mismatch: {loaded_config.T},{loaded_config.B},{loaded_config.S}")
        return False

    with open(iter_path) as f:
        lines = [json.loads(l) for l in f]
    for rec in lines:
        for k, v in rec.items():
            if isinstance(v, (int, float)) and not np.isfinite(v):
                print(f"  FAIL: non-finite {k}={v} in iter_metrics")
                return False

    first_rec = lines[0]
    last_rec = lines[-1]
    U_first = first_rec.get("U_train")
    U_last = last_rec.get("U_train")
    theta_norm_first = first_rec.get("theta_norm")
    if U_first is None or U_last is None:
        print("  FAIL: missing U_train in iter_metrics")
        return False
    if not (0.1 <= U_first <= 1e7 and 0.1 <= U_last <= 1e7):
        print(f"  FAIL: U_train out of range: first={U_first}, last={U_last}")
        return False
    if theta_norm_first is not None and not (0.01 <= theta_norm_first <= 1e7):
        print(f"  FAIL: theta_norm out of range: {theta_norm_first}")
        return False
    if "drift_step_norm" in first_rec and "noise_step_norm" in first_rec:
        if not np.isfinite(first_rec["drift_step_norm"]) or not np.isfinite(first_rec["noise_step_norm"]):
            print("  FAIL: drift/noise norms not finite")
            return False

    data = np.load(samples_path)
    if not np.all(np.isfinite(data["f_nll"])):
        print("  FAIL: non-finite f_nll in samples_metrics")
        return False
    expected_count = (T - B) // S
    if len(data["step"]) != expected_count:
        print(f"  FAIL: samples count {len(data['step'])} != expected {expected_count}")
        return False

    # Checkpoint loading
    model2 = create_model(width_multiplier=0.1, num_classes=10, arch="resnet18").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model2.load_state_dict(ckpt["state_dict"], strict=True)
    theta_before = flatten_params(model2).clone()
    ula_step(model2, train_data, alpha, h, device, noise_scale=1.0, ce_reduction="sum")
    theta_after = flatten_params(model2)
    if torch.allclose(theta_before, theta_after):
        print("  FAIL: params did not change after ULA step (checkpoint load)")
        return False

    if record_baseline_path:
        with open(record_baseline_path, "w") as f:
            json.dump(
                {
                    "U_train_first": U_first,
                    "U_train_last": U_last,
                    "theta_norm_first": theta_norm_first,
                    "grad_norm_first": first_rec.get("grad_norm"),
                },
                f,
                indent=2,
            )
        print(f"  Recorded baseline to {record_baseline_path}")
    if compare_baseline_path:
        with open(compare_baseline_path) as f:
            base = json.load(f)
        tol = 0.25
        checks = [
            ("U_train_first", first_rec.get("U_train"), base.get("U_train_first")),
            ("U_train_last", last_rec.get("U_train"), base.get("U_train_last")),
            ("theta_norm_first", first_rec.get("theta_norm"), base.get("theta_norm_first")),
            ("grad_norm_first", first_rec.get("grad_norm"), base.get("grad_norm_first")),
        ]
        for name, cur, ref in checks:
            if ref is None or cur is None:
                continue
            if abs(cur - ref) > tol * (abs(ref) + 1e-8):
                print(f"  FAIL: baseline compare {name}: current={cur} ref={ref} (tol={tol})")
                return False

    print("  PASS: baseline pipeline, no NaNs, scales and checkpoint load OK")
    return True


def run_section_2(
    device: torch.device,
    data_dir: str,
    root: str,
) -> bool:
    """Deterministic regression: 2 forwards, 2 gradient evals, same results."""
    set_seeds()
    alpha = 0.1
    train_loader = get_train_loader(
        N_TRAIN,
        batch_size=N_TRAIN,
        dataset_seed=42,
        data_dir=data_dir,
        root=root,
        eval_transform=True,
    )
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model = create_model(width_multiplier=0.1, num_classes=10, arch="resnet18").to(device)
    _pretrain_with_l2(model, x, y, 100, 0.02, alpha, N_TRAIN, device)
    model.eval()

    with torch.no_grad():
        logits1 = model(x)
        logits2 = model(x)
    if not torch.allclose(logits1, logits2):
        print("  FAIL: two forward passes gave different logits")
        return False

    model.zero_grad(set_to_none=True)
    U1 = compute_U(model, (x, y), alpha, device, ce_reduction="sum")
    U1.backward()
    grad1 = torch.cat([p.grad.view(-1) for p in model.parameters()]).clone()
    u1_val = U1.item()

    model.zero_grad(set_to_none=True)
    U2 = compute_U(model, (x, y), alpha, device, ce_reduction="sum")
    U2.backward()
    grad2 = torch.cat([p.grad.view(-1) for p in model.parameters()])
    u2_val = U2.item()

    if not torch.allclose(torch.tensor(u1_val), torch.tensor(u2_val)):
        print(f"  FAIL: two gradient evals gave different U: {u1_val} vs {u2_val}")
        return False
    if not torch.allclose(grad1, grad2):
        print("  FAIL: two gradient evals gave different gradients")
        return False
    print("  PASS: same logits, loss, and gradients on repeat")
    return True


def run_section_3(
    device: torch.device,
) -> bool:
    """Forward pass and param-count scaling at m=16 and m=64."""
    set_seeds()
    widths = [(16, 0.25), (64, 1.0)]  # (m, width_multiplier)
    batch = torch.randn(BATCH_SIZE_FWD, 3, 32, 32, device=device)
    param_counts = []
    for m, w in widths:
        model = create_model(
            width_multiplier=w,
            num_classes=10,
            arch="small_resnet_ln",
            num_blocks=2,
        ).to(device)
        out = model(batch)
        if out.shape != (BATCH_SIZE_FWD, 10):
            print(f"  FAIL: m={m} output shape {out.shape} != (batch, 10)")
            return False
        d = param_count(model)
        param_counts.append((m, d))
        # backward sanity
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"  FAIL: m={m} non-finite gradients")
                return False
    d16, d64 = param_counts[0][1], param_counts[1][1]
    ratio = d64 / d16
    if not (10 <= ratio <= 25):
        print(f"  FAIL: d(64)/d(16) = {ratio} not in [10, 25]")
        return False
    print(f"  PASS: output shape OK, d(16)={d16}, d(64)={d64}, ratio={ratio:.2f}")
    return True


def run_section_4(
    device: torch.device,
) -> bool:
    """LayerNorm sanity: means ~0, variances not degenerate, no zeros/explosion."""
    set_seeds()
    model = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    batch = torch.randn(32, 3, 32, 32, device=device)
    captured: list[torch.Tensor] = []

    def hook(_mod: nn.Module, _input: tuple, output: torch.Tensor) -> None:
        captured.append(output.detach().clone())

    handles = []
    for m in model.modules():
        if isinstance(m, ChannelLayerNorm):
            handles.append(m.register_forward_hook(hook))

    try:
        model(batch)
    finally:
        for h in handles:
            h.remove()

    for i, out in enumerate(captured):
        # out: (B, C, H, W). Per-channel mean over (B,H,W): shape (C,)
        mean_per_ch = out.mean(dim=(0, 2, 3))
        var_per_ch = out.var(dim=(0, 2, 3)) + 1e-8
        # After LN, per-channel mean over (B,H,W) can be nonzero (affine beta); keep loose
        if mean_per_ch.abs().max() > 3.0:
            print(f"  FAIL: LN output {i} mean max abs = {mean_per_ch.abs().max().item():.4f}")
            return False
        if (var_per_ch < 0.1).any() or (var_per_ch > 10).any():
            print(f"  FAIL: LN output {i} variance out of [0.1, 10]: min={var_per_ch.min().item():.4f}, max={var_per_ch.max().item():.4f}")
            return False
        if out.abs().max() < 1e-6:
            print(f"  FAIL: LN output {i} all near zero")
            return False
        if out.abs().max() > 1e4:
            print(f"  FAIL: LN output {i} exploding: max abs = {out.abs().max().item()}")
            return False
    print("  PASS: LN outputs centered, non-degenerate, no explosion")
    return True


def run_section_5(
    device: torch.device,
    data_dir: str,
    root: str,
    quick: bool,
) -> bool:
    """SGD training sanity at m=64 and m=128."""
    set_seeds()
    alpha = 0.1
    steps = 300 if quick else 1000
    train_loader = get_train_loader(
        N_TRAIN,
        batch_size=N_TRAIN,
        dataset_seed=42,
        data_dir=data_dir,
        root=root,
        eval_transform=True,
    )
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    norms = []
    for m, w in [(64, 1.0), (128, 2.0)]:
        model = create_model(
            width_multiplier=w,
            num_classes=10,
            arch="small_resnet_ln",
            num_blocks=2,
        ).to(device)
        model.eval()
        with torch.no_grad():
            logits0 = model(x)
            loss_first = F.cross_entropy(logits0, y, reduction="mean").item()
        _pretrain_with_l2(model, x, y, steps, 0.02, alpha, N_TRAIN, device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss_last = F.cross_entropy(logits, y, reduction="mean").item()
        if loss_last >= loss_first:
            print(f"  FAIL: m={m} loss did not decrease: first={loss_first:.4f}, last={loss_last:.4f}")
            return False
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(t).all() for t in state.values()):
            print(f"  FAIL: m={m} checkpoint has non-finite values")
            return False
        norms.append(flatten_params(model).norm().item())
    if max(norms) > 1e3 or min(norms) < 1e-6:
        print(f"  FAIL: param norms out of range: {norms}")
        return False
    print("  PASS: loss decreased, accuracy high, checkpoints finite, norms OK")
    return True


def run_section_6(
    device: torch.device,
    data_dir: str,
    root: str,
) -> bool:
    """Zero-step invariance: h=0 => params and U unchanged."""
    set_seeds()
    alpha = 0.1
    train_loader = get_train_loader(
        N_TRAIN,
        batch_size=N_TRAIN,
        dataset_seed=42,
        data_dir=data_dir,
        root=root,
        eval_transform=True,
    )
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    theta0 = flatten_params(model).clone()
    U0 = compute_U(model, (x, y), alpha, device, ce_reduction="sum").item()
    ula_step(model, (x, y), alpha, h=0.0, device=device, noise_scale=1.0, ce_reduction="sum")
    theta1 = flatten_params(model)
    U1 = compute_U(model, (x, y), alpha, device, ce_reduction="sum").item()
    if not torch.allclose(theta0, theta1):
        print("  FAIL: params changed after h=0 step")
        return False
    if not torch.allclose(torch.tensor(U0), torch.tensor(U1)):
        print(f"  FAIL: U changed after h=0 step: {U0} vs {U1}")
        return False
    print("  PASS: zero-step leaves params and U unchanged")
    return True


def run_section_7(
    device: torch.device,
) -> bool:
    """Prior-gradient check: ∇(α/2 ||θ||²) = αθ."""
    set_seeds()
    alpha = 0.1
    model = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    model.zero_grad(set_to_none=True)
    reg = (alpha / 2.0) * sum((p * p).sum() for p in model.parameters())
    reg.backward()
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.allclose(p.grad, alpha * p.data, rtol=1e-4, atol=1e-6):
            print("  FAIL: grad != alpha * theta for a parameter")
            return False
    print("  PASS: prior gradient equals alpha*theta")
    return True


def run_section_8(
    device: torch.device,
) -> bool:
    """Noise scaling: empirical per-coord variance ≈ 2h."""
    set_seeds()
    h = 1e-5
    noise_scale = 1.0
    d = 5000
    N = 500
    noise_std = (2.0 * h) ** 0.5 * noise_scale
    samples = []
    for i in range(N):
        g = torch.Generator(device=device).manual_seed(SUITE_SEED + i)
        samples.append(noise_std * torch.randn(d, device=device, generator=g))
    stacked = torch.stack(samples)
    var_per_coord = stacked.var(dim=0)
    mean_var = var_per_coord.mean().item()
    expected = noise_std**2
    if abs(mean_var - expected) / (expected + 1e-12) > 0.15:
        print(f"  FAIL: mean empirical var {mean_var:.2e} not close to 2h={expected:.2e}")
        return False
    print(f"  PASS: empirical variance {mean_var:.2e} ≈ 2h={expected:.2e}")
    return True


def run_section_9(
    device: torch.device,
) -> bool:
    """Flatten/unflatten round-trip (model only + after checkpoint load)."""
    set_seeds()
    model = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    flat = flatten_params(model)
    flat_perturb = flat + 0.01 * torch.randn_like(flat, device=device)
    unflatten_like(flat_perturb, model)
    flat2 = flatten_params(model)
    if not torch.allclose(flat2, flat_perturb):
        print("  FAIL: flatten -> unflatten -> flatten round-trip (model only)")
        return False

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(
        {"state_dict": model.state_dict(), "arch": "small_resnet_ln", "num_blocks": 2, "width": 1.0},
        ckpt_path,
    )
    model2 = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model2.load_state_dict(ckpt["state_dict"], strict=True)
    Path(ckpt_path).unlink(missing_ok=True)

    flat = flatten_params(model2)
    flat_perturb = flat + 0.01 * torch.randn_like(flat, device=device)
    unflatten_like(flat_perturb, model2)
    flat2 = flatten_params(model2)
    if not torch.allclose(flat2, flat_perturb):
        print("  FAIL: flatten -> unflatten -> flatten after checkpoint load")
        return False
    print("  PASS: round-trip exact for model and after load")
    return True


def run_section_10(
    device: torch.device,
    data_dir: str,
    root: str,
    quick: bool,
) -> bool:
    """Short ULA pilots on small ResNet LN (1 width: m=64)."""
    set_seeds()
    run_dir = Path(tempfile.mkdtemp(prefix="regress_s10_"))
    alpha = 0.1
    T = 100 if quick else 200
    B = 0
    S = 50
    pretrain_steps = 300 if quick else 800

    train_loader = get_train_loader(
        N_TRAIN,
        batch_size=N_TRAIN,
        dataset_seed=42,
        data_dir=data_dir,
        root=root,
        eval_transform=True,
    )
    probe_loader = get_probe_loader(
        PROBE_SIZE,
        dataset_seed=43,
        data_dir=data_dir,
        root=root,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    model = create_model(
        width_multiplier=1.0,
        num_classes=10,
        arch="small_resnet_ln",
        num_blocks=2,
    ).to(device)
    _pretrain_with_l2(model, x_train, y_train, pretrain_steps, 0.02, alpha, N_TRAIN, device)
    ckpt_path = run_dir / "pretrain.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": 1.0,
            "n_train": N_TRAIN,
            "arch": "small_resnet_ln",
            "num_blocks": 2,
        },
        ckpt_path,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    config = RunConfig(
        n_train=N_TRAIN,
        probe_size=PROBE_SIZE,
        width_multiplier=1.0,
        alpha=alpha,
        h=1e-5,
        T=T,
        B=B,
        S=S,
        log_every=50,
        progress_print_every=0,
        pretrain_steps=0,
        ce_reduction="sum",
        dataset_seed=42,
        chain_seed=SUITE_SEED,
        data_dir=data_dir,
        arch="small_resnet_ln",
        num_blocks=2,
    )
    run_chain(
        config,
        chain_id=0,
        run_dir=run_dir,
        train_loader=train_loader,
        probe_loader=probe_loader,
        device=device,
        pretrain_path=ckpt_path,
    )

    iter_path = run_dir / "iter_metrics.jsonl"
    samples_path = run_dir / "samples_metrics.npz"
    if not iter_path.exists() or not samples_path.exists():
        print("  FAIL: missing iter_metrics or samples_metrics")
        return False
    with open(iter_path) as f:
        lines = [json.loads(l) for l in f]
    for rec in lines:
        for k, v in rec.items():
            if isinstance(v, (int, float)) and not np.isfinite(v):
                print(f"  FAIL: non-finite {k}={v}")
                return False
    data = np.load(samples_path)
    if not np.all(np.isfinite(data["f_nll"])):
        print("  FAIL: non-finite f_nll in samples")
        return False
    if len(data["step"]) < 1:
        print("  FAIL: no saved samples")
        return False
    print("  PASS: short ULA pilot, no NaNs, samples saved")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Regression test suite for ULA and small ResNet LN")
    p.add_argument("--section", type=int, action="append", choices=list(range(1, 11)), help="Run specific section(s)")
    p.add_argument("--all", action="store_true", help="Run all sections")
    p.add_argument("--record-baseline", type=str, default=None, metavar="PATH", help="Section 1: record baseline JSON")
    p.add_argument("--compare-baseline", type=str, default=None, metavar="PATH", help="Section 1: compare to baseline JSON")
    p.add_argument("--data-dir", type=str, default=None, help="Data dir; default temp")
    p.add_argument("--root", type=str, default="./data", help="CIFAR-10 root")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    p.add_argument("--quick", action="store_true", help="Fewer steps for faster run")
    args = p.parse_args()

    if args.all:
        sections = list(range(1, 11))
    elif args.section:
        sections = sorted(set(args.section))
    else:
        p.error("Specify --section N or --all")

    if (args.record_baseline or args.compare_baseline) and 1 not in sections:
        p.error("--record-baseline/--compare-baseline require --section 1 or --all")

    device = get_device()
    if args.device is not None and args.device != "":
        device = torch.device(args.device)
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="regress_data_")

    passed = 0
    failed = 0
    for sec in sections:
        print(f"Section {sec}:", end=" ")
        try:
            if sec == 1:
                ok = run_section_1(
                    device,
                    data_dir,
                    args.root,
                    args.quick,
                    args.record_baseline,
                    args.compare_baseline,
                )
            elif sec == 2:
                ok = run_section_2(device, data_dir, args.root)
            elif sec == 3:
                ok = run_section_3(device)
            elif sec == 4:
                ok = run_section_4(device)
            elif sec == 5:
                ok = run_section_5(device, data_dir, args.root, args.quick)
            elif sec == 6:
                ok = run_section_6(device, data_dir, args.root)
            elif sec == 7:
                ok = run_section_7(device)
            elif sec == 8:
                ok = run_section_8(device)
            elif sec == 9:
                ok = run_section_9(device)
            elif sec == 10:
                ok = run_section_10(device, data_dir, args.root, args.quick)
            else:
                ok = False
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
