#!/usr/bin/env python3
"""
Tests T1, T2, T3: Partition invariance over time, guard force-fail, BN mode A/B comparison.

T1: At checkpoints 0, 100, 200, compute U/grad one-shot vs accumulated; report ΔU, grad_diff_rel.
T2: Verify guard throws when batchstat_frozen + microbatch.
T3: Two 2k-step chains (eval vs batchstat_frozen), compare drift metrics.
T4: Same θ, same data, one-shot 1024: grad_norm_eval vs grad_norm_batchstat_frozen (ratio check).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, get_device
from data import get_train_loader, get_probe_loader
from models import create_model, flatten_params, unflatten_like
from run.bn_mode import set_bn_batchstats_freeze_buffers
from ula.step import ula_step


def _compute_U_and_grad(model, x, y, alpha, device, ce_reduction="sum"):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    ce = F.cross_entropy(logits, y, reduction=ce_reduction)
    reg = (alpha / 2.0) * sum((p * p).sum() for p in model.parameters())
    U = ce + reg
    U.backward()
    grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return U.item(), grad


def _partition_test(model, x_full, y_full, alpha, device, n_train, bn_mode):
    """One-shot vs accumulated; returns (diff_U, grad_diff_rel)."""
    half = n_train // 2
    x1, x2 = x_full[:half], x_full[half:]
    y1, y2 = y_full[:half], y_full[half:]

    if bn_mode == "eval":
        model.eval()
    else:
        set_bn_batchstats_freeze_buffers(model)

    theta = flatten_params(model).clone()
    unflatten_like(theta.clone(), model)
    U_oneshot, grad_oneshot = _compute_U_and_grad(model, x_full, y_full, alpha, device, "sum")

    unflatten_like(theta.clone(), model)
    model.zero_grad(set_to_none=True)
    ce1 = F.cross_entropy(model(x1), y1, reduction="sum")
    ce1.backward()
    ce2 = F.cross_entropy(model(x2), y2, reduction="sum")
    ce2.backward()
    grad_ce_acc = torch.cat([p.grad.view(-1) for p in model.parameters()])
    prior = (alpha / 2.0) * (theta * theta).sum().item()
    grad_acc = grad_ce_acc + alpha * theta
    U_acc = ce1.item() + ce2.item() + prior

    diff_U = abs(U_oneshot - U_acc)
    diff_grad = (grad_oneshot - grad_acc).abs().max().item()
    grad_diff_rel = diff_grad / (grad_oneshot.norm().item() + 1e-12)
    return diff_U, grad_diff_rel


def run_t1(n_train=1024, width=0.1, alpha=0.1, h=1e-9, seed=42, steps=200, checkpoints=None):
    """T1: Partition invariance at checkpoints 0, 100, 200."""
    if checkpoints is None:
        checkpoints = (0, min(100, steps // 2), steps)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = get_device()
    loader = get_train_loader(n_train, batch_size=n_train, dataset_seed=seed, eval_transform=True)
    x_full, y_full = next(iter(loader))
    x_full, y_full = x_full.to(device), y_full.to(device)
    train_data = (x_full, y_full)

    model = create_model(width_multiplier=width).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(50):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_full), y_full, reduction="mean")
        loss.backward()
        opt.step()

    d = flatten_params(model).numel()
    theta = flatten_params(model).clone()

    results = {"eval": [], "batchstat_frozen": []}
    for step in range(steps + 1):
        if step > 0:
            gen = torch.Generator(device=device).manual_seed(seed + 1000 + step)
            ula_step(model, train_data, alpha, h, device, ce_reduction="sum", generator=gen)
        if step in checkpoints:
            for mode in ["eval", "batchstat_frozen"]:
                diff_U, grad_diff_rel = _partition_test(
                    model, x_full, y_full, alpha, device, n_train, mode
                )
                results[mode].append({"step": step, "diff_U": diff_U, "grad_diff_rel": grad_diff_rel})

    return results


def run_t2():
    """T2: Guard throws when batchstat_frozen + microbatch."""
    from run.chain import run_chain

    config = RunConfig(
        n_train=256, width_multiplier=0.1, h=1e-9, alpha=0.1,
        T=10, B=5, S=5, log_every=5,
        bn_mode="batchstat_frozen",
        effective_batch_size=256,
        num_microbatches=2,
        microbatch_size=128,
    )
    device = get_device()
    loader = get_train_loader(256, batch_size=256, eval_transform=True)
    probe_loader = get_probe_loader(512, dataset_seed=43)
    run_dir = Path("/tmp/t2_guard_test")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_chain(config, 0, run_dir, loader, probe_loader, device)
        return False, "Guard did not throw (FAIL)"
    except ValueError as e:
        if "batchstat_frozen" in str(e).lower() or "microbatch" in str(e).lower():
            return True, f"Guard threw: {e}"
        return False, f"Wrong error: {e}"


def run_t3(n_train=1024, width=0.1, h=1e-9, alpha=0.1, seed=42, steps=2000):
    """T3: Two chains (eval vs batchstat_frozen), compare drift metrics."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="rsc_t3_"))
    proj = Path(__file__).resolve().parents[1]

    runs = {}
    for bn_mode in ["eval", "batchstat_frozen"]:
        runs_dir = tmp / bn_mode
        runs_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "scripts/run_single_chain.py",
            "--width", str(width), "--h", str(h), "--alpha", str(alpha),
            "--chain", "0", "--n_train", str(n_train),
            "--T", str(steps), "--B", str(steps // 4), "--S", "100", "--log-every", "500",
            "--bn-mode", bn_mode,
            "--pretrain-steps", "100",
            "--runs_dir", str(runs_dir),
        ]
        r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None, f"Chain {bn_mode} failed: {r.stderr[:500]}"
        dirs = list(runs_dir.glob("w*chain0"))
        if not dirs:
            return None, f"No run dir for {bn_mode}"
        metrics_path = dirs[0] / "iter_metrics.jsonl"
        if not metrics_path.exists():
            return None, f"No iter_metrics for {bn_mode}"
        lines = [json.loads(l) for l in metrics_path.read_text().strip().splitlines() if l.strip()]
        runs[bn_mode] = lines

    return runs, None


def run_t4(n_train=1024, width=0.1, alpha=0.1, seed=42, pretrain_steps=50):
    """
    Same θ (one checkpoint loaded once), same data subset, same partitioning (one-shot n_train).
    Compute grad_norm under eval vs batchstat_frozen. If ratio ~11× → real modeling difference.
    If ratio ~1 → T3 difference was from different θ / evaluation path.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = get_device()
    loader = get_train_loader(n_train, batch_size=n_train, dataset_seed=seed, eval_transform=True)
    x_full, y_full = next(iter(loader))
    x_full, y_full = x_full.to(device), y_full.to(device)

    model = create_model(width_multiplier=width).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(pretrain_steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_full), y_full, reduction="mean")
        loss.backward()
        opt.step()

    theta = flatten_params(model).clone()

    # Eval: same θ, one-shot
    unflatten_like(theta.clone(), model)
    model.eval()
    _, grad_eval = _compute_U_and_grad(model, x_full, y_full, alpha, device, "sum")
    grad_norm_eval = grad_eval.norm().item()

    # Batchstat_frozen: same θ, same data, one-shot (BN uses batch stats of x_full)
    unflatten_like(theta.clone(), model)
    set_bn_batchstats_freeze_buffers(model)
    _, grad_bf = _compute_U_and_grad(model, x_full, y_full, alpha, device, "sum")
    grad_norm_batchstat_frozen = grad_bf.norm().item()

    ratio = grad_norm_eval / (grad_norm_batchstat_frozen + 1e-12)
    return {
        "grad_norm_eval": grad_norm_eval,
        "grad_norm_batchstat_frozen": grad_norm_batchstat_frozen,
        "ratio_eval_over_bf": ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=str, choices=["t1", "t2", "t3", "t4", "all"], default="all")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--width", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=200, help="T1/T3 chain steps")
    args = p.parse_args()

    if args.test in ("t1", "all"):
        print("=" * 60)
        print("T1: Partition invariance at checkpoints 0, 100, 200")
        print("=" * 60)
        r = run_t1(n_train=args.n_train, width=args.width, steps=args.steps, checkpoints=(0, 100, 200))
        for mode in ["eval", "batchstat_frozen"]:
            print(f"\nbn_mode={mode}:")
            for row in r[mode]:
                print(f"  step {row['step']}: ΔU={row['diff_U']:.6f}, grad_diff_rel={row['grad_diff_rel']:.6f}")
            passes = all(row["grad_diff_rel"] < 1e-3 for row in r[mode])
            print(f"  -> {'PASS (all < 1e-3)' if passes else 'FAIL'}")
        print()

    if args.test in ("t2", "all"):
        print("=" * 60)
        print("T2: Guard force-fail (batchstat_frozen + microbatch)")
        print("=" * 60)
        ok, msg = run_t2()
        print(f"  {msg}")
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        print()

    if args.test in ("t3", "all"):
        print("=" * 60)
        print("T3: BN mode A/B (eval vs batchstat_frozen, 2k steps)")
        print("=" * 60)
        runs, err = run_t3(n_train=args.n_train, width=args.width, steps=args.steps)
        if err:
            print(f"  ERROR: {err}")
        else:
            metrics = ["U_data", "grad_norm", "act_max_abs", "nll_probe_mean"]
            for m in metrics:
                vals_a = [row.get(m) for row in runs["eval"] if row.get(m) is not None]
                vals_b = [row.get(m) for row in runs["batchstat_frozen"] if row.get(m) is not None]
                print(f"  {m}: eval={vals_a[:5]}... batchstat_frozen={vals_b[:5]}...")
            print("  (Compare early drift: eval vs batchstat_frozen)")
        print()

    if args.test in ("t4", "all"):
        print("=" * 60)
        print("T4: Same θ, same data, one-shot — grad_norm eval vs batchstat_frozen")
        print("=" * 60)
        r = run_t4(n_train=args.n_train, width=args.width, pretrain_steps=50)
        print(f"  grad_norm_eval = {r['grad_norm_eval']:.4f}")
        print(f"  grad_norm_batchstat_frozen = {r['grad_norm_batchstat_frozen']:.4f}")
        print(f"  ratio (eval / batchstat_frozen) = {r['ratio_eval_over_bf']:.2f}")
        if r["ratio_eval_over_bf"] > 5:
            print("  -> ~11×-like ratio: real modeling difference (expected).")
        else:
            print("  -> Ratio collapsed: T3 difference likely from different θ / evaluation path.")
        print()


if __name__ == "__main__":
    main()
