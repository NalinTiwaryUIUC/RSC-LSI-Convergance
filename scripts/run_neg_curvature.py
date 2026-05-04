#!/usr/bin/env python3
"""
Negative-curvature / NME experiment runner.

Implements minimal_negative_curvature_experiment.md verbatim:
    * For each (width, seed), train short SGD on a fixed CIFAR-10 subset
      (n_train=512), then at each requested checkpoint compute the smallest
      algebraic eigenvalues of the model-Hessian / NME operator
      ``N(theta) = H_L(theta) - G(theta)`` (CE loss) on the first 128
      examples of the training subset.
    * Optionally estimate ``T_neg = tr((-N)_+)`` via stochastic Lanczos
      quadrature (default: 8 probes x 30 Lanczos steps).
    * Optionally evaluate local-stability of gamma_emp at 5 nearby points
      around the final checkpoint.

Multi-seed layout (mirrors scripts/bayeslin_lsi_width_convergence.py):
    if ``--seeds`` is set, each run writes to
        ``{parent(out_dir)}/{stem(out_dir)}_seed{s}/``;
    if only ``--seed`` is used, outputs go directly to ``--out-dir``.

Outputs per run dir:
    config.yaml
    curvature_summary.csv               (writeup Files §1)
    negative_eigs_width{w}_seed{s}_{ckpt}.csv  (writeup Files §2)
    local_stability_width{w}_seed{s}.csv  (only at --local-check && checkpoint=final)

Pilot / main examples::

    # Pilot (seeds 0,1,2, final only, top-20, no SLQ, no local check):
    python3 scripts/run_neg_curvature.py \\
        --widths 1,2,4 --seeds 0,1,2 --checkpoints final \\
        --no-slq --no-local-check \\
        --out-dir experiments/neg_curv/pilot

    # Main (writeup defaults):
    python3 scripts/run_neg_curvature.py \\
        --widths 1,2,4 --seeds 0,1,2 --checkpoints init,mid,final \\
        --slq --local-check \\
        --out-dir experiments/neg_curv/main

    # Main matched (train 2000 steps, grid snapshots, curvature only at matched + final):
    python3 scripts/run_neg_curvature.py \\
        --widths 1,2,4 --seeds 0,1,2 --max-steps 2000 \\
        --curvature-mode matched_final \\
        --snapshot-steps 250,500,750,1000,1500,2000 \\
        --matched-train-acc 90 --matched-label matched --match-backup closest_acc \\
        --save-ckpts --slq --num-probes 16 --slq-steps 30 \\
        --no-local-check \\
        --out-dir experiments/neg_curv/main_matched
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from data import get_train_loader  # noqa: E402
from models import create_model  # noqa: E402
from models.params import flatten_params, param_count, unflatten_like  # noqa: E402

from curvature_common import (  # noqa: E402
    cumulative_eta_metrics,
    e_aniso,
    e_iso,
    ggn_ce_vp,
    hvp_full,
    make_torch_linop,
    nme_vp,
    r_eff_neg,
    slq_trace_neg,
    top_k_smallest_eigs,
)


SUMMARY_FIELDNAMES = [
    "width",
    "m",
    "seed",
    "checkpoint",
    "step",
    "p",
    "train_loss",
    "train_acc",
    "curv_loss",
    "curv_acc",
    "gamma_emp",
    "sqrt_m_gamma_emp",
    "T_neg_top20",
    "T_neg_SLQ",
    "T_neg_used",
    "r_eff_top20",
    "r_eff_neg",
    "r_eff_over_p",
    "r_eff_top20_over_p",
    "r_eff_over_sqrt_m",
    "E_iso",
    "E_aniso",
    "E_aniso_over_E_iso",
    "E_aniso_top20_over_E_iso",
    "k50",
    "k80",
    "k90",
    "local_gamma_max",
    "local_gamma_mean",
    "local_gamma_std",
]

EIG_FIELDNAMES = ["rank", "lambda", "eta", "cum_eta", "cum_eta_over_Tneg"]
LOCAL_FIELDNAMES = [
    "width",
    "m",
    "seed",
    "checkpoint",
    "point_idx",
    "epsilon",
    "gamma_emp_local",
    "T_neg_top10_local",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("empty integer csv")
    return out


def _parse_str_csv(s: str) -> list[str]:
    out = [tok.strip() for tok in s.split(",") if tok.strip()]
    if not out:
        raise ValueError("empty string csv")
    return out


def _parse_seeds(s: str) -> list[int] | None:
    s = s.strip()
    if not s:
        return None
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    return out or None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Negative-curvature experiment (NME spectrum) on small_resnet_ln.")
    # widths / seeds / model
    p.add_argument("--widths", type=str, default="1,2,4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seeds; if set, writes to {parent}/{stem}_seed{s}")
    p.add_argument("--arch", type=str, default="small_resnet_ln", choices=["small_resnet_ln", "resnet18"])
    p.add_argument("--num-blocks", type=int, default=1)
    # data
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-curv", type=int, default=128)
    p.add_argument("--dataset-seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    # SGD
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--mid-step", type=int, default=500)
    # checkpoints
    p.add_argument("--checkpoints", type=str, default="init,mid,final",
                   help="Comma list from {init,mid,final} or integer steps (e.g. 250,500).")
    p.add_argument(
        "--curvature-mode",
        type=str,
        default="legacy",
        choices=["legacy", "matched_final"],
        help="legacy: curvature at --checkpoints. matched_final: train up to --max-steps, "
        "save grid snapshots, curvature only at primary matched row + final (see --matched-train-acc).",
    )
    p.add_argument(
        "--snapshot-steps",
        type=str,
        default="",
        help="Comma-separated SGD steps at which to log train loss/acc and (with --save-ckpts) save "
        "ckpt_width{w}_seed{s}_step{S}.pt for backup matching. Steps refer to the same counting as "
        "named checkpoints (final is at --max-steps).",
    )
    p.add_argument(
        "--match-backup",
        type=str,
        default="closest_acc",
        choices=["none", "closest_acc", "closest_loss", "use_final"],
        help="If train accuracy never reaches --matched-train-acc, pick a primary 'matched' row via "
        "this rule using grid snapshots (except use_final / none).",
    )
    p.add_argument(
        "--match-target-loss",
        type=float,
        default=float("nan"),
        help="Target mean CE for --match-backup closest_loss (required when that mode is selected).",
    )
    p.add_argument(
        "--matched-label",
        type=str,
        default="",
        help="curvature_summary 'checkpoint' string for the primary matched row. "
        "Default: matched_acc<int> from --matched-train-acc.",
    )
    p.add_argument("--save-ckpts", action="store_true",
                   help="Also save state_dicts to disk under the run dir.")
    # Lanczos
    p.add_argument("--num-neg", type=int, default=20)
    p.add_argument("--lanczos-steps", type=int, default=80, help="eigsh maxiter")
    p.add_argument("--ncv", type=int, default=-1, help="ncv for eigsh; -1 => 4*k+1")
    p.add_argument("--lanczos-tol", type=float, default=0.0)
    # SLQ
    p.add_argument("--slq", dest="slq", action="store_true", default=True)
    p.add_argument("--no-slq", dest="slq", action="store_false")
    p.add_argument("--num-probes", type=int, default=8)
    p.add_argument("--slq-steps", type=int, default=30)
    # local stability
    p.add_argument("--local-check", dest="local_check", action="store_true", default=True)
    p.add_argument("--no-local-check", dest="local_check", action="store_false")
    p.add_argument("--num-local", type=int, default=5)
    p.add_argument("--eps-rel", type=float, default=0.01)
    p.add_argument("--num-local-neg", type=int, default=10)
    p.add_argument("--local-lanczos-steps", type=int, default=60)
    # numerics
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # optional: first time train_acc >= threshold (after an SGD update), record extra checkpoint
    p.add_argument(
        "--matched-train-acc",
        type=float,
        default=None,
        help="If set, also record curvature when train_acc first reaches this value (percent). "
        "Checkpoint name: matched_acc<int>, e.g. matched_acc95.",
    )
    # output
    p.add_argument("--out-dir", type=str, required=True)
    return p


# ---------------------------------------------------------------------------
# Determinism / device helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(s: str) -> torch.dtype:
    return torch.float64 if s == "float64" else torch.float32


def _np_dtype(s: str) -> np.dtype:
    return np.float64  # always do scipy in float64; bridge casts back


def _parse_ckpts(spec: str, *, mid_step: int, max_steps: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name in _parse_str_csv(spec):
        if name in seen:
            continue
        seen.add(name)
        if name == "init":
            out.append((name, 0))
        elif name == "mid":
            out.append((name, mid_step))
        elif name == "final":
            out.append((name, max_steps))
        elif name.isdigit():
            step = int(name)
            if step < 0 or step > max_steps:
                raise ValueError(f"Numeric checkpoint step {step} out of range [0, {max_steps}]")
            out.append((f"s{step}", step))
        else:
            raise ValueError(f"Unknown checkpoint name: {name}")
    out.sort(key=lambda kv: kv[1])
    return out


def _parse_snapshot_steps(spec: str) -> list[int]:
    if not spec.strip():
        return []
    out = sorted(set(_parse_int_csv(spec)))
    if any(s < 1 for s in out):
        raise ValueError("snapshot steps must be >= 1 (step 0 is init; use explicit 0 if needed)")
    return out


def _effective_matched_label(matched_train_acc: float | None, matched_label: str) -> str:
    lab = matched_label.strip()
    if lab:
        return lab
    if matched_train_acc is None:
        return "matched"
    return f"matched_acc{int(round(float(matched_train_acc)))}"


def _pick_backup_match_step(
    probes: list[tuple[int, float, float]],
    *,
    backup: str,
    acc_target: float,
    target_loss: float,
    max_steps: int,
) -> tuple[int | None, str]:
    """Return (physical_step, reason) for a backup primary checkpoint when threshold was never hit.

    ``probes`` are (step, train_loss, train_acc) sorted by increasing step (grid snapshots).
    """
    if backup == "none":
        return None, "none"
    if backup == "use_final":
        return max_steps, "use_final"

    if not probes:
        return max_steps, "no_grid_probes_fallback_final"

    if backup == "closest_acc":
        best_step, best_d = probes[0][0], abs(probes[0][2] - acc_target)
        for st, _loss, acc in probes:
            d = abs(acc - acc_target)
            if d < best_d or (d == best_d and st < best_step):
                best_d, best_step = d, st
        return best_step, "closest_acc"

    if backup == "closest_loss":
        if not np.isfinite(target_loss):
            return max_steps, "closest_loss_missing_target_fallback_final"
        best_step, best_d = probes[0][0], abs(probes[0][1] - target_loss)
        for st, loss, _acc in probes:
            d = abs(loss - target_loss)
            if d < best_d or (d == best_d and st < best_step):
                best_d, best_step = d, st
        return best_step, "closest_loss"

    raise ValueError(f"Unknown match-backup mode: {backup!r}")


def _append_grid_probe_row(
    out_dir: Path, width: int, seed: int, step: int, train_loss: float, train_acc: float,
) -> None:
    path = out_dir / "train_grid_probe.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["width", "seed", "step", "train_loss", "train_acc"],
        )
        if new_file:
            wr.writeheader()
        wr.writerow({
            "width": width,
            "seed": seed,
            "step": step,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })


def _save_width_ckpt(
    out_dir: Path, width: int, seed: int, step: int, model: torch.nn.Module, args: argparse.Namespace,
) -> Path:
    path = out_dir / f"ckpt_width{width}_seed{seed}_step{step}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "width": width,
            "arch": args.arch,
            "num_blocks": args.num_blocks,
        },
        path,
    )
    return path


def _load_width_ckpt(path: Path, model: torch.nn.Module) -> None:
    blob = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(blob["state_dict"])


# ---------------------------------------------------------------------------
# Data loading (fixed subset; train + curv first 128)
# ---------------------------------------------------------------------------

def _load_fixed_subset(
    n_train: int,
    n_curv: int,
    dataset_seed: int,
    data_dir: str,
    root: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    use_gpu = device.type == "cuda"
    loader = get_train_loader(
        n_train,
        batch_size=n_train,
        dataset_seed=dataset_seed,
        data_dir=data_dir,
        root=root,
        pin_memory=use_gpu,
        eval_transform=True,
    )
    x_train, y_train = next(iter(loader))
    x_train = x_train.to(device=device, dtype=torch_dtype, non_blocking=True)
    y_train = y_train.to(device=device, non_blocking=True).long()
    if n_curv > n_train:
        raise ValueError(f"n_curv={n_curv} > n_train={n_train}")
    x_curv = x_train[:n_curv]
    y_curv = y_train[:n_curv]
    return x_train, y_train, x_curv, y_curv


# ---------------------------------------------------------------------------
# Model build (deterministic per seed)
# ---------------------------------------------------------------------------

def _build_model(
    width: float,
    arch: str,
    num_blocks: int,
    seed: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> torch.nn.Module:
    _set_seed(seed)
    model = create_model(
        width_multiplier=float(width),
        num_classes=10,
        arch=arch,
        num_blocks=num_blocks,
    )
    model = model.to(device=device, dtype=torch_dtype)
    return model


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _eval_loss_acc(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="mean").item()
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100.0
    return float(ce), float(acc)


# ---------------------------------------------------------------------------
# Curvature pipeline at a checkpoint
# ---------------------------------------------------------------------------

def _curvature_pipeline(
    model: torch.nn.Module,
    x_curv: torch.Tensor,
    y_curv: torch.Tensor,
    p: int,
    *,
    num_neg: int,
    lanczos_steps: int,
    ncv: int,
    lanczos_tol: float,
    slq: bool,
    num_probes: int,
    slq_steps: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    np_dtype: np.dtype,
    probe_rng: np.random.Generator,
) -> dict:
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)

    def matvec_torch(v_t: torch.Tensor) -> torch.Tensor:
        return nme_vp(model, x_curv, y_curv, v_t)

    linop = make_torch_linop(
        matvec_torch, p,
        device=device,
        torch_dtype=torch_dtype,
        np_dtype=np_dtype,
    )

    ncv_eff = ncv if ncv > 0 else None
    eigs = top_k_smallest_eigs(
        linop, k=num_neg, ncv=ncv_eff, maxiter=lanczos_steps, tol=lanczos_tol,
    )

    metrics = cumulative_eta_metrics(eigs)

    T_neg_slq = float("nan")
    if slq and num_probes > 0 and slq_steps > 0:
        def matvec_np(v_np: np.ndarray) -> np.ndarray:
            return linop.matvec(v_np)
        T_neg_slq = float(slq_trace_neg(
            matvec_np, p,
            num_probes=num_probes,
            lanczos_steps=slq_steps,
            rng=probe_rng,
        ))

    return {
        "eigs": np.asarray(eigs, dtype=np.float64),
        "gamma_emp": float(metrics["gamma_emp"]),
        "T_neg_top": float(metrics["T_neg_top"]),
        "T_neg_slq": T_neg_slq,
        "eta_sorted": metrics["eta_sorted"],
        "cum_eta": metrics["cum_eta"],
        "k50": int(metrics["k50"]),
        "k80": int(metrics["k80"]),
        "k90": int(metrics["k90"]),
    }


# ---------------------------------------------------------------------------
# Local stability check at the final checkpoint
# ---------------------------------------------------------------------------

def _local_stability(
    model: torch.nn.Module,
    theta_t: torch.Tensor,
    theta_init: torch.Tensor,
    x_curv: torch.Tensor,
    y_curv: torch.Tensor,
    p: int,
    *,
    num_local: int,
    eps_rel: float,
    num_local_neg: int,
    local_lanczos_steps: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    np_dtype: np.dtype,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    if num_local <= 0 or num_local_neg <= 0:
        return [], {"local_gamma_max": float("nan"),
                    "local_gamma_mean": float("nan"),
                    "local_gamma_std": float("nan")}

    delta = theta_t - theta_init
    delta_norm = float(torch.linalg.norm(delta).item())
    eps = eps_rel * delta_norm / max(np.sqrt(p), 1.0)

    rows: list[dict] = []
    gammas: list[float] = []
    for j in range(num_local):
        z_np = rng.normal(size=p)
        z_np = z_np / max(np.linalg.norm(z_np), 1e-30)
        z = torch.as_tensor(z_np, dtype=torch_dtype, device=device)
        unflatten_like(theta_t + eps * z, model)

        def matvec_torch(v_t: torch.Tensor) -> torch.Tensor:
            return nme_vp(model, x_curv, y_curv, v_t)

        linop = make_torch_linop(matvec_torch, p, device=device,
                                 torch_dtype=torch_dtype, np_dtype=np_dtype)
        ncv = min(max(2 * num_local_neg + 1, 4 * num_local_neg + 1), p - 1)
        eigs = top_k_smallest_eigs(
            linop, k=num_local_neg, ncv=ncv, maxiter=local_lanczos_steps, tol=0.0,
        )
        m = cumulative_eta_metrics(eigs)
        rows.append({
            "point_idx": j,
            "epsilon": eps,
            "gamma_emp_local": float(m["gamma_emp"]),
            "T_neg_top10_local": float(m["T_neg_top"]),
        })
        gammas.append(float(m["gamma_emp"]))

    unflatten_like(theta_t, model)

    arr = np.asarray(gammas, dtype=np.float64)
    summary = {
        "local_gamma_max": float(arr.max()) if arr.size > 0 else float("nan"),
        "local_gamma_mean": float(arr.mean()) if arr.size > 0 else float("nan"),
        "local_gamma_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }
    return rows, summary


def _record_checkpoint_metrics(
    *,
    model: torch.nn.Module,
    out_dir: Path,
    w: int,
    seed: int,
    m_hidden: int,
    p_int: int,
    ckpt_name: str,
    phys_step: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_curv: torch.Tensor,
    y_curv: torch.Tensor,
    theta_init: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    torch_dtype: torch.dtype,
    np_dtype: np.dtype,
    probe_rng: np.random.Generator,
    local_rng: np.random.Generator,
    summary_rows: list[dict],
    eig_rows_by_file: dict[str, list[dict]],
    local_rows_by_file: dict[str, list[dict]],
) -> None:
    """Evaluate train/curv batch metrics, run Lanczos (+ optional SLQ), append CSV rows."""
    if args.save_ckpts:
        ckpt_path = out_dir / f"ckpt_width{w}_seed{seed}_{ckpt_name}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "step": phys_step,
             "width": w, "arch": args.arch, "num_blocks": args.num_blocks},
            ckpt_path,
        )

    train_loss, train_acc = _eval_loss_acc(model, x_train, y_train)
    curv_loss, curv_acc = _eval_loss_acc(model, x_curv, y_curv)

    cur = _curvature_pipeline(
        model, x_curv, y_curv, p_int,
        num_neg=args.num_neg,
        lanczos_steps=args.lanczos_steps,
        ncv=args.ncv,
        lanczos_tol=args.lanczos_tol,
        slq=args.slq,
        num_probes=args.num_probes,
        slq_steps=args.slq_steps,
        device=device,
        torch_dtype=torch_dtype,
        np_dtype=np_dtype,
        probe_rng=probe_rng,
    )

    T_top = float(cur["T_neg_top"])
    T_slq = cur["T_neg_slq"]
    T_used = float(T_slq) if (args.slq and np.isfinite(T_slq) and float(T_slq) > 0.0) else T_top
    gamma_emp = float(cur["gamma_emp"])
    r_top = r_eff_neg(T_top, gamma_emp)
    r_eff = r_eff_neg(T_used, gamma_emp)
    E_iso = e_iso(gamma_emp, p_int)
    E_an = e_aniso(T_used)
    E_ratio = (E_an / E_iso) if E_iso > 0.0 else float("nan")

    local_summary = {"local_gamma_max": float("nan"),
                     "local_gamma_mean": float("nan"),
                     "local_gamma_std": float("nan")}
    if args.local_check and ckpt_name == "final":
        with torch.no_grad():
            theta_final = flatten_params(model).clone()
        local_rows, local_summary = _local_stability(
            model,
            theta_t=theta_final,
            theta_init=theta_init,
            x_curv=x_curv,
            y_curv=y_curv,
            p=p_int,
            num_local=args.num_local,
            eps_rel=args.eps_rel,
            num_local_neg=args.num_local_neg,
            local_lanczos_steps=args.local_lanczos_steps,
            device=device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            rng=local_rng,
        )
        fname = f"local_stability_width{w}_seed{seed}.csv"
        rows = local_rows_by_file.setdefault(fname, [])
        for r in local_rows:
            rows.append({
                "width": w, "m": m_hidden, "seed": seed, "checkpoint": ckpt_name,
                **r,
            })

    eigs_sorted_asc = np.sort(np.asarray(cur["eigs"], dtype=np.float64))
    T_for_frac = T_used if T_used > 0.0 else 1.0
    eig_fname = f"negative_eigs_width{w}_seed{seed}_{ckpt_name}.csv"
    eig_rows = eig_rows_by_file.setdefault(eig_fname, [])
    cum = 0.0
    for rank_idx, lam in enumerate(eigs_sorted_asc, start=1):
        eta_val = max(0.0, -float(lam))
        cum += eta_val
        eig_rows.append({
            "rank": rank_idx,
            "lambda": float(lam),
            "eta": eta_val,
            "cum_eta": cum,
            "cum_eta_over_Tneg": (cum / T_for_frac) if T_for_frac > 0.0 else float("nan"),
        })

    summary_rows.append({
        "width": w,
        "m": m_hidden,
        "seed": seed,
        "checkpoint": ckpt_name,
        "step": phys_step,
        "p": p_int,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "curv_loss": curv_loss,
        "curv_acc": curv_acc,
        "gamma_emp": gamma_emp,
        "sqrt_m_gamma_emp": float(np.sqrt(m_hidden) * gamma_emp),
        "T_neg_top20": T_top,
        "T_neg_SLQ": T_slq,
        "T_neg_used": T_used,
        "r_eff_top20": r_top,
        "r_eff_neg": r_eff,
        "r_eff_over_p": (r_eff / p_int) if p_int > 0 else float("nan"),
        "r_eff_top20_over_p": (r_top / p_int) if p_int > 0 else float("nan"),
        "r_eff_over_sqrt_m": r_eff / float(np.sqrt(m_hidden)),
        "E_iso": E_iso,
        "E_aniso": E_an,
        "E_aniso_over_E_iso": E_ratio,
        "E_aniso_top20_over_E_iso": (T_top / E_iso) if E_iso > 0.0 else float("nan"),
        "k50": cur["k50"],
        "k80": cur["k80"],
        "k90": cur["k90"],
        **local_summary,
    })


# ---------------------------------------------------------------------------
# Main per-seed loop
# ---------------------------------------------------------------------------

def _run_one_seed(out_dir: Path, seed: int, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    torch_dtype = _torch_dtype(args.dtype)
    np_dtype = _np_dtype(args.dtype)

    snap_list = _parse_snapshot_steps(args.snapshot_steps)
    for s in snap_list:
        if s > args.max_steps:
            raise ValueError(f"snapshot step {s} exceeds --max-steps {args.max_steps}")
    snap_steps_set = set(snap_list)
    if args.curvature_mode == "matched_final":
        if args.matched_train_acc is None:
            raise ValueError("curvature_mode=matched_final requires --matched-train-acc (e.g. 90)")
        if args.match_backup == "closest_loss" and not np.isfinite(args.match_target_loss):
            raise ValueError("curvature_mode=matched_final with match-backup closest_loss requires "
                             "finite --match-target-loss")
        if snap_steps_set and args.match_backup in ("closest_acc", "closest_loss") and not args.save_ckpts:
            args.save_ckpts = True
            print("matched_final: enabling --save-ckpts (grid reload for backup matching)", flush=True)

    matched_lab = _effective_matched_label(args.matched_train_acc, args.matched_label)

    cfg = {
        "experiment": "negative_curvature",
        "arch": args.arch,
        "num_blocks": args.num_blocks,
        "widths": _parse_int_csv(args.widths),
        "seed": seed,
        "n_train": args.n_train,
        "n_curv": args.n_curv,
        "dataset_seed": args.dataset_seed,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "mid_step": args.mid_step,
        "checkpoints": _parse_str_csv(args.checkpoints),
        "curvature_mode": args.curvature_mode,
        "snapshot_steps": snap_list,
        "match_backup": args.match_backup,
        "match_target_loss": float(args.match_target_loss)
        if np.isfinite(args.match_target_loss) else None,
        "matched_label_effective": matched_lab,
        "num_neg": args.num_neg,
        "lanczos_steps": args.lanczos_steps,
        "ncv": args.ncv,
        "slq": args.slq,
        "num_probes": args.num_probes,
        "slq_steps": args.slq_steps,
        "local_check": args.local_check,
        "num_local": args.num_local,
        "eps_rel": args.eps_rel,
        "num_local_neg": args.num_local_neg,
        "local_lanczos_steps": args.local_lanczos_steps,
        "dtype": args.dtype,
        "device": str(device),
        "init": "factory_default",
        "loss": "cross_entropy_mean",
        "matched_train_acc": getattr(args, "matched_train_acc", None),
    }
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.curvature_mode == "matched_final":
        ckpts: list[tuple[str, int]] = [("matched", -1), ("final", args.max_steps)]
        ckpt_steps: list[int] = []
        ckpt_name_by_step: dict[int, str] = {}
    else:
        ckpts = _parse_ckpts(args.checkpoints, mid_step=args.mid_step, max_steps=args.max_steps)
        ckpt_steps = [s for _, s in ckpts]
        ckpt_name_by_step = {s: name for name, s in ckpts}

    x_train, y_train, x_curv, y_curv = _load_fixed_subset(
        n_train=args.n_train,
        n_curv=args.n_curv,
        dataset_seed=args.dataset_seed,
        data_dir=args.data_dir,
        root=args.root,
        device=device,
        torch_dtype=torch_dtype,
    )

    summary_rows: list[dict] = []
    eig_rows_by_file: dict[str, list[dict]] = {}
    local_rows_by_file: dict[str, list[dict]] = {}

    for w in _parse_int_csv(args.widths):
        m_hidden = max(1, int(64 * float(w)))
        model = _build_model(
            width=float(w),
            arch=args.arch,
            num_blocks=args.num_blocks,
            seed=seed,
            device=device,
            torch_dtype=torch_dtype,
        )
        p = param_count(model)

        with torch.no_grad():
            theta_init = flatten_params(model).clone()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )

        if args.curvature_mode == "matched_final":
            grid_probes: list[tuple[int, float, float]] = []
            matched_fired = False
            num_updates = 0
            for step in range(0, args.max_steps + 1):
                if step in snap_steps_set and step > 0:
                    if args.save_ckpts:
                        _save_width_ckpt(out_dir, w, seed, step, model, args)
                    tl, ta = _eval_loss_acc(model, x_train, y_train)
                    _append_grid_probe_row(out_dir, w, seed, step, tl, ta)
                    grid_probes.append((step, tl, ta))

                if step == args.max_steps:
                    ps0 = (int(seed) * 1_000_003 + 424242 + int(w) * 7) % (2**31)
                    probe_rng = np.random.default_rng(ps0)
                    local_rng = np.random.default_rng(ps0 + 1)
                    _record_checkpoint_metrics(
                        model=model,
                        out_dir=out_dir,
                        w=w,
                        seed=seed,
                        m_hidden=m_hidden,
                        p_int=int(p),
                        ckpt_name="final",
                        phys_step=step,
                        x_train=x_train,
                        y_train=y_train,
                        x_curv=x_curv,
                        y_curv=y_curv,
                        theta_init=theta_init,
                        args=args,
                        device=device,
                        torch_dtype=torch_dtype,
                        np_dtype=np_dtype,
                        probe_rng=probe_rng,
                        local_rng=local_rng,
                        summary_rows=summary_rows,
                        eig_rows_by_file=eig_rows_by_file,
                        local_rows_by_file=local_rows_by_file,
                    )
                    model.train()
                    break

                optimizer.zero_grad(set_to_none=True)
                model.train()
                logits = model(x_train)
                loss = F.cross_entropy(logits, y_train, reduction="mean")
                loss.backward()
                optimizer.step()
                num_updates += 1

                if args.matched_train_acc is not None and not matched_fired:
                    _tr_loss, tr_acc = _eval_loss_acc(model, x_train, y_train)
                    if tr_acc >= float(args.matched_train_acc):
                        ps = (int(seed) * 991 + int(w) * 17 + num_updates * 1_009 + 99_999) % (2**31)
                        loc = (ps + 1) % (2**31)
                        _record_checkpoint_metrics(
                            model=model,
                            out_dir=out_dir,
                            w=w,
                            seed=seed,
                            m_hidden=m_hidden,
                            p_int=int(p),
                            ckpt_name=matched_lab,
                            phys_step=num_updates,
                            x_train=x_train,
                            y_train=y_train,
                            x_curv=x_curv,
                            y_curv=y_curv,
                            theta_init=theta_init,
                            args=args,
                            device=device,
                            torch_dtype=torch_dtype,
                            np_dtype=np_dtype,
                            probe_rng=np.random.default_rng(ps),
                            local_rng=np.random.default_rng(loc),
                            summary_rows=summary_rows,
                            eig_rows_by_file=eig_rows_by_file,
                            local_rows_by_file=local_rows_by_file,
                        )
                        model.train()
                        matched_fired = True

            if not matched_fired:
                if args.match_backup == "none":
                    raise RuntimeError(
                        f"width={w} seed={seed}: never reached train_acc>={args.matched_train_acc} "
                        "and --match-backup none",
                    )
                st_sel, reason = _pick_backup_match_step(
                    grid_probes,
                    backup=args.match_backup,
                    acc_target=float(args.matched_train_acc),
                    target_loss=float(args.match_target_loss),
                    max_steps=args.max_steps,
                )
                ps = (int(seed) * 991 + int(w) * 17 + 777_777) % (2**31)
                loc = (ps + 1) % (2**31)
                ckpt_path = out_dir / f"ckpt_width{w}_seed{seed}_step{st_sel}.pt"
                need_load = (
                    st_sel < args.max_steps
                    and args.match_backup in ("closest_acc", "closest_loss")
                    and ckpt_path.exists()
                )
                if need_load:
                    _load_width_ckpt(ckpt_path, model)
                elif st_sel < args.max_steps:
                    print(
                        f"warning: width={w} seed={seed}: backup step {st_sel} ckpt missing "
                        f"({reason}); recording '{matched_lab}' at final weights.",
                        flush=True,
                    )
                _record_checkpoint_metrics(
                    model=model,
                    out_dir=out_dir,
                    w=w,
                    seed=seed,
                    m_hidden=m_hidden,
                    p_int=int(p),
                    ckpt_name=matched_lab,
                    phys_step=int(st_sel),
                    x_train=x_train,
                    y_train=y_train,
                    x_curv=x_curv,
                    y_curv=y_curv,
                    theta_init=theta_init,
                    args=args,
                    device=device,
                    torch_dtype=torch_dtype,
                    np_dtype=np_dtype,
                    probe_rng=np.random.default_rng(ps),
                    local_rng=np.random.default_rng(loc),
                    summary_rows=summary_rows,
                    eig_rows_by_file=eig_rows_by_file,
                    local_rows_by_file=local_rows_by_file,
                )
                model.train()
            continue

        matched_fired = False
        num_updates = 0
        for step in range(0, args.max_steps + 1):
            if step in ckpt_steps:
                ckpt_name = ckpt_name_by_step[step]
                probe_seed = (int(seed) * 1_000_003 + ckpt_steps.index(step) * 17 + int(w) * 7) % (2**31)
                probe_rng = np.random.default_rng(probe_seed)
                local_rng = np.random.default_rng(probe_seed + 1)
                _record_checkpoint_metrics(
                    model=model,
                    out_dir=out_dir,
                    w=w,
                    seed=seed,
                    m_hidden=m_hidden,
                    p_int=int(p),
                    ckpt_name=ckpt_name,
                    phys_step=step,
                    x_train=x_train,
                    y_train=y_train,
                    x_curv=x_curv,
                    y_curv=y_curv,
                    theta_init=theta_init,
                    args=args,
                    device=device,
                    torch_dtype=torch_dtype,
                    np_dtype=np_dtype,
                    probe_rng=probe_rng,
                    local_rng=local_rng,
                    summary_rows=summary_rows,
                    eig_rows_by_file=eig_rows_by_file,
                    local_rows_by_file=local_rows_by_file,
                )
                model.train()

            if step == args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train, reduction="mean")
            loss.backward()
            optimizer.step()
            num_updates += 1

            if args.matched_train_acc is not None and not matched_fired:
                _tr_loss, tr_acc = _eval_loss_acc(model, x_train, y_train)
                if tr_acc >= float(args.matched_train_acc):
                    mname = matched_lab
                    ps = (int(seed) * 991 + int(w) * 17 + num_updates * 1_009 + 99_999) % (2**31)
                    loc = (ps + 1) % (2**31)
                    _record_checkpoint_metrics(
                        model=model,
                        out_dir=out_dir,
                        w=w,
                        seed=seed,
                        m_hidden=m_hidden,
                        p_int=int(p),
                        ckpt_name=mname,
                        phys_step=num_updates,
                        x_train=x_train,
                        y_train=y_train,
                        x_curv=x_curv,
                        y_curv=y_curv,
                        theta_init=theta_init,
                        args=args,
                        device=device,
                        torch_dtype=torch_dtype,
                        np_dtype=np_dtype,
                        probe_rng=np.random.default_rng(ps),
                        local_rng=np.random.default_rng(loc),
                        summary_rows=summary_rows,
                        eig_rows_by_file=eig_rows_by_file,
                        local_rows_by_file=local_rows_by_file,
                    )
                    model.train()
                    matched_fired = True

    with (out_dir / "curvature_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        wr.writeheader()
        wr.writerows(summary_rows)

    for fname, rows in eig_rows_by_file.items():
        with (out_dir / fname).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=EIG_FIELDNAMES)
            wr.writeheader()
            wr.writerows(rows)

    for fname, rows in local_rows_by_file.items():
        with (out_dir / fname).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=LOCAL_FIELDNAMES)
            wr.writeheader()
            wr.writerows(rows)

    ckpt_names_out = (
        ["matched", "final"] if args.curvature_mode == "matched_final"
        else [name for name, _ in ckpts]
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "seed": seed,
        "rows": len(summary_rows),
        "widths": _parse_int_csv(args.widths),
        "checkpoints": ckpt_names_out,
        "curvature_mode": args.curvature_mode,
    }, indent=2))


def _resolve_run_dirs(out_dir: Path, seeds_csv: str | None, single_seed: int) -> list[tuple[int, Path]]:
    if seeds_csv is None:
        parsed = None
    else:
        parsed = _parse_seeds(seeds_csv)
    if parsed:
        parent = out_dir.parent
        stem = out_dir.name
        return [(s, parent / f"{stem}_seed{s}") for s in parsed]
    return [(single_seed, out_dir)]


def main() -> None:
    args = build_argparser().parse_args()
    out_base = Path(args.out_dir)
    seeds_csv = args.seeds.strip() or None
    runs = _resolve_run_dirs(out_base, seeds_csv, args.seed)
    for seed_val, run_dir in runs:
        _run_one_seed(run_dir, seed_val, args)


if __name__ == "__main__":
    main()
