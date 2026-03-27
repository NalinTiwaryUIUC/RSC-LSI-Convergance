#!/usr/bin/env python3
"""
Compute convergence + probe diagnostics for the β-sweep conditions.

Primary metrics (tail + early):
  - raw f_nll / beta
  - raw ce_mean_train / beta
  - margin_probe
  - pmax_mean

Secondary metrics:
  - R̂(f_nll), R̂(f_margin) (+ bulk ESS as extra)
  - dist_to_ref (tail + early)
  - theta_norm (tail + early)
  - grad_norm (tail + early), snr (tail + early)

Extra behavioural diagnostics:
  - finite_params & finite_grad (any failures?)
  - abort_suggested (any fired?)
  - grad_norm max (explosion check)

Outputs:
  experiments/summaries/convergence_probes_beta_conditions.md
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "summaries"
OUT_PATH = OUT_DIR / "convergence_probes_beta_conditions.md"

# Expected condition grid from the user:
#  A: T=20000, h=5e-8,  beta=10
#  B: T=20000, h=5e-8,  beta=20
#  C: T=200000,h=5e-9,  beta=10
#  D: T=400000,h=2.5e-9,beta=20
TARGET_BETAS = {10.0, 20.0}
TARGET_H = {5e-8, 5e-9, 2.5e-9}
TARGET_T = {20_000, 200_000, 400_000}
ALPHA_STR = "a0.3"
N_TRAIN = 512
WIDTH = 1

TAIL_FRAC = 0.1
EPS = 1e-12
GRAD_EXPLODE_THRESHOLD = 1e7

PRIMARY = ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean"]
SECONDARY = ["dist_to_ref", "theta_norm", "grad_norm", "snr"]

# samples_metrics traces for R̂/ESS
SAMPLES_FOR_RHAT = ["f_nll", "f_margin"]


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if not math.isfinite(xf):
        return None
    return xf


def _split_rhat(traces: np.ndarray) -> float:
    """Split-Rhat using the usual split-chain heuristic."""
    n_chains, n = traces.shape
    half = n // 2
    if half < 2:
        return float("nan")
    first = traces[:, :half]
    second = traces[:, half : 2 * half]
    split = np.concatenate([first, second], axis=0)
    m, n_per = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    B = n_per * ((chain_means - overall_mean) ** 2).sum() / (m - 1)
    W = chain_vars.mean()
    var_plus = (n_per - 1) / n_per * W + B / n_per
    if W <= 0 or not math.isfinite(var_plus):
        return float("nan")
    return float(np.sqrt(var_plus / W))


def _ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    """Bulk ESS using an autocorrelation truncation at the first non-positive lag."""
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :][: max_lag + 1]
    ac = ac / (ac[0] + EPS)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    if tau <= 0 or not math.isfinite(tau):
        return float("nan")
    return float(n / tau)


def _parse_beta_str(beta_str: str) -> float | None:
    # run naming: beta_str is like "20p0"
    s = beta_str.replace("p", ".")
    try:
        return float(s)
    except Exception:
        return None


@dataclass(frozen=True)
class RunKey:
    T: int
    h: float
    beta: float
    alpha: float


def _parse_run_dir_name(name: str) -> RunKey | None:
    """
    Expected naming pattern (from scripts/run_single_chain.py):
      w{width}_n{n_train}_h{h}_T{T}_a{alpha_str}_b{beta_str}_chain{chain}
    where alpha_str is like 0.3 and beta_str uses p for decimal.
    """
    m = re.match(
        r"w(?P<w>[\d.]+)_n(?P<n>\d+)_h(?P<h>[^_]+)_T(?P<T>\d+)_a(?P<a>[^_]+)_b(?P<b>[^_]+)_chain(?P<chain>\d+)$",
        name,
    )
    if not m:
        return None
    w = float(m.group("w"))
    n = int(m.group("n"))
    T = int(m.group("T"))
    alpha = float(m.group("a").replace("m", "-"))
    h = float(m.group("h"))
    beta = _parse_beta_str(m.group("b"))
    if beta is None:
        return None
    return RunKey(T=T, h=h, beta=beta, alpha=alpha) if (w == WIDTH and n == N_TRAIN) else None


def load_iter_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "iter_metrics.jsonl"
    if not path.exists():
        return []
    recs: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        recs.append(json.loads(line))
    return recs


def load_samples_metrics(run_dir: Path, burnin: int = 0) -> dict[str, np.ndarray]:
    path = run_dir / "samples_metrics.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    steps = z["step"]
    post = steps > burnin
    out: dict[str, np.ndarray] = {}
    for k in SAMPLES_FOR_RHAT:
        if k in z:
            out[k] = z[k][post]
    return out


def tail_early_stats(recs: list[dict], key: str) -> tuple[float, float]:
    """Return (early_mean, tail_mean) across records within one chain."""
    if not recs:
        return float("nan"), float("nan")
    n = len(recs)
    start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
    end_early = max(1, int(n * TAIL_FRAC))
    tail_vals = [_safe_float(r.get(key)) for r in recs[start_tail:] if _safe_float(r.get(key)) is not None]
    early_vals = [_safe_float(r.get(key)) for r in recs[:end_early] if _safe_float(r.get(key)) is not None]
    tail_mean = float(np.mean(tail_vals)) if tail_vals else float("nan")
    early_mean = float(np.mean(early_vals)) if early_vals else float("nan")
    return early_mean, tail_mean


def classify_behavior(early_mean: float, tail_mean: float, tail_std: float) -> str:
    """
    Simple heuristic classification:
      - diverge: chains don't agree in the tail (tail_std large relative to |tail_mean|)
      - drift: tail differs from early but chains agree
      - stabilize: tail close to early and chains agree
    """
    if not math.isfinite(tail_mean) or not math.isfinite(early_mean):
        return "unknown"
    rel_std = tail_std / (abs(tail_mean) + EPS)
    rel_delta = abs(tail_mean - early_mean) / (abs(early_mean) + EPS)
    if rel_std > 0.15:
        return "diverge"
    if rel_delta > 0.1:
        return "drift"
    return "stabilize"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    parsed: list[tuple[RunKey, Path]] = []
    for d in run_dirs:
        k = _parse_run_dir_name(d.name)
        if k is None:
            continue
        if k.alpha != 0.3:
            continue
        if k.beta not in TARGET_BETAS:
            continue
        # numerical tolerance on h
        if not any(abs(k.h - ht) / (abs(ht) + EPS) < 1e-6 for ht in TARGET_H):
            continue
        if k.T not in TARGET_T:
            continue
        parsed.append((k, d))

    if not parsed:
        msg = "No matching run directories found under experiments/runs for the target β/h/T grid."
        OUT_PATH.write_text(msg)
        print(msg)
        return

    # group by (T,h,beta)
    by_key: dict[RunKey, list[Path]] = {}
    for k, d in parsed:
        by_key.setdefault(k, []).append(d)
    for k in by_key:
        by_key[k] = sorted(by_key[k], key=lambda p: p.name)

    lines: list[str] = []
    lines.append("# Convergence probes report (β=10/20 conditions)")
    lines.append("")
    lines.append("Tail fraction: last 10% of logged `iter_metrics` records; early = first 10%.")
    lines.append("R̂ computed from `samples_metrics.npz` using split-chain heuristic; bulk ESS estimated from autocorrelation.")
    lines.append("")

    # deterministic ordering by T, h, beta
    keys_sorted = sorted(by_key.keys(), key=lambda k: (k.T, k.h, k.beta))
    summary_rows: list[dict[str, Any]] = []

    for k in keys_sorted:
        run_dirs_k = by_key[k]
        chain_ids = [re.search(r"chain(\d+)$", d.name).group(1) for d in run_dirs_k if re.search(r"chain(\d+)$", d.name)]
        lines.append(f"## Condition group: T={k.T}, h={k.h:g}, beta={k.beta:g} (n_chains={len(run_dirs_k)})")
        lines.append("")

        iter_per_chain = [load_iter_metrics(d) for d in run_dirs_k]

        # finite/abort/grad explosion diagnostics
        finite_all = True
        abort_any = False
        max_grad = 0.0
        for recs in iter_per_chain:
            for r in recs:
                fp = r.get("finite_params")
                fg = r.get("finite_grad")
                if fp is False or fg is False:
                    finite_all = False
                if r.get("abort_suggested", False):
                    abort_any = True
                g = _safe_float(r.get("grad_norm"))
                if g is not None:
                    max_grad = max(max_grad, g)

        grad_exploded = max_grad > GRAD_EXPLODE_THRESHOLD

        lines.append("### Diagnostics (global across chains)")
        lines.append("")
        lines.append(f"- finite_params & finite_grad always True: **{finite_all}**")
        lines.append(f"- any abort_suggested fired: **{abort_any}**")
        lines.append(f"- max grad_norm seen: **{max_grad:.4g}**")
        lines.append(f"- gradients explode (> {GRAD_EXPLODE_THRESHOLD:.0e}): **{grad_exploded}**")
        lines.append("")

        # Primary and secondary tail/early stats aggregated across chains
        per_chain: dict[str, list[float]] = {key: [] for key in PRIMARY + SECONDARY}
        per_chain_early: dict[str, list[float]] = {key: [] for key in PRIMARY + SECONDARY}
        for recs in iter_per_chain:
            for key in PRIMARY + SECONDARY:
                early_mean, tail_mean = tail_early_stats(recs, key)
                if math.isfinite(early_mean):
                    per_chain_early[key].append(early_mean)
                if math.isfinite(tail_mean):
                    per_chain[key].append(tail_mean)

        def agg_tail(key: str) -> tuple[float, float]:
            vals = per_chain[key]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0

        def agg_early(key: str) -> float:
            vals = per_chain_early[key]
            return float(np.mean(vals)) if vals else float("nan")

        # Primary table
        lines.append("### Primary probes (early vs tail, across chains)")
        lines.append("")
        lines.append("| probe | early_mean | tail_mean | tail_std | behavior |")
        lines.append("|-------|------------|-----------|-----------|----------|")

        # f_nll / beta, ce_mean_train / beta are normalized
        # For behavior classification, use the unnormalized tail/early difference relative to early.
        for key in ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean"]:
            early = agg_early(key)
            tail, tail_std = agg_tail(key)
            behavior = classify_behavior(early, tail, tail_std)
            if key in ("f_nll", "ce_mean_train"):
                lines.append(
                    f"| {key}/β | {early/k.beta:.4f} | {tail/k.beta:.4f} | {tail_std/k.beta:.4f} | {behavior} |"
                )
            else:
                lines.append(f"| {key} | {early:.4f} | {tail:.4f} | {tail_std:.4f} | {behavior} |")
        lines.append("")

        # Secondary table
        lines.append("### Secondary probes + convergence signals (tail, early)")
        lines.append("")
        lines.append("| probe | early_mean | tail_mean | tail_std |")
        lines.append("|-------|------------|-----------|-----------|")
        for key in ["dist_to_ref", "theta_norm", "grad_norm", "snr"]:
            early = agg_early(key)
            tail, tail_std = agg_tail(key)
            lines.append(f"| {key} | {early:.4g} | {tail:.4g} | {tail_std:.4g} |")
        lines.append("")

        # R-hat from samples_metrics (f_nll, f_margin) + ESS (extra)
        # load chain traces
        traces: dict[str, list[np.ndarray]] = {p: [] for p in SAMPLES_FOR_RHAT}
        for d in run_dirs_k:
            sm = load_samples_metrics(d, burnin=0)
            for p in SAMPLES_FOR_RHAT:
                if p in sm:
                    traces[p].append(sm[p])

        lines.append("### R̂ / ESS (from `samples_metrics.npz`)")
        lines.append("")
        lines.append("| probe | R̂ | ESS_bulk | n_samples_used |")
        lines.append("|-------|-----|----------|------------------|")
        rhat_row: dict[str, float] = {}
        for p in SAMPLES_FOR_RHAT:
            if not traces[p]:
                lines.append(f"| {p} | — | — | — |")
                continue
            n_min = min(len(t) for t in traces[p])
            arr = np.stack([t[:n_min] for t in traces[p]], axis=0)
            rhat_val = _split_rhat(arr)
            ess_list = [_ess_bulk(t[:n_min]) for t in traces[p]]
            ess_val = float(np.nanmean(ess_list))
            lines.append(f"| {p} | {rhat_val:.4f} | {ess_val:.1f} | {n_min} |")
            rhat_row[p] = float(rhat_val)
        lines.append("")

        # collect summary row for across-condition table
        summary_rows.append(
            {
                "T": k.T,
                "h": k.h,
                "beta": k.beta,
                "f_nll_beta_tail": agg_tail("f_nll")[0] / k.beta,
                "ce_beta_tail": agg_tail("ce_mean_train")[0] / k.beta,
                "margin_tail": agg_tail("margin_probe")[0],
                "pmax_tail": agg_tail("pmax_mean")[0],
                "Rhat_f_nll": rhat_row.get("f_nll", float("nan")),
                "Rhat_f_margin": rhat_row.get("f_margin", float("nan")),
                "grad_tail": agg_tail("grad_norm")[0],
                "snr_tail": agg_tail("snr")[0],
                "finite_all": finite_all,
                "abort_any": abort_any,
            }
        )
        lines.append("---")
        lines.append("")

    # final compact summary table
    lines.append("## Compact summary (tail means)")
    lines.append("")
    lines.append("| T | h | beta | f_nll/β (tail) | ce_mean_train/β (tail) | margin (tail) | pmax_mean (tail) | R̂(f_nll) | R̂(f_margin) | grad_norm(tail) | snr(tail) |")
    lines.append("|---|---|------|-----------------|-------------------------|----------------|------------------|----------|-------------|-----------------|---------|")
    for row in summary_rows:
        lines.append(
            f"| {row['T']} | {row['h']:.3g} | {row['beta']:.0f} | "
            f"{row['f_nll_beta_tail']:.4f} | {row['ce_beta_tail']:.4f} | "
            f"{row['margin_tail']:.4f} | {row['pmax_tail']:.4f} | "
            f"{row['Rhat_f_nll']:.3f} | {row['Rhat_f_margin']:.3f} | "
            f"{row['grad_tail']:.3g} | {row['snr_tail']:.3g} |"
        )

    # Write report
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote report: {OUT_PATH}")


if __name__ == "__main__":
    main()

