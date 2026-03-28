#!/usr/bin/env python3
"""
Summarize beta-sweep runs (T=6000, 2 chains per beta): tail means, early vs tail,
finite/abort/grad diagnostics, R̂ and ESS for f_nll, f_margin, f_dist.
Output: experiments/summaries/beta_sweep_summary.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "summaries"

# Beta values and run naming: dirs are w1_n512_h5e-08_T6000_a0.3_b{beta_str}_chain{0,1}
BETAS = [1, 3, 10, 30, 100, 300]
TAIL_FRAC = 0.1
PROBES_TAIL = ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean", "dist_to_ref"]
DIAG_KEYS = ["grad_norm", "snr", "U_train", "U_prior", "U_data", "theta_norm"]
# Probes in samples_metrics for Rhat/ESS (user asked for f_nll, f_margin, f_dist)
SAMPLES_PROBES_RHAT = ["f_nll", "f_margin", "f_dist"]
# All probes in samples_metrics for optional extra table
SAMPLES_PROBES_ALL = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
GRAD_EXPLODE_THRESHOLD = 1e7  # flag if max grad_norm > this


def _split_rhat(traces: np.ndarray) -> float:
    n_chains, n = traces.shape
    half = n // 2
    if half < 2:
        return float("nan")
    first = traces[:, :half]
    second = traces[:, half : 2 * half]
    split = np.concatenate([first, second], axis=0)
    m, n_per = split.shape[0], split.shape[1]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    B = n_per * ((chain_means - overall_mean) ** 2).sum() / (m - 1)
    W = chain_vars.mean()
    var_plus = (n_per - 1) / n_per * W + B / n_per
    if W <= 0:
        return float("nan")
    return np.sqrt(var_plus / W)


def _ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    max_lag = max_lag or min(n // 2, 1000)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :][: max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def _beta_from_dir_name(name: str) -> float | None:
    """Parse beta from run dir name, e.g. w1_n512_..._b3p0_chain0 -> 3.0."""
    m = re.search(r"_b(\d+p?\d*)_chain", name)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def get_run_dirs_by_beta(runs_dir: Path) -> dict[float, list[Path]]:
    """Group run dirs by beta. Expects names like ..._b1p0_chain0, ..._b300p0_chain1."""
    by_beta: dict[float, list[Path]] = {}
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        beta = _beta_from_dir_name(d.name)
        if beta is None:
            continue
        if "T6000" not in d.name or "a0.3" not in d.name:
            continue
        if beta not in by_beta:
            by_beta[beta] = []
        by_beta[beta].append(d)
    for beta in by_beta:
        by_beta[beta] = sorted(by_beta[beta], key=lambda p: p.name)
    return by_beta


def load_iter_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "iter_metrics.jsonl"
    if not path.exists():
        return []
    data = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        data.append(json.loads(line))
    return data


def load_samples(run_dir: Path, burnin: int) -> dict[str, np.ndarray]:
    path = run_dir / "samples_metrics.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    steps = z["step"]
    post = steps > burnin
    return {p: z[p][post] for p in SAMPLES_PROBES_ALL if p in z}


def main() -> None:
    out_path = OUT_DIR / "beta_sweep_summary.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_beta = get_run_dirs_by_beta(RUNS_DIR)
    if not by_beta:
        lines = ["# Beta sweep summary", "", "No run dirs found matching ..._b*_chain* with T6000, a0.3.", ""]
        out_path.write_text("\n".join(lines))
        print("\n".join(lines))
        return

    lines = [
        "# Beta sweep summary (T=6000, 2 chains per beta)",
        "",
        "Runs: w1_n512_h5e-08_T6000_a0.3_b{beta}_chain{0,1}. B=0, S=100, log_every=20.",
        "",
    ]

    for beta in sorted(by_beta.keys()):
        run_dirs = by_beta[beta]
        if len(run_dirs) < 1:
            continue
        lines.append(f"## β = {beta}")
        lines.append("")
        B = 0

        # Load iter_metrics per chain
        iter_per_chain = [load_iter_metrics(d) for d in run_dirs]
        n_chains = len(run_dirs)

        # --- Tail means across chains + early vs tail ---
        keys_tail = PROBES_TAIL
        tail_means = {k: [] for k in keys_tail}
        early_means = {k: [] for k in keys_tail}

        finite_all_true = True
        abort_any_fired = False
        grad_exploded = False
        max_grad_norm_seen = 0.0

        for recs in iter_per_chain:
            if not recs:
                continue
            n = len(recs)
            start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
            end_early = max(1, int(n * TAIL_FRAC))
            tail_recs = recs[start_tail:]
            early_recs = recs[:end_early]
            for k in keys_tail:
                tv = [r[k] for r in tail_recs if r.get(k) is not None]
                if tv:
                    tail_means[k].append(np.mean(tv))
                ev = [r[k] for r in early_recs if r.get(k) is not None]
                if ev:
                    early_means[k].append(np.mean(ev))
            finite_all_true = finite_all_true and all(
                r.get("finite_params", True) and r.get("finite_grad", True) for r in recs
            )
            abort_any_fired = abort_any_fired or any(r.get("abort_suggested", False) for r in recs)
            for r in recs:
                g = r.get("grad_norm")
                if g is not None and np.isfinite(g):
                    max_grad_norm_seen = max(max_grad_norm_seen, g)
        if max_grad_norm_seen > GRAD_EXPLODE_THRESHOLD:
            grad_exploded = True

        lines.append("### Tail means across chains")
        lines.append("")
        lines.append("| probe | tail_mean | tail_std (across chains) | early_mean |")
        lines.append("|-------|-----------|----------------------------|------------|")
        for k in keys_tail:
            tm = tail_means[k]
            em = early_means[k]
            if tm:
                tail_mean = np.mean(tm)
                tail_std = np.std(tm) if len(tm) > 1 else 0.0
                early_mean = np.mean(em) if em else float("nan")
                lines.append(f"| {k} | {tail_mean:.4f} | {tail_std:.4f} | {early_mean:.4f} |")
        lines.append("")

        # --- Diagnostics: finite, abort, gradients ---
        lines.append("### Diagnostics")
        lines.append("")
        lines.append(f"- **finite_params & finite_grad all True:** {finite_all_true}")
        lines.append(f"- **any abort_suggested:** {abort_any_fired}")
        lines.append(f"- **gradients explode (max grad_norm > {GRAD_EXPLODE_THRESHOLD:.0e}):** {grad_exploded}")
        if max_grad_norm_seen > 0:
            lines.append(f"- max grad_norm seen: {max_grad_norm_seen:.4g}")
        lines.append("")

        # --- R̂ and raw ESS (f_nll, f_margin, f_dist) ---
        all_traces: dict[str, list[np.ndarray]] = {}
        for d in run_dirs:
            samples = load_samples(d, B)
            for p, arr in samples.items():
                if p not in all_traces:
                    all_traces[p] = []
                all_traces[p].append(arr)

        lines.append("### R̂ and raw ESS (f_nll, f_margin, f_dist)")
        lines.append("")
        lines.append("| probe | R̂ | ESS (bulk) |")
        lines.append("|-------|-----|------------|")
        for p in SAMPLES_PROBES_RHAT:
            if p not in all_traces or not all_traces[p]:
                lines.append(f"| {p} | — | — |")
                continue
            n_min = min(len(t) for t in all_traces[p])
            arr = np.array([t[:n_min] for t in all_traces[p]])
            rhat = _split_rhat(arr)
            ess_list = [_ess_bulk(t[:n_min]) for t in all_traces[p]]
            ess = float(np.nanmean(ess_list))
            lines.append(f"| {p} | {rhat:.4f} | {ess:.1f} |")
        lines.append("")

        # --- Other analytics: theta_norm, U_train, snr (tail) ---
        tail_diag = {k: [] for k in DIAG_KEYS}
        for recs in iter_per_chain:
            if not recs:
                continue
            n = len(recs)
            start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
            tail_recs = recs[start_tail:]
            for k in DIAG_KEYS:
                v = [r[k] for r in tail_recs if r.get(k) is not None]
                if v:
                    tail_diag[k].append(np.mean(v))
        lines.append("### Other (tail means)")
        lines.append("")
        for k in DIAG_KEYS:
            if tail_diag[k]:
                lines.append(f"- {k}: {np.mean(tail_diag[k]):.4g}")
        # n_steps
        n_steps = [len(load_iter_metrics(d)) for d in run_dirs]
        if n_steps:
            lines.append(f"- n_steps (iter_metrics): min={min(n_steps)}, max={max(n_steps)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table across betas: tail_mean for f_nll, ce_mean_train, margin_probe, pmax_mean
    lines.append("## Summary: tail means by β")
    lines.append("")
    lines.append("| β | f_nll | ce_mean_train | margin_probe | pmax_mean | dist_to_ref |")
    lines.append("|---|-------|---------------|--------------|-----------|-------------|")
    for beta in sorted(by_beta.keys()):
        run_dirs = by_beta[beta]
        iter_per_chain = [load_iter_metrics(d) for d in run_dirs]
        tail_means = {k: [] for k in PROBES_TAIL}
        for recs in iter_per_chain:
            if not recs:
                continue
            n = len(recs)
            start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
            tail_recs = recs[start_tail:]
            for k in PROBES_TAIL:
                v = [r[k] for r in tail_recs if r.get(k) is not None]
                if v:
                    tail_means[k].append(np.mean(v))
        vals = [str(beta)]
        for k in PROBES_TAIL:
            if tail_means[k]:
                vals.append(f"{np.mean(tail_means[k]):.4f}")
            else:
                vals.append("—")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
