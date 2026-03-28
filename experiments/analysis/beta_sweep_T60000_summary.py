#!/usr/bin/env python3
"""
Summarize longer beta-sweep runs (T=60000, 4 chains per beta=5,10,20,30):
- Primary: raw f_nll / beta, raw ce_mean_train / beta, margin_probe, pmax_mean
- Secondary: R-hat and ESS for f_nll, f_margin; tail grad_norm and snr

Output: experiments/summaries/beta_sweep_T60000_summary.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "summaries"

BETAS = [5.0, 10.0, 20.0, 30.0]
TAIL_FRAC = 0.1
PRIMARY_PROBES = ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean"]
SAMPLES_PROBES_RHAT = ["f_nll", "f_margin"]


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
    return float(np.sqrt(var_plus / W))


def _ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :][: max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return float(n / tau) if tau > 0 else float("nan")


def _beta_from_dir_name(name: str) -> float | None:
    """Parse beta from run dir name, e.g. ..._b5p0_chain0 -> 5.0."""
    m = re.search(r"_b(\d+p?\d*)_chain", name)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_iter_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "iter_metrics.jsonl"
    if not path.exists():
        return []
    data: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        data.append(json.loads(line))
    return data


def load_samples(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "samples_metrics.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    return {k: z[k] for k in SAMPLES_PROBES_RHAT if k in z}


def main() -> None:
    out_path = OUT_DIR / "beta_sweep_T60000_summary.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Beta sweep summary (T=60000, 4 chains per beta)",
        "",
        "Runs: w1_n512_h5e-08_T60000_a0.3_b{beta}_chain{0..3}. B=0, S=200, log_every=50.",
        "",
    ]

    # Per-beta detailed sections
    summary_rows: list[dict] = []

    for beta in BETAS:
        beta_str = str(beta).replace(".", "p")
        run_dirs = sorted(
            d
            for d in RUNS_DIR.iterdir()
            if d.is_dir()
            and "T60000" in d.name
            and f"_b{beta_str}_" in d.name
        )
        if not run_dirs:
            lines.append(f"## β = {beta} — no run dirs found")
            lines.append("")
            continue

        lines.append(f"## β = {beta}")
        lines.append("")

        iter_per_chain = [load_iter_metrics(d) for d in run_dirs]

        # Tail vs early means
        tail_means: dict[str, list[float]] = {k: [] for k in PRIMARY_PROBES}
        early_means: dict[str, list[float]] = {k: [] for k in PRIMARY_PROBES}
        grad_tail: list[float] = []
        snr_tail: list[float] = []

        for recs in iter_per_chain:
            if not recs:
                continue
            n = len(recs)
            start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
            end_early = max(1, int(n * TAIL_FRAC))
            tail_recs = recs[start_tail:]
            early_recs = recs[:end_early]
            for k in PRIMARY_PROBES:
                tv = [r.get(k) for r in tail_recs if r.get(k) is not None]
                ev = [r.get(k) for r in early_recs if r.get(k) is not None]
                if tv:
                    tail_means[k].append(float(np.mean(tv)))
                if ev:
                    early_means[k].append(float(np.mean(ev)))
            gv = [r.get("grad_norm") for r in tail_recs if r.get("grad_norm") is not None]
            sv = [r.get("snr") for r in tail_recs if r.get("snr") is not None]
            if gv:
                grad_tail.append(float(np.mean(gv)))
            if sv:
                snr_tail.append(float(np.mean(sv)))

        # Table of tail vs early; f_nll and ce_mean_train normalized by beta
        lines.append("### Tail vs early means (across chains)")
        lines.append("")
        lines.append("| probe | tail_mean | tail_mean/β (if applicable) | early_mean | early_mean/β |")
        lines.append("|-------|-----------|-----------------------------|------------|--------------|")
        for k in PRIMARY_PROBES:
            tm = tail_means[k]
            em = early_means[k]
            if not tm:
                continue
            tail_mean = float(np.mean(tm))
            early_mean = float(np.mean(em)) if em else float("nan")
            if k in ("f_nll", "ce_mean_train"):
                tail_div = tail_mean / beta
                early_div = early_mean / beta if np.isfinite(early_mean) else float("nan")
                lines.append(
                    f"| {k} | {tail_mean:.4f} | {tail_div:.4f} | {early_mean:.4f} | {early_div:.4f} |"
                )
            else:
                lines.append(
                    f"| {k} | {tail_mean:.4f} | — | {early_mean:.4f} | — |"
                )
        lines.append("")

        # Grad norm and snr
        lines.append("### Gradients and SNR (tail)")
        lines.append("")
        if grad_tail:
            lines.append(f"- grad_norm (tail mean across chains): {float(np.mean(grad_tail)):.4g}")
        if snr_tail:
            lines.append(f"- snr (tail mean across chains): {float(np.mean(snr_tail)):.4g}")
        lines.append("")

        # R-hat and ESS from samples_metrics
        all_traces: dict[str, list[np.ndarray]] = {}
        for d in run_dirs:
            sm = load_samples(d)
            for p, arr in sm.items():
                if p not in all_traces:
                    all_traces[p] = []
                all_traces[p].append(arr)

        lines.append("### R̂ and raw ESS (samples)")
        lines.append("")
        lines.append("| probe | R̂ | ESS (bulk) |")
        lines.append("|-------|-----|------------|")
        rhat_row: dict[str, float] = {}
        for p in SAMPLES_PROBES_RHAT:
            if p not in all_traces or not all_traces[p]:
                lines.append(f"| {p} | — | — |")
                continue
            n_min = min(len(t) for t in all_traces[p])
            arr = np.stack([t[:n_min] for t in all_traces[p]], axis=0)
            rhat_val = _split_rhat(arr)
            ess_list = [_ess_bulk(t[:n_min]) for t in all_traces[p]]
            ess_val = float(np.nanmean(ess_list))
            lines.append(f"| {p} | {rhat_val:.4f} | {ess_val:.1f} |")
            rhat_row[p] = rhat_val
        lines.append("")
        lines.append("---")
        lines.append("")

        # For summary table
        row = {
            "beta": beta,
            "f_nll_tail_div": float(np.mean(tail_means["f_nll"]) / beta)
            if tail_means["f_nll"]
            else float("nan"),
            "ce_tail_div": float(np.mean(tail_means["ce_mean_train"]) / beta)
            if tail_means["ce_mean_train"]
            else float("nan"),
            "margin_tail": float(np.mean(tail_means["margin_probe"]))
            if tail_means["margin_probe"]
            else float("nan"),
            "pmax_tail": float(np.mean(tail_means["pmax_mean"]))
            if tail_means["pmax_mean"]
            else float("nan"),
            "grad_tail": float(np.mean(grad_tail)) if grad_tail else float("nan"),
            "snr_tail": float(np.mean(snr_tail)) if snr_tail else float("nan"),
            "rhat_f_nll": rhat_row.get("f_nll", float("nan")),
            "rhat_f_margin": rhat_row.get("f_margin", float("nan")),
        }
        summary_rows.append(row)

    # Summary table across betas
    if summary_rows:
        lines.append("## Summary across β (tail statistics)")
        lines.append("")
        lines.append(
            "| β | f_nll/β (tail) | ce_mean_train/β (tail) | margin_probe (tail) | "
            "pmax_mean (tail) | grad_norm (tail) | snr (tail) | R̂(f_nll) | R̂(f_margin) |"
        )
        lines.append(
            "|---|----------------|------------------------|----------------------|"
            "-------------------|--------------------|-----------|----------|-------------|"
        )
        for row in summary_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{row['beta']:.0f}",
                        f"{row['f_nll_tail_div']:.4f}",
                        f"{row['ce_tail_div']:.4f}",
                        f"{row['margin_tail']:.4f}",
                        f"{row['pmax_tail']:.4f}",
                        f"{row['grad_tail']:.4g}",
                        f"{row['snr_tail']:.4g}",
                        f"{row['rhat_f_nll']:.4f}",
                        f"{row['rhat_f_margin']:.4f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

