#!/usr/bin/env python3
"""
Summarize runs B, C, D (m=64, alpha=0.3, T=20k/60k/200k): probes, R̂, ESS, tail stats,
diagnostics, and trend assessment (stabilize / drift / diverge).
Output: experiments/summaries/horizon_summary_BCD.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "summaries"

# Runs B, C, D: T=20000, 60000, 200000; B=0, S=100, log_every=20
HORIZONS = [
    ("B", 20_000, 0, 100),
    ("C", 60_000, 0, 100),
    ("D", 200_000, 0, 100),
]

PRIMARY_PROBES = ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean"]
SECONDARY_PROBES = ["dist_to_ref", "theta_norm"]
DIAG_PROBES = ["grad_norm", "snr", "U_train", "U_prior", "U_data"]
# Probes saved in samples_metrics (for Rhat/ESS)
SAMPLES_PROBES = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
TAIL_FRAC = 0.1


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


def load_samples(run_dir: Path, B: int) -> tuple[list[np.ndarray], dict[str, list]]:
    path = run_dir / "samples_metrics.npz"
    if not path.exists():
        return [], {}
    z = np.load(path)
    steps = z["step"]
    post = steps > B
    traces = {}
    for p in SAMPLES_PROBES:
        if p not in z:
            continue
        traces[p] = [z[p][post]]
    return list(z["step"][post]), traces


def main() -> None:
    out_path = OUT_DIR / "horizon_summary_BCD.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Horizon summary: Runs B, C, D (m=64, alpha=0.3)",
        "",
        "Runs: B (T=20k), C (T=60k), D (T=200k). B=0, S=100, log_every=20. 4 chains each.",
        "",
    ]

    for label, T, B, S in HORIZONS:
        pattern = f"w1_n512_h5e-08_T{T}_a0.3_chain*"
        run_dirs = sorted(RUNS_DIR.glob(pattern))
        if not run_dirs:
            lines.append(f"## Run {label} (T={T}) — no run dirs found")
            lines.append("")
            continue

        lines.append(f"## Run {label} (T={T}, B={B}, S={S})")
        lines.append("")

        # Load iter_metrics per chain
        iter_per_chain = [load_iter_metrics(d) for d in run_dirs]
        n_chains = len(run_dirs)
        # Load samples for Rhat/ESS
        all_traces = {}
        for d in run_dirs:
            _, traces = load_samples(d, B)
            for p, lst in traces.items():
                if p not in all_traces:
                    all_traces[p] = []
                all_traces[p].append(lst[0])
        # --- R̂ and raw ESS (from samples_metrics) ---
        lines.append("### R̂ and raw ESS (saved samples)")
        lines.append("")
        lines.append("| probe | R̂ | ESS (bulk) |")
        lines.append("|-------|-----|------------|")
        for p in SAMPLES_PROBES:
            if p not in all_traces or not all_traces[p]:
                continue
            n_min = min(len(t) for t in all_traces[p])
            arr = np.array([t[:n_min] for t in all_traces[p]])
            rhat = _split_rhat(arr)
            ess_list = [_ess_bulk(t[:n_min]) for t in all_traces[p]]
            ess = float(np.nanmean(ess_list))
            lines.append(f"| {p} | {rhat:.4f} | {ess:.1f} |")
        lines.append("")
        lines.append("*ce_mean_train and pmax_mean are not in samples_metrics; see tail summaries from iter_metrics below.*")
        lines.append("")

        # --- Tail summaries (last 10% of iter_metrics) ---
        keys_tail = PRIMARY_PROBES + SECONDARY_PROBES
        tail_means = {k: [] for k in keys_tail}
        early_means = {k: [] for k in keys_tail}
        diag_tail = {k: [] for k in DIAG_PROBES}
        finite_ok = []
        abort_any = []

        for chain_idx, recs in enumerate(iter_per_chain):
            if not recs:
                continue
            n = len(recs)
            start_tail = max(0, n - max(1, int(n * TAIL_FRAC)))
            start_early = 0
            end_early = max(1, int(n * TAIL_FRAC))
            tail_recs = recs[start_tail:]
            early_recs = recs[start_early:end_early]
            for k in keys_tail:
                vals = [r[k] for r in tail_recs if r.get(k) is not None]
                if vals:
                    tail_means[k].append(np.mean(vals))
                vals_e = [r[k] for r in early_recs if r.get(k) is not None]
                if vals_e:
                    early_means[k].append(np.mean(vals_e))
            for k in DIAG_PROBES:
                vals = [r[k] for r in tail_recs if r.get(k) is not None]
                if vals:
                    diag_tail[k].append(np.mean(vals))
            finite_ok.append(
                all(r.get("finite_params", True) and r.get("finite_grad", True) for r in recs)
            )
            abort_any.append(any(r.get("abort_suggested", False) for r in recs))

        lines.append("### Tail summaries (last 10% of steps)")
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

        # --- Diagnostics ---
        lines.append("### Diagnostics (tail)")
        lines.append("")
        lines.append("| quantity | tail_mean |")
        lines.append("|----------|-----------|")
        for k in DIAG_PROBES:
            if diag_tail[k]:
                lines.append(f"| {k} | {np.mean(diag_tail[k]):.4g} |")
        lines.append("")
        lines.append(f"- finite_params & finite_grad all True: {all(finite_ok)}")
        lines.append(f"- any abort_suggested: {any(abort_any)}")
        lines.append("")

        # --- Trend assessment ---
        lines.append("### Probe trend assessment (stabilize / drift / diverge)")
        lines.append("")
        for k in PRIMARY_PROBES:
            tm = tail_means.get(k, [])
            em = early_means.get(k, [])
            if not tm or not em:
                lines.append(f"- **{k}**: (no data)")
                continue
            tail_mean = np.mean(tm)
            tail_std = np.std(tm) if len(tm) > 1 else 0.0
            early_mean = np.mean(em)
            delta = tail_mean - early_mean
            # Heuristic: large tail_std across chains -> diverge; consistent drift -> drift; small delta and small std -> stabilize
            if tail_std > 0.5 * (abs(tail_mean) + 1e-8):
                verdict = "diverge (chains spread in tail)"
            elif abs(delta) > 0.1 * (abs(early_mean) + 1e-8):
                verdict = "drift (tail differs from early)"
            else:
                verdict = "stabilize (tail ~ early, chains agree)"
            lines.append(f"- **{k}**: {verdict} (tail_mean={tail_mean:.4f}, early_mean={early_mean:.4f}, tail_std_across_chains={tail_std:.4f})")
        lines.append("")

        # --- Other diagnostics ---
        lines.append("### Other diagnostics")
        lines.append("")
        # logit stats if present
        logit_vals = []
        for recs in iter_per_chain:
            for r in recs[-max(1, len(recs)//10):]:
                if r.get("logit_max_abs") is not None:
                    logit_vals.append(r["logit_max_abs"])
        lines.append("### Fixed / probe logits (tail)")
        lines.append("")
        if logit_vals:
            lines.append(f"- **Model logits** (logit_max_abs): mean = {np.mean(logit_vals):.2f}")
        logits_finite = []
        for recs in iter_per_chain:
            for r in recs[-max(1, len(recs)//10):]:
                if r.get("logits_finite") is not None:
                    logits_finite.append(r["logits_finite"])
        if logits_finite:
            lines.append(f"- logits_finite: {sum(logits_finite)} / {len(logits_finite)} steps True")
        nll_probe = []
        for recs in iter_per_chain:
            for r in recs[-max(1, len(recs)//10):]:
                if r.get("nll_probe_mean") is not None:
                    nll_probe.append(r["nll_probe_mean"])
        if nll_probe:
            lines.append(f"- **Probe logits** (nll_probe_mean): mean = {np.mean(nll_probe):.4f}")
        lines.append("")
        lines.append("### Verdict")
        lines.append("")
        lines.append("- See R̂ table above: values > 1.05 indicate chains have not converged to a common target.")
        lines.append("- Primary probes (f_nll, ce_mean_train, margin_probe): trend assessment above (stabilize / drift / diverge).")
        lines.append("- Between-chain agreement: tail_std across chains is small → chains move together; combined with drift → trajectory is not stationary.")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Overall assessment
    lines.append("## Overall assessment")
    lines.append("")
    lines.append("- **R̂**: Values > 1.05 indicate chains have not converged to a common target; compare across horizons.")
    lines.append("- **ESS**: Raw ESS from saved samples; longer horizons yield more samples and typically higher ESS.")
    lines.append("- **Tail vs early**: Drift suggests the chain is still moving; stabilize suggests stationarity in the tail.")
    lines.append("- **Diverge**: Large tail_std across chains suggests chains are in different regions.")
    lines.append("")

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
