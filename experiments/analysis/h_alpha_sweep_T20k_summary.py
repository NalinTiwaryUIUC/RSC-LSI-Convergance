#!/usr/bin/env python3
"""
Summarize T=20000 runs: 2×2 grid (h ∈ {5e-7, 5e-6}, α ∈ {3, 30}), 4 chains, β=1.

Primary probes + R̂/ESS + extended diagnostics to interpret bad f_nll:
  U_train, U_prior, U_data; prior_ratio = U_prior/U_train; data_ratio = U_data/U_train;
  data_prior_ratio = U_data/U_prior; |U_data_minus_ce|; nll_probe_mean, logit scale;
  drift vs noise step norms; delta_U.

Output: experiments/summaries/h_alpha_sweep_T20k_summary.md
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"
OUT_PATH = Path(__file__).resolve().parents[2] / "experiments" / "summaries" / "h_alpha_sweep_T20k_summary.md"

PAT = re.compile(
    r"w1_n512_h(?P<h>[^_]+)_T20000_a(?P<a>[^_]+)_b(?P<b>[^_]+)_chain(?P<c>\d+)$"
)
TAIL_FRAC = 0.1


def split_rhat(traces: np.ndarray) -> float:
    n_chains, n = traces.shape
    half = n // 2
    if half < 2:
        return float("nan")
    first, second = traces[:, :half], traces[:, half : 2 * half]
    split = np.concatenate([first, second], axis=0)
    m, n_per = split.shape
    means = split.mean(axis=1)
    vars_ = split.var(axis=1, ddof=1)
    overall = means.mean()
    B = n_per * ((means - overall) ** 2).sum() / (m - 1)
    W = vars_.mean()
    var_plus = (n_per - 1) / n_per * W + B / n_per
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_plus / W))


def ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    max_lag = max_lag or min(n // 2, 1000)
    ac = np.correlate(trace, trace, mode="full")[n - 1 :][: max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return float(n / tau) if tau > 0 else float("nan")


def load_iter(p: Path) -> list[dict]:
    recs = []
    for line in (p / "iter_metrics.jsonl").read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def tail_early(recs: list[dict], key: str) -> tuple[float, float]:
    n = len(recs)
    if n == 0:
        return float("nan"), float("nan")
    st = max(0, n - max(1, int(n * TAIL_FRAC)))
    ee = max(1, int(n * TAIL_FRAC))

    def mean_seg(seg, k):
        v = [r.get(k) for r in seg if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    return mean_seg(recs[:ee], key), mean_seg(recs[st:], key)


def tail_slice(recs: list[dict]) -> list[dict]:
    n = len(recs)
    if n == 0:
        return []
    st = max(0, n - max(1, int(n * TAIL_FRAC)))
    return recs[st:]


def tail_mean_ratio_per_chain(recs: list[dict], num_key: str, den_key: str, eps: float = 1e-30) -> float:
    """Mean of num/den over tail records in one chain."""
    tail = tail_slice(recs)
    vals: list[float] = []
    for r in tail:
        num = r.get(num_key)
        den = r.get(den_key)
        if num is None or den is None:
            continue
        try:
            nf, df = float(num), float(den)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(nf) or not math.isfinite(df) or abs(df) < eps:
            continue
        vals.append(nf / df)
    return float(np.mean(vals)) if vals else float("nan")


def tail_mean_abs_per_chain(recs: list[dict], key: str) -> float:
    tail = tail_slice(recs)
    vals = [abs(float(r[key])) for r in tail if r.get(key) is not None and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def agg_across_chains(iters: list[list[dict]], fn) -> tuple[float, float]:
    """Apply fn(recs)->float per chain; return (mean, std across chains)."""
    out = [fn(recs) for recs in iters]
    out = [x for x in out if math.isfinite(x)]
    if not out:
        return float("nan"), float("nan")
    return float(np.mean(out)), float(np.std(out)) if len(out) > 1 else 0.0


def main() -> None:
    groups: dict[tuple[float, float], list[Path]] = {}
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if not m:
            continue
        h, a = float(m.group("h")), float(m.group("a"))
        groups.setdefault((h, a), []).append(d)

    lines: list[str] = [
        "# h × α sweep summary (T=20000, 4 chains, β=1)",
        "",
        "Runs: `w1_n512_h{h}_T20000_a{α}_b1p0_chain{0..3}`. B=0, S=100, log_every=20.",
        "Tail = last 10% of iter_metrics; early = first 10%.",
        "",
        "### Reading the extended diagnostics",
        "",
        "- **U_train** = logged total potential (β-scaled in your runs; here β=1). **U_prior** ≈ β·(α/2)||θ||², **U_data** ≈ β·CE so **U_train ≈ U_prior + U_data**.",
        "- **prior_ratio** = U_prior/U_train — fraction of total energy from the prior term.",
        "- **data_ratio** = U_data/U_train — fraction from the data (NLL) term.",
        "- **data_prior_ratio** = U_data/U_prior — how large data term is vs prior (high ⇒ likelihood dominates that decomposition).",
        "- **|U_data_minus_ce|** should be ~0 if ce_mean_train tracks U_data/β consistently.",
        "- **nll_probe_mean** vs **ce_mean_train**: both probe CE; mismatch suggests batch/probe inconsistency.",
        "- **logit_max_abs**, **logsumexp_max**: logit scale; explosion can worsen CE.",
        "- **drift_step_norm** vs **noise_step_norm**: large drift/noise ratio aligns with SNR and non-diffusive behaviour.",
        "",
    ]

    for (h, a) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        dirs = sorted(groups[(h, a)])
        lines.append(f"## h = {h:g}, α = {a:g} (n_chains={len(dirs)})")
        lines.append("")
        iters = [load_iter(d) for d in dirs]
        finite_all = True
        abort_any = False
        max_g = 0.0
        for recs in iters:
            for r in recs:
                if r.get("finite_params") is False or r.get("finite_grad") is False:
                    finite_all = False
                if r.get("abort_suggested"):
                    abort_any = True
                g = r.get("grad_norm")
                if g is not None and math.isfinite(g):
                    max_g = max(max_g, float(g))
        lines.append(f"- finite_params & finite_grad always True: **{finite_all}**")
        lines.append(f"- any abort_suggested: **{abort_any}**")
        lines.append(f"- max grad_norm: **{max_g:.4g}**")
        lines.append("")
        primary = ["f_nll", "ce_mean_train", "margin_probe", "pmax_mean"]
        lines.append("### Primary probes (early → tail, mean across chains)")
        lines.append("")
        lines.append("| probe | early_mean | tail_mean | tail_std |")
        lines.append("|-------|------------|-----------|----------|")
        for key in primary:
            tails, earlies = [], []
            for recs in iters:
                e, t = tail_early(recs, key)
                if math.isfinite(t):
                    tails.append(t)
                if math.isfinite(e):
                    earlies.append(e)
            tm = float(np.mean(tails)) if tails else float("nan")
            ts = float(np.std(tails)) if len(tails) > 1 else 0.0
            em = float(np.mean(earlies)) if earlies else float("nan")
            label = f"{key}/β (β=1)" if key in ("f_nll", "ce_mean_train") else key
            lines.append(f"| {label} | {em:.4f} | {tm:.4f} | {ts:.4f} |")
        lines.append("")
        sec = ["dist_to_ref", "theta_norm", "grad_norm", "snr"]
        lines.append("### Secondary (tail)")
        lines.append("")
        for key in sec:
            tails = []
            for recs in iters:
                _, t = tail_early(recs, key)
                if math.isfinite(t):
                    tails.append(t)
            if tails:
                lines.append(
                    f"- {key}: mean={float(np.mean(tails)):.4g}, "
                    f"std_across_chains={float(np.std(tails)) if len(tails) > 1 else 0:.4g}"
                )
        lines.append("")
        # --- Extended: U decomposition & ratios (explain f_nll / CE growth) ---
        lines.append("### Energy decomposition & prior ratios (tail, mean across chains)")
        lines.append("")
        for label, key in [
            ("U_train", "U_train"),
            ("U_prior", "U_prior"),
            ("U_data", "U_data"),
        ]:
            tails = []
            for recs in iters:
                _, t = tail_early(recs, key)
                if math.isfinite(t):
                    tails.append(t)
            if tails:
                lines.append(
                    f"- **{label}**: mean={float(np.mean(tails)):.4g}, "
                    f"std_across_chains={float(np.std(tails)) if len(tails) > 1 else 0:.4g}"
                )
        pr_m, pr_s = agg_across_chains(iters, lambda r: tail_mean_ratio_per_chain(r, "U_prior", "U_train"))
        dr_m, dr_s = agg_across_chains(iters, lambda r: tail_mean_ratio_per_chain(r, "U_data", "U_train"))
        dpr_m, dpr_s = agg_across_chains(iters, lambda r: tail_mean_ratio_per_chain(r, "U_data", "U_prior"))
        lines.append(
            f"- **prior_ratio** (U_prior/U_train, tail): mean={pr_m:.4f}, std_across_chains={pr_s:.4f}"
        )
        lines.append(
            f"- **data_ratio** (U_data/U_train, tail): mean={dr_m:.4f}, std_across_chains={dr_s:.4f}"
        )
        lines.append(
            f"- **data_prior_ratio** (U_data/U_prior, tail): mean={dpr_m:.4f}, std_across_chains={dpr_s:.4f}"
        )

        def udm_tail_mean(recs: list[dict]) -> float:
            tail = tail_slice(recs)
            v = [abs(float(r["U_data_minus_ce"])) for r in tail if r.get("U_data_minus_ce") is not None]
            return float(np.mean(v)) if v else float("nan")

        udm_m, udm_s = agg_across_chains(iters, udm_tail_mean)
        lines.append(
            f"- **|U_data_minus_ce|** (tail mean per chain): mean={udm_m:.4g}, std_across_chains={udm_s:.4g}"
        )
        lines.append("")
        lines.append("### Probe / logit scale (tail)")
        lines.append("")
        for key in ["nll_probe_mean", "logit_max_abs", "logsumexp_max"]:
            tails = []
            for recs in iters:
                _, t = tail_early(recs, key)
                if math.isfinite(t):
                    tails.append(t)
            if tails:
                lines.append(
                    f"- **{key}**: mean={float(np.mean(tails)):.4g}, "
                    f"std_across_chains={float(np.std(tails)) if len(tails) > 1 else 0:.4g}"
                )
        lines.append("")
        lines.append("### Step geometry: drift vs noise (tail)")
        lines.append("")
        for key in ["drift_step_norm", "noise_step_norm", "delta_theta_norm"]:
            tails = []
            for recs in iters:
                _, t = tail_early(recs, key)
                if math.isfinite(t):
                    tails.append(t)
            if tails:
                lines.append(
                    f"- **{key}**: mean={float(np.mean(tails)):.4g}, "
                    f"std_across_chains={float(np.std(tails)) if len(tails) > 1 else 0:.4g}"
                )
        du_m, du_s = agg_across_chains(iters, lambda r: tail_mean_abs_per_chain(r, "delta_U"))
        lines.append(
            f"- **|delta_U|** (tail mean abs per chain): mean={du_m:.4g}, std_across_chains={du_s:.4g}"
        )
        # drift/noise ratio in tail
        def drift_over_noise(recs):
            tail = tail_slice(recs)
            ratios = []
            for r in tail:
                d = r.get("drift_step_norm")
                n = r.get("noise_step_norm")
                if d is None or n is None:
                    continue
                d, n = float(d), float(n)
                if math.isfinite(d) and math.isfinite(n) and n > 1e-30:
                    ratios.append(d / n)
            return float(np.mean(ratios)) if ratios else float("nan")

        r_m, r_s = agg_across_chains(iters, drift_over_noise)
        if math.isfinite(r_m):
            lines.append(
                f"- **drift_step_norm / noise_step_norm** (tail): mean={r_m:.4f}, std_across_chains={r_s:.4f}"
            )
        lines.append("")
        traces_fnll, traces_fm = [], []
        for d in dirs:
            sm = d / "samples_metrics.npz"
            if not sm.exists():
                continue
            z = np.load(sm)
            if "f_nll" in z:
                traces_fnll.append(z["f_nll"])
            if "f_margin" in z:
                traces_fm.append(z["f_margin"])
        lines.append("### Convergence (samples_metrics)")
        lines.append("")
        lines.append("| probe | R̂ | ESS_bulk | n_samples |")
        lines.append("|-------|-----|----------|-----------|")
        for name, chains in [("f_nll", traces_fnll), ("f_margin", traces_fm)]:
            if not chains:
                lines.append(f"| {name} | — | — | — |")
                continue
            n_min = min(len(c) for c in chains)
            arr = np.stack([c[:n_min] for c in chains], axis=0)
            rh = split_rhat(arr)
            ess = float(np.nanmean([ess_bulk(c[:n_min]) for c in chains]))
            lines.append(f"| {name} | {rh:.4f} | {ess:.1f} | {n_min} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Compact comparison (tail means across chains)")
    lines.append("")
    lines.append(
        "| h | α | f_nll (tail) | ce_mean_train (tail) | margin | pmax | "
        "R̂(f_nll) | R̂(f_margin) | grad_norm (tail) | snr (tail) |"
    )
    lines.append("|---|---|-----|-----|-----|-----|-----|-----|-----|-----|")

    def agg_tail(iters, k):
        v = []
        for recs in iters:
            _, t = tail_early(recs, k)
            if math.isfinite(t):
                v.append(t)
        return float(np.mean(v)) if v else float("nan")

    def rh(chains):
        if not chains:
            return float("nan")
        n_min = min(len(c) for c in chains)
        arr = np.stack([c[:n_min] for c in chains], axis=0)
        return split_rhat(arr)

    for (h, a) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        dirs = sorted(groups[(h, a)])
        iters = [load_iter(d) for d in dirs]
        traces_fnll, traces_fm = [], []
        for d in dirs:
            sm = d / "samples_metrics.npz"
            if not sm.exists():
                continue
            z = np.load(sm)
            if "f_nll" in z:
                traces_fnll.append(z["f_nll"])
            if "f_margin" in z:
                traces_fm.append(z["f_margin"])
        lines.append(
            f"| {h:.1e} | {a:.0f} | {agg_tail(iters, 'f_nll'):.4f} | {agg_tail(iters, 'ce_mean_train'):.4f} | "
            f"{agg_tail(iters, 'margin_probe'):.4f} | {agg_tail(iters, 'pmax_mean'):.4f} | "
            f"{rh(traces_fnll):.3f} | {rh(traces_fm):.3f} | {agg_tail(iters, 'grad_norm'):.3g} | {agg_tail(iters, 'snr'):.3g} |"
        )

    lines.append("")
    lines.append("## Extended metrics comparison (tail, mean across chains)")
    lines.append("")
    lines.append(
        "| h | α | U_train | U_data | U_prior | data_prior_ratio | prior_ratio | "
        "nll_probe_mean | logit_max_abs | drift/noise | |ΔU| (tail) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def pr(iters, num_k, den_k):
        m, _ = agg_across_chains(iters, lambda r: tail_mean_ratio_per_chain(r, num_k, den_k))
        return m

    for (h, a) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        dirs = sorted(groups[(h, a)])
        iters = [load_iter(d) for d in dirs]
        ut = agg_tail(iters, "U_train")
        ud = agg_tail(iters, "U_data")
        up = agg_tail(iters, "U_prior")
        dpr = pr(iters, "U_data", "U_prior")
        prior_r = pr(iters, "U_prior", "U_train")
        npm = agg_tail(iters, "nll_probe_mean")
        lma = agg_tail(iters, "logit_max_abs")

        def drift_over_noise(recs):
            tail = tail_slice(recs)
            ratios = []
            for r in tail:
                d = r.get("drift_step_norm")
                n = r.get("noise_step_norm")
                if d is None or n is None:
                    continue
                d, n = float(d), float(n)
                if math.isfinite(d) and math.isfinite(n) and n > 1e-30:
                    ratios.append(d / n)
            return float(np.mean(ratios)) if ratios else float("nan")

        dn_m, _ = agg_across_chains(iters, drift_over_noise)
        dabs_m, _ = agg_across_chains(iters, lambda r: tail_mean_abs_per_chain(r, "delta_U"))
        lines.append(
            f"| {h:.1e} | {a:.0f} | {ut:.4g} | {ud:.4g} | {up:.4g} | {dpr:.3f} | {prior_r:.3f} | "
            f"{npm:.4f} | {lma:.2f} | {dn_m:.3f} | {dabs_m:.4g} |"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
