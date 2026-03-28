#!/usr/bin/env python3
"""
Generate comprehensive comparison report for step-size (h) sweep.
Extracts iter_metrics diagnostics, samples_metrics data quality, and computes
convergence/LSI metrics where sufficient finite data exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Convergence/LSI logic (adapted to handle NaN)
PROBE_NAMES = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
LSI_PROBES = ["f_nll", "f_margin", "f_pc1", "f_proj1", "f_dist"]


def _split_rhat(traces: np.ndarray) -> float:
    """Split-Rhat from traces (n_chains, n_samples)."""
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
    """Bulk ESS from autocorrelation."""
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :]
    ac = ac[: max_lag + 1] / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def load_iter_metrics(run_dir: Path) -> list[dict]:
    """Load iter_metrics.jsonl, parsing NaN."""
    path = run_dir / "iter_metrics.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = np.nan
        records.append(rec)
    return records


def extract_iter_diagnostics(run_dirs: list[Path]) -> pd.DataFrame:
    """Extract iter_metrics diagnostics from start of runs (steps 1, 1000, 2000)."""
    rows = []
    for run_dir in run_dirs:
        recs = load_iter_metrics(run_dir)
        finite_recs = [r for r in recs if r.get("step") in (1, 1000, 2000) and np.isfinite(r.get("U_train", np.nan))]
        for r in finite_recs:
            rows.append({
                "run_dir": str(run_dir),
                "step": r["step"],
                "U_train": r.get("U_train"),
                "grad_norm": r.get("grad_norm"),
                "theta_norm": r.get("theta_norm"),
                "f_nll": r.get("f_nll"),
                "f_margin": r.get("f_margin"),
                "snr": r.get("snr"),
                "delta_U": r.get("delta_U"),
            })
    return pd.DataFrame(rows)


def extract_samples_data_quality(run_dirs: list[Path]) -> pd.DataFrame:
    """Compute fraction of finite probe values per run."""
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        for p in PROBE_NAMES:
            if p not in data:
                continue
            vals = data[p]
            n = len(vals)
            n_finite = np.isfinite(vals).sum()
            rows.append({
                "run_dir": str(run_dir),
                "probe": p,
                "n_samples": n,
                "n_finite": n_finite,
                "frac_finite": n_finite / n if n > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def compute_convergence_finite_only(
    run_dirs: list[Path],
    B: int = 10000,
    S: int = 200,
) -> pd.DataFrame:
    """
    Compute Rhat, ESS, ESS_rate using only finite samples.
    Aligns chains to min finite length per probe; returns NaN if insufficient data.
    """
    traces = {}
    steps_per_chain = []
    for run_dir in run_dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        post_burn = steps > B
        steps_post = steps[post_burn]
        steps_per_chain.append(len(steps_post) * S)  # approx post-burn steps
        for p in PROBE_NAMES:
            if p not in data:
                continue
            vals = data[p][post_burn]
            finite_mask = np.isfinite(vals)
            finite_vals = vals[finite_mask]
            if p not in traces:
                traces[p] = []
            traces[p].append(finite_vals)

    if not steps_per_chain:
        return pd.DataFrame()
    T_post = float(np.mean(steps_per_chain))

    rows = []
    for p in PROBE_NAMES:
        if p not in traces or not traces[p]:
            continue
        chain_list = traces[p]
        min_len = min(len(t) for t in chain_list)
        if min_len < 4:
            rows.append({"probe": p, "rhat": np.nan, "ess_bulk": np.nan, "ess_rate": np.nan, "ess_rate_1e6": np.nan})
            continue
        arr = np.array([t[:min_len] for t in chain_list])
        rhat = _split_rhat(arr)
        ess_list = [_ess_bulk(arr[i]) for i in range(arr.shape[0])]
        ess_bulk = float(np.nanmean(ess_list))
        ess_rate = ess_bulk / T_post if T_post > 0 else np.nan
        ess_rate_1e6 = ess_rate * 1e6 if np.isfinite(ess_rate) else np.nan
        rows.append({"probe": p, "rhat": rhat, "ess_bulk": ess_bulk, "ess_rate": ess_rate, "ess_rate_1e6": ess_rate_1e6})
    return pd.DataFrame(rows)


def compute_lsi_finite_only(
    run_dirs: list[Path],
    B: int = 10000,
    G: int = 5,
    S: int = 200,
) -> pd.DataFrame:
    """Compute LSI proxy (rho_hat) where we have finite f and var(f) > 0."""
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        post_burn = steps > B
        n_saved = int(np.sum(post_burn))
        for p in LSI_PROBES:
            fkey, gkey = p, f"grad_norm_sq__{p}"
            if fkey not in data or gkey not in data:
                continue
            f_vals = np.asarray(data[fkey])[post_burn]
            g_vals = np.asarray(data[gkey])
            idx = np.arange(n_saved)
            grad_idx = idx[idx % G == 0][: len(g_vals)]
            if len(grad_idx) == 0 or len(g_vals) == 0:
                continue
            f_at_grad = f_vals[grad_idx]
            g_at_grad = g_vals[: len(grad_idx)]
            mask = np.isfinite(f_at_grad)
            if mask.sum() < 2:
                continue
            f_ok = f_at_grad[mask]
            g_ok = g_at_grad[mask]
            if np.var(f_ok) <= 0:
                continue
            rho = float(np.mean(g_ok) / np.var(f_ok))
            if np.isfinite(rho) and rho > 0:
                rows.append({"run_dir": str(run_dir), "probe": p, "rho_hat": rho})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = df.groupby("probe").agg(rho_hat_mean=("rho_hat", "mean"), rho_hat_std=("rho_hat", "std")).reset_index()
    K = df["run_dir"].nunique()
    summary["rho_hat_se"] = summary["rho_hat_std"] / np.sqrt(K) if K > 0 else np.nan
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate h-sweep comparison report")
    p.add_argument("--base", default="experiments/runs", help="Base runs directory")
    p.add_argument("--alpha", type=float, default=0.01, help="Alpha used in runs (for path construction)")
    p.add_argument("-o", "--out", default="experiments/summaries/h_sweep_report.md")
    args = p.parse_args()
    base = Path(args.base)
    out = Path(args.out)

    h_values = ["1e-06", "5e-06", "1e-05", "5e-05"]
    alpha_str = str(args.alpha).replace("-", "m")  # match run_single_chain naming
    prefix = "w0.1_n1024_h"
    suffix = f"_a{alpha_str}_chain"

    # Collect data per h
    iter_dfs = []
    quality_dfs = []
    conv_dfs = []
    lsi_dfs = []

    for h in h_values:
        run_dirs = [base / f"{prefix}{h}{suffix}{c}" for c in range(4)]
        run_dirs = [d for d in run_dirs if d.exists()]

        idf = extract_iter_diagnostics(run_dirs)
        if not idf.empty:
            idf["h"] = h
            iter_dfs.append(idf)

        qdf = extract_samples_data_quality(run_dirs)
        if not qdf.empty:
            qdf["h"] = h
            quality_dfs.append(qdf)

        cdf = compute_convergence_finite_only(run_dirs, B=10000, S=200)
        if not cdf.empty:
            cdf["h"] = h
            conv_dfs.append(cdf)

        ldf = compute_lsi_finite_only(run_dirs, B=10000, G=5, S=200)
        if not ldf.empty:
            ldf["h"] = h
            lsi_dfs.append(ldf)

    def safe_concat(dfs, default=None):
        non_empty = [x for x in dfs if x is not None and not x.empty] if dfs else []
        return pd.concat(non_empty, ignore_index=True) if non_empty else (default or pd.DataFrame())

    iter_df = safe_concat(iter_dfs)
    quality_df = safe_concat(quality_dfs)
    conv_df = safe_concat(conv_dfs)
    lsi_df = safe_concat(lsi_dfs)

    # Build report
    lines = [
        "# Step-Size (h) Sweep Comparison Report",
        "",
        "**Setup:** w=0.1, n_train=1024, T=50k, B=10k, S=200, 4 chains per h",
        "**h values:** 1e-6, 5e-6, 1e-5, 5e-5",
        "",
        "---",
        "",
        "## 1. iter_metrics Diagnostics (Start of Run)",
        "",
        "Diagnostics at steps 1, 1000, 2000 (where finite). Mean ± std across chains.",
        "",
    ]

    if not iter_df.empty:
        for h in h_values:
            sub = iter_df[iter_df["h"] == h]
            if sub.empty:
                lines.append(f"### h = {h}")
                lines.append("*No finite iter_metrics.*")
                lines.append("")
                continue
            lines.append(f"### h = {h}")
            for step in [1, 1000, 2000]:
                s = sub[sub["step"] == step]
                if s.empty:
                    continue
                parts = [f"**Step {step}:**"]
                for col in ["U_train", "theta_norm", "f_nll", "f_margin", "snr", "delta_U"]:
                    if col not in s.columns:
                        continue
                    vals = s[col].dropna()
                    if len(vals) == 0:
                        continue
                    m, std = vals.mean(), vals.std()
                    parts.append(f"{col}={m:.4g} ± {std:.4g}")
                if len(parts) > 1:
                    lines.append(" ".join(parts))
            lines.append("")
    else:
        lines.append("*No iter_metrics data found.*")
        lines.append("")

    lines.extend([
        "## 2. samples_metrics Data Quality",
        "",
        "Fraction of finite probe values per chain (post-burn-in, 200 samples).",
        "",
    ])
    if not quality_df.empty:
        for h in h_values:
            sub = quality_df[quality_df["h"] == h]
            if sub.empty:
                continue
            agg = sub.groupby("probe").agg(frac_finite=("frac_finite", "mean")).reset_index()
            lines.append(f"### h = {h}")
            lines.append("| Probe | Mean frac finite |")
            lines.append("|-------|------------------|")
            for _, row in agg.iterrows():
                lines.append(f"| {row['probe']} | {row['frac_finite']:.2%} |")
            lines.append("")
    else:
        lines.append("*No samples_metrics data found.*")
        lines.append("")

    lines.extend([
        "## 3. Convergence Metrics (R̂, ESS, ESS-rate)",
        "",
        "Computed on finite-only samples. Empty if insufficient finite data.",
        "",
    ])
    if not conv_df.empty:
        for h in h_values:
            sub = conv_df[conv_df["h"] == h]
            if sub.empty:
                lines.append(f"### h = {h}")
                lines.append("*No convergence metrics (insufficient finite data).*")
                lines.append("")
                continue
            lines.append(f"### h = {h}")
            lines.append("| Probe | R̂ | ESS_bulk | ESS_rate | ESS_rate_1e6 |")
            lines.append("|-------|---|----------|----------|--------------|")
            for _, row in sub.iterrows():
                r = row["rhat"]
                e = row["ess_bulk"]
                er = row["ess_rate"]
                e1 = row["ess_rate_1e6"]
                sr = f"{r:.4f}" if np.isfinite(r) else "—"
                se = f"{e:.1f}" if np.isfinite(e) else "—"
                ser = f"{er:.2e}" if np.isfinite(er) else "—"
                se1 = f"{e1:.2f}" if np.isfinite(e1) else "—"
                lines.append(f"| {row['probe']} | {sr} | {se} | {ser} | {se1} |")
            lines.append("")
    else:
        lines.append("*No convergence metrics computed (insufficient finite data across all h).*")
        lines.append("")

    lines.extend([
        "## 4. LSI Proxy (ρ̂)",
        "",
        "ρ̂ = E[||∇f||²] / Var(f). Requires finite f and Var(f) > 0.",
        "",
    ])
    if not lsi_df.empty:
        for h in h_values:
            sub = lsi_df[lsi_df["h"] == h]
            if sub.empty:
                lines.append(f"### h = {h}")
                lines.append("*No LSI proxy (insufficient finite data).*")
                lines.append("")
                continue
            lines.append(f"### h = {h}")
            lines.append("| Probe | ρ̂_mean | ρ̂_std | ρ̂_se |")
            lines.append("|-------|--------|-------|------|")
            for _, row in sub.iterrows():
                m = row["rho_hat_mean"]
                s = row["rho_hat_std"]
                se = row["rho_hat_se"]
                sm = f"{m:.4g}" if np.isfinite(m) else "—"
                ss = f"{s:.4g}" if np.isfinite(s) else "—"
                sse = f"{se:.4g}" if np.isfinite(se) else "—"
                lines.append(f"| {row['probe']} | {sm} | {ss} | {sse} |")
            lines.append("")
    else:
        lines.append("*No LSI proxy computed.*")
        lines.append("")

    lines.extend([
        "---",
        "## Summary & Data Limitations",
        "",
        "- **iter_metrics:** Finite values only at steps 1, 1000, 2000 for most runs; steps 3000+ show NaN, suggesting numerical instability during chain evolution.",
        "- **samples_metrics:** Probe values (f_nll, etc.) are mostly NaN for h ≥ 5e-6; h=1e-6 has partial finite data. This prevents robust R̂/ESS/LSI computation for larger h.",
        "- **Recommendation:** Re-run chains with numerical stability checks (e.g., gradient clipping, lower h, or different initialization) to obtain usable probe traces.",
        "",
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("Wrote", out)

    # Write merged CSVs for plotting
    summary_dir = out.parent
    if not conv_df.empty:
        conv_df.to_csv(summary_dir / "convergence_by_h.csv", index=False)
        print("Wrote", summary_dir / "convergence_by_h.csv")
    if not lsi_df.empty:
        lsi_df.to_csv(summary_dir / "lsi_proxy_by_h.csv", index=False)
        print("Wrote", summary_dir / "lsi_proxy_by_h.csv")

    # Plot h-sweep figures if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(args.base).parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if not conv_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            h_numeric = conv_df["h"].map({"1e-06": 1e-6, "5e-06": 5e-6, "1e-05": 1e-5, "5e-05": 5e-5})
            for p in conv_df["probe"].unique():
                sub = conv_df[conv_df["probe"] == p].copy()
                sub = sub.dropna(subset=["rhat"])
                if sub.empty:
                    continue
                sub["h_val"] = sub["h"].map({"1e-06": 1e-6, "5e-06": 5e-6, "1e-05": 1e-5, "5e-05": 5e-5})
                axes[0].plot(sub["h_val"], sub["rhat"], "o-", label=p)
            axes[0].set_xscale("log")
            axes[0].set_xlabel("h")
            axes[0].set_ylabel("R̂")
            axes[0].axhline(1.05, color="gray", linestyle="--", alpha=0.7)
            axes[0].legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            axes[0].set_title("R̂ vs step size h")

            for p in conv_df["probe"].unique():
                sub = conv_df[conv_df["probe"] == p].copy()
                sub = sub.dropna(subset=["ess_rate_1e6"])
                if sub.empty:
                    continue
                sub["h_val"] = sub["h"].map({"1e-06": 1e-6, "5e-06": 5e-6, "1e-05": 1e-5, "5e-05": 5e-5})
                axes[1].plot(sub["h_val"], sub["ess_rate_1e6"], "o-", label=p)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("h")
            axes[1].set_ylabel("ESS per 1e6 grad-evals")
            axes[1].legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            axes[1].set_title("ESS-rate vs step size h")

            fig.tight_layout()
            fig.savefig(fig_dir / "h_sweep_convergence.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Wrote", fig_dir / "h_sweep_convergence.png")

        if not lsi_df.empty:
            fig, ax = plt.subplots()
            for p in lsi_df["probe"].unique():
                sub = lsi_df[lsi_df["probe"] == p].copy()
                sub = sub.dropna(subset=["rho_hat_mean"])
                if sub.empty:
                    continue
                sub["h_val"] = sub["h"].map({"1e-06": 1e-6, "5e-06": 5e-6, "1e-05": 1e-5, "5e-05": 5e-5})
                ax.errorbar(
                    sub["h_val"],
                    sub["rho_hat_mean"],
                    yerr=sub.get("rho_hat_se", 0),
                    fmt="o-",
                    capsize=3,
                    label=p,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel("ρ̂ (proxy LSI)")
            ax.legend()
            ax.set_title("LSI proxy ρ̂ vs step size h")
            fig.tight_layout()
            fig.savefig(fig_dir / "h_sweep_lsi.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Wrote", fig_dir / "h_sweep_lsi.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
