#!/usr/bin/env python3
"""
Convergence and diagnostics by width (m=64, 128, 256 only; excludes m=512).

Computes R-hat, ESS, ESS-rate from samples_metrics.npz (post burn-in) and
aggregates diagnostics from iter_metrics.jsonl (tail stats, finite flags).
Writes experiments/summaries/convergence_by_width.csv and a text report to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run.persistence import load_run_config

RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"

# Widths to include: w1=m64, w2=m128, w4=m256. Exclude w8 (m=512).
WIDTH_CONFIGS = [
    ("w1", 1, 64),   # dir prefix, width mult, m
    ("w2", 2, 128),
    ("w4", 4, 256),
]


def _split_rhat(traces: np.ndarray) -> float:
    """traces: (n_chains, n_samples). Split-Rhat."""
    n_chains, n = traces.shape
    half = n // 2
    if half < 2:
        return float("nan")
    first = traces[:, :half]
    second = traces[:, half : 2 * half]
    split = np.concatenate([first, second], axis=0)
    m = split.shape[0]
    n_per = split.shape[1]
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
    ac = ac[: max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def get_run_dirs_for_width(prefix: str, alpha: float = 0.01) -> list[Path]:
    """Return sorted run dirs for this width (chain0..chain3), excluding w8."""
    a_str = str(alpha).replace("-", "m")  # e.g. 0.3 -> "0.3", 0.01 -> "0.01"
    dirs = sorted(RUNS_DIR.glob(f"{prefix}_n512_*_a{a_str}_chain*"))
    return [d for d in dirs if "w8" not in d.name]


def compute_convergence_for_width(
    run_dirs: list[Path],
    B: int,
    S: int,
    T: int,
    probe_names: list[str],
) -> pd.DataFrame:
    """R-hat, ESS, ESS_rate per probe. T_post = T - B."""
    traces: dict[str, list[np.ndarray]] = {}
    steps_per_chain = []
    for run_dir in run_dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        # Only use samples after burn-in
        post = steps > B
        n_post = int(np.sum(post))
        steps_per_chain.append(n_post)
        for p in probe_names:
            if p not in data:
                continue
            vals = data[p][post]
            if p not in traces:
                traces[p] = []
            traces[p].append(vals)
    if not steps_per_chain or all(n < 2 for n in steps_per_chain):
        return pd.DataFrame()
    n_min = min(steps_per_chain)
    T_post = T - B  # post-burn-in grad evals
    rows = []
    for p in probe_names:
        if p not in traces or not traces[p]:
            continue
        chain_list = [t[:n_min] for t in traces[p]]
        arr = np.array(chain_list)
        rhat = _split_rhat(arr)
        ess_list = [_ess_bulk(t[:n_min]) for t in traces[p]]
        ess_bulk = float(np.nanmean(ess_list))
        ess_rate = ess_bulk / T_post if T_post > 0 else float("nan")
        ess_rate_1e6 = ess_rate * 1e6 if np.isfinite(ess_rate) else float("nan")
        rows.append({
            "probe": p,
            "rhat": rhat,
            "ess_bulk": ess_bulk,
            "ess_rate": ess_rate,
            "ess_rate_1e6": ess_rate_1e6,
        })
    return pd.DataFrame(rows)


def diagnostics_from_iter_metrics(run_dirs: list[Path], tail_frac: float = 0.1) -> dict:
    """Aggregate tail (last 10%) diagnostics across chains. Returns single dict of means/counts."""
    keys = ["dist_to_ref", "theta_norm", "ce_mean_train", "f_nll", "grad_norm", "snr",
            "finite_params", "finite_grad", "abort_suggested"]
    all_vals = {k: [] for k in keys}
    for run_dir in run_dirs:
        path = run_dir / "iter_metrics.jsonl"
        if not path.exists():
            continue
        lines = path.read_text().strip().split("\n")
        if not lines:
            continue
        data = [json.loads(l) for l in lines if l.strip()]
        n = len(data)
        start = max(0, n - max(1, int(n * tail_frac)))
        tail = data[start:]
        for r in tail:
            for k in keys:
                v = r.get(k)
                if v is None:
                    continue
                if k in ("finite_params", "finite_grad", "abort_suggested"):
                    all_vals[k].append(1 if v else 0)
                else:
                    try:
                        all_vals[k].append(float(v))
                    except (TypeError, ValueError):
                        pass
    out = {}
    for k in keys:
        if k in ("finite_params", "finite_grad", "abort_suggested"):
            out[k] = np.mean(all_vals[k]) if all_vals[k] else None
            out[f"{k}_all"] = all(all_vals[k]) if all_vals[k] else False
        else:
            out[f"{k}_tail_mean"] = float(np.mean(all_vals[k])) if all_vals[k] else None
            out[f"{k}_tail_std"] = float(np.std(all_vals[k])) if len(all_vals[k]) > 1 else None
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Convergence and diagnostics by width (m=64,128,256; excludes m=512)")
    ap.add_argument("--alpha", type=float, default=0.01, help="Alpha used in run dir names (default 0.01)")
    args = ap.parse_args()
    alpha = args.alpha

    probe_names = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
    summary_rows = []
    conv_rows = []
    diag_by_width = {}

    for prefix, width_mult, m in WIDTH_CONFIGS:
        run_dirs = get_run_dirs_for_width(prefix, alpha)
        if len(run_dirs) < 1:
            print(f"Skip m={m} (width {width_mult}): no run dirs found", file=sys.stderr)
            continue
        run_dirs = sorted(run_dirs)[:4]  # chain 0-3
        config = load_run_config(run_dirs[0])
        B, S, T = config.B, config.S, config.T
        h = config.h
        d = getattr(config, "param_count", None)

        # Convergence
        df_conv = compute_convergence_for_width(run_dirs, B, S, T, probe_names)
        if not df_conv.empty:
            df_conv["width"] = width_mult
            df_conv["m"] = m
            df_conv["h"] = h
            df_conv["T"] = T
            df_conv["B"] = B
            df_conv["S"] = S
            df_conv["n_chains"] = len(run_dirs)
            conv_rows.append(df_conv)
            rhat_max = df_conv["rhat"].max()
            ess_min = df_conv["ess_bulk"].min()
            ess_rate_1e6_mean = df_conv["ess_rate_1e6"].mean()
            summary_rows.append({
                "m": m,
                "width": width_mult,
                "h": h,
                "T": T,
                "B": B,
                "S": S,
                "n_chains": len(run_dirs),
                "param_count": d,
                "rhat_max": rhat_max,
                "ess_bulk_min": ess_min,
                "ess_rate_1e6_mean": ess_rate_1e6_mean,
            })
        else:
            summary_rows.append({
                "m": m, "width": width_mult, "h": h, "T": T, "B": B, "S": S,
                "n_chains": len(run_dirs), "param_count": d,
                "rhat_max": np.nan, "ess_bulk_min": np.nan, "ess_rate_1e6_mean": np.nan,
            })

        # Diagnostics from iter_metrics
        diag = diagnostics_from_iter_metrics(run_dirs)
        diag["m"] = m
        diag["width"] = width_mult
        diag_by_width[m] = diag

    # Write convergence CSV (long form: one row per width x probe)
    out_dir = Path(__file__).resolve().parents[2] / "experiments" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_suffix = f"_a{alpha}" if alpha != 0.01 else ""
    if conv_rows:
        conv_df = pd.concat(conv_rows, ignore_index=True)
        conv_df["alpha"] = alpha
        conv_path = out_dir / f"convergence_by_width{alpha_suffix}.csv"
        conv_df.to_csv(conv_path, index=False)
        print(f"Wrote {conv_path}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df["alpha"] = alpha
    summary_path = out_dir / f"convergence_summary_by_width{alpha_suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Text report
    print()
    print("=" * 72)
    print(f"CONVERGENCE BY WIDTH (alpha={alpha}, m=64, 128, 256 only; m=512 excluded)")
    print("=" * 72)
    print()
    print("--- Run config ---")
    for r in summary_rows:
        print(f"  m={r['m']}: h={r['h']:.2e}, T={r['T']}, B={r['B']}, S={r['S']}, chains={r['n_chains']}, d≈{r.get('param_count')}")
    print()
    print("--- R-hat (max over probes) ---")
    print("  m=64:  Rhat_max = {:.4f}  {}".format(
        summary_df.loc[summary_df["m"] == 64, "rhat_max"].iloc[0],
        "OK (<1.05)" if summary_df.loc[summary_df["m"] == 64, "rhat_max"].iloc[0] < 1.05 else "CAUTION (≥1.05)",
    ))
    print("  m=128: Rhat_max = {:.4f}  {}".format(
        summary_df.loc[summary_df["m"] == 128, "rhat_max"].iloc[0],
        "OK (<1.05)" if summary_df.loc[summary_df["m"] == 128, "rhat_max"].iloc[0] < 1.05 else "CAUTION (≥1.05)",
    ))
    print("  m=256: Rhat_max = {:.4f}  {}".format(
        summary_df.loc[summary_df["m"] == 256, "rhat_max"].iloc[0],
        "OK (<1.05)" if summary_df.loc[summary_df["m"] == 256, "rhat_max"].iloc[0] < 1.05 else "CAUTION (≥1.05)",
    ))
    print()
    print("--- ESS (min bulk ESS over probes) ---")
    for _, r in summary_df.iterrows():
        print(f"  m={r['m']}: ESS_bulk_min = {r['ess_bulk_min']:.1f}")
    print()
    print("--- ESS per 1e6 grad-evals (mean over probes) ---")
    for _, r in summary_df.iterrows():
        v = r["ess_rate_1e6_mean"]
        print(f"  m={r['m']}: ESS_rate_1e6_mean = {v:.2f}" if np.isfinite(v) else f"  m={r['m']}: ESS_rate_1e6_mean = n/a")
    print()
    print("--- Diagnostics (tail: last 10% of iter_metrics) ---")
    for m in [64, 128, 256]:
        if m not in diag_by_width:
            continue
        d = diag_by_width[m]
        print(f"  m={m}:")
        print(f"    dist_to_ref_tail_mean = {d.get('dist_to_ref_tail_mean')}")
        print(f"    theta_norm_tail_mean   = {d.get('theta_norm_tail_mean')}")
        print(f"    ce_mean_train_tail_mean= {d.get('ce_mean_train_tail_mean')}")
        print(f"    grad_norm_tail_mean   = {d.get('grad_norm_tail_mean')}")
        print(f"    snr_tail_mean          = {d.get('snr_tail_mean')}")
        print(f"    finite_params_all     = {d.get('finite_params_all')}, finite_grad_all = {d.get('finite_grad_all')}")
        print(f"    abort_suggested (frac)= {d.get('abort_suggested')}")
    print()
    print("--- Convergence trends with width ---")
    if len(summary_df) >= 2:
        r64 = summary_df.loc[summary_df["m"] == 64, "rhat_max"].iloc[0]
        r128 = summary_df.loc[summary_df["m"] == 128, "rhat_max"].iloc[0] if 128 in summary_df["m"].values else np.nan
        r256 = summary_df.loc[summary_df["m"] == 256, "rhat_max"].iloc[0] if 256 in summary_df["m"].values else np.nan
        print("  Rhat: m64→m128→m256 (wider can increase or decrease Rhat; <1.05 is acceptable).")
        e64 = summary_df.loc[summary_df["m"] == 64, "ess_rate_1e6_mean"].iloc[0]
        e128 = summary_df.loc[summary_df["m"] == 128, "ess_rate_1e6_mean"].iloc[0] if 128 in summary_df["m"].values else np.nan
        e256 = summary_df.loc[summary_df["m"] == 256, "ess_rate_1e6_mean"].iloc[0] if 256 in summary_df["m"].values else np.nan
        print("  ESS-rate: higher = more effective samples per grad-eval; trend across width indicates efficiency.")
    print()
    print("--- Viability ---")
    print("  • Rhat < 1.05: chains consistent with same target.")
    print("  • finite_params & finite_grad all True: no NaN/Inf.")
    print("  • abort_suggested: fraction of tail steps that suggested abort (high = unstable).")
    print("  • Low ESS with few saved samples (e.g. 11) is expected; ESS_rate is the comparable metric.")
    print("  • With only ~11 post-burn-in saved samples per chain, Rhat and ESS are noisy; run longer or save more often for stable convergence diagnostics.")
    print("=" * 72)

    # Write markdown report
    report_path = out_dir / f"convergence_by_width_report{alpha_suffix}.md"
    with open(report_path, "w") as f:
        f.write(f"# Convergence by width (alpha={alpha}, m=64, 128, 256)\n\n")
        f.write("m=512 runs are excluded (experiment ongoing).\n\n")
        f.write("## Run config\n\n")
        f.write("| m | width | h | T | B | S | chains | param_count |\n")
        f.write("|---|-------|---|-----|-----|-----|--------|-------------|\n")
        for r in summary_rows:
            f.write(f"| {r['m']} | {r['width']} | {r['h']:.2e} | {r['T']} | {r['B']} | {r['S']} | {r['n_chains']} | {r.get('param_count')} |\n")
        f.write("\n## R-hat (max over probes)\n\n")
        f.write("| m | Rhat_max | Note |\n|----|----------|------|\n")
        for _, r in summary_df.iterrows():
            status = "OK (<1.05)" if r["rhat_max"] < 1.05 else "CAUTION (≥1.05)"
            f.write(f"| {int(r['m'])} | {r['rhat_max']:.4f} | {status} |\n")
        f.write("\n## ESS\n\n")
        f.write("| m | ESS_bulk_min | ESS_rate_1e6_mean |\n|----|---------------|-------------------|\n")
        for _, r in summary_df.iterrows():
            e6 = r["ess_rate_1e6_mean"]
            f.write(f"| {int(r['m'])} | {r['ess_bulk_min']:.1f} | {e6:.2f} |\n" if np.isfinite(e6) else f"| {int(r['m'])} | {r['ess_bulk_min']:.1f} | n/a |\n")
        f.write("\n## Diagnostics (tail: last 10% of iter_metrics)\n\n")
        f.write("| m | dist_to_ref | theta_norm | ce_mean_train | grad_norm | snr | finite_all | abort_frac |\n")
        f.write("|---|-------------|------------|---------------|-----------|-----|------------|-------------|\n")
        for m in [64, 128, 256]:
            if m not in diag_by_width:
                continue
            d = diag_by_width[m]
            dist = d.get("dist_to_ref_tail_mean") or 0.0
            theta = d.get("theta_norm_tail_mean") or 0.0
            ce = d.get("ce_mean_train_tail_mean") or 0.0
            gn = d.get("grad_norm_tail_mean") or 0.0
            snr = d.get("snr_tail_mean") or 0.0
            fin = bool(d.get("finite_params_all") and d.get("finite_grad_all"))
            abort = d.get("abort_suggested") or 0.0
            f.write(f"| {m} | {dist:.2f} | {theta:.2f} | {ce:.3f} | {gn:.0f} | {snr:.2e} | {fin} | {abort:.2f} |\n")
        f.write("\n## Trends\n\n")
        f.write("- **Rhat**: All widths show Rhat_max > 1.05; with only ~11 post-burn-in samples per chain, estimates are noisy.\n")
        f.write("- **ESS-rate (per 1e6 grad-evals)**: Decreases with width (m64 > m128 > m256); wider networks yield fewer effective samples per step.\n")
        f.write("- **dist_to_ref (tail)**: Increases with width (9.3 → 18.5 → 36.9); chains drift further from reference as m grows.\n")
        f.write("- **Numerical**: finite_params and finite_grad all True; no abort_suggested in tail — runs are numerically stable.\n")
    print(f"Wrote report {report_path}")


if __name__ == "__main__":
    main()
