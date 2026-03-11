#!/usr/bin/env python3
"""
Analyze 4-chain results for a given width.
Convergence (Rhat, ESS, ESS/1e6), proxy LSI (ρ̂), U_train/U_data/dist_to_ref included.
Writes experiments/summaries/convergence_w{width}.csv and lsi_proxy_w{width}.csv.

Usage:
  python scripts/analyze_chains.py --width 0.1
  python scripts/analyze_chains.py --width 0.1 --runs-dir experiments/runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import yaml
except ImportError:
    yaml = None


def load_iter_metrics(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = np.nan
        records.append(rec)
    return records


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
        max_lag = min(n // 2, 500)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :]
    ac = ac[: max_lag + 1] / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze chains for a given width")
    p.add_argument("--width", type=float, default=0.1, help="Width multiplier (e.g. 0.1, 1.0)")
    p.add_argument("--runs-dir", type=str, default="experiments/runs", help="Parent of run dirs")
    p.add_argument("--summaries-dir", type=str, default="experiments/summaries", help="Output for CSVs")
    args = p.parse_args()

    base = Path(args.runs_dir)
    summaries_dir = Path(args.summaries_dir)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    w_str = int(args.width) if args.width == int(args.width) else args.width
    pattern = f"w{w_str}_n*_h*_chain*"
    dirs = sorted(base.glob(pattern))
    if len(dirs) < 2:
        print(f"No or too few chains found for width={args.width} (pattern {pattern})")
        return

    # Load B, S, T from first run's config
    B, S, T = 50_000, 2_000, 200_000
    grad_norm_stride = 5
    for d in dirs:
        cfg_path = d / "run_config.yaml"
        if cfg_path.exists() and yaml is not None:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            B = int(cfg.get("B", B))
            S = int(cfg.get("S", S))
            T = int(cfg.get("T", T))
            grad_norm_stride = int(cfg.get("grad_norm_stride", 5))
            break

    T_post = T - B
    all_recs = {d.name: load_iter_metrics(d / "iter_metrics.jsonl") for d in dirs}

    # --- Probes: from samples_metrics and iter_metrics ---
    sample_probes = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
    iter_probes = ["nll_probe_mean", "U_train", "U_data", "dist_to_ref"]

    # Convergence from samples_metrics
    traces = {p: [] for p in sample_probes}
    for d in dirs:
        path = d / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        for p in sample_probes:
            if p in data:
                post = data["step"] > B
                traces[p].append(data[p][post])

    n_post_samples = min(len(t) for t in traces["f_nll"]) if traces["f_nll"] else 0

    # Convergence from iter_metrics (U_train, U_data, nll_probe_mean, dist_to_ref)
    consolidated = {}
    for p in iter_probes:
        chain_traces = []
        for d in dirs:
            recs = [r for r in all_recs[d.name] if r.get("step", 0) > B and r.get(p) is not None]
            recs = sorted(recs, key=lambda r: r["step"])
            vals = np.array([r[p] for r in recs])
            vals = vals[np.isfinite(vals)]
            chain_traces.append(vals)
        if not chain_traces:
            continue
        n_min = min(len(t) for t in chain_traces)
        if n_min < 4:
            continue
        arr = np.array([t[:n_min] for t in chain_traces])
        ess = np.nanmean([_ess_bulk(t[:n_min]) for t in chain_traces])
        ess_per_1e6 = ess * 1e6 / T_post if T_post > 0 else float("nan")
        consolidated[p] = {"rhat": _split_rhat(arr), "ess": ess, "ess_per_1e6": ess_per_1e6}

    # Add sample_probes to consolidated for output
    for p in sample_probes:
        if not traces[p]:
            continue
        n = min(len(t) for t in traces[p])
        if n < 4:
            continue
        arr = np.array([t[:n] for t in traces[p]])
        ess = np.nanmean([_ess_bulk(t[:n]) for t in traces[p]])
        n_saved = n
        T_post_steps = n_saved * S
        ess_per_1e6 = ess * 1e6 / T_post_steps if T_post_steps > 0 else float("nan")
        consolidated[p] = {"rhat": _split_rhat(arr), "ess": ess, "ess_per_1e6": ess_per_1e6}

    # Write convergence CSV (all probes: iter + sample)
    conv_rows = []
    for metric in iter_probes + sample_probes:
        if metric not in consolidated:
            continue
        c = consolidated[metric]
        conv_rows.append({
            "width": args.width,
            "metric": metric,
            "rhat": c["rhat"],
            "ess": c["ess"],
            "ess_per_1e6_steps": c["ess_per_1e6"],
        })
    conv_path = summaries_dir / f"convergence_w{w_str}.csv"
    if conv_rows:
        with open(conv_path, "w") as f:
            f.write("width,metric,rhat,ess,ess_per_1e6_steps\n")
            for r in conv_rows:
                f.write(f"{r['width']},{r['metric']},{r['rhat']:.6f},{r['ess']:.4f},{r['ess_per_1e6_steps']:.4f}\n")
        print("Wrote", conv_path)

    # --- Proxy LSI: samples_metrics probes + U_train, U_data from iter_metrics ---
    LSI_SAMPLE_PROBES = ["f_nll", "f_margin", "f_pc1", "f_proj1", "f_dist"]
    lsi_rows = []

    for run_dir in dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        post_burn = steps > B
        n_saved = int(np.sum(post_burn))
        for p in LSI_SAMPLE_PROBES:
            fkey, gkey = p, f"grad_norm_sq__{p}"
            if fkey not in data or gkey not in data:
                continue
            f_vals = data[fkey][post_burn]
            g_vals = data[gkey]
            if len(g_vals) == 0 or n_saved == 0:
                continue
            idx = np.arange(n_saved)
            grad_idx = idx[idx % grad_norm_stride == 0][: len(g_vals)]
            if len(grad_idx) < 2:
                continue
            f_sub = f_vals[grad_idx]
            var_f = np.var(f_sub)
            if var_f <= 0:
                continue
            rho = float(np.mean(g_vals[: len(grad_idx)]) / var_f)
            lsi_rows.append({"width": args.width, "run_dir": str(run_dir), "probe": p, "rho_hat": rho})

    # LSI for U_train, U_data from iter_metrics: rho = E[grad_norm²] / Var(metric)
    for run_dir in dirs:
        recs = [r for r in all_recs[run_dir.name] if r.get("step", 0) > B]
        recs = sorted(recs, key=lambda r: r["step"])
        if len(recs) < 2:
            continue
        for p in ["U_train", "U_data"]:
            grad_sq = []
            vals = []
            for r in recs:
                g, v = r.get("grad_norm"), r.get(p)
                if g is not None and v is not None and np.isfinite(g) and np.isfinite(v):
                    grad_sq.append(g ** 2)
                    vals.append(v)
            if len(vals) < 2:
                continue
            var_f = np.var(vals)
            if var_f <= 0:
                continue
            rho = float(np.mean(grad_sq) / var_f)
            lsi_rows.append({"width": args.width, "run_dir": str(run_dir), "probe": p, "rho_hat": rho})

    # dist_to_ref: we don't have ||∇(dist_to_ref)|| in logs; f_dist = dist_to_ref² and we have grad_norm_sq for f_dist.
    # So LSI for dist_to_ref could be derived from f_dist. Skip or add as optional. Omit for simplicity.

    lsi_path = summaries_dir / f"lsi_proxy_w{w_str}.csv"
    if lsi_rows:
        by_probe = {}
        for row in lsi_rows:
            key = row["probe"]
            if key not in by_probe:
                by_probe[key] = []
            by_probe[key].append(row["rho_hat"])
        with open(lsi_path, "w") as f:
            f.write("width,probe,rho_hat_mean,rho_hat_std\n")
            for probe in LSI_SAMPLE_PROBES + ["U_train", "U_data"]:
                if probe not in by_probe:
                    continue
                vals = np.array(by_probe[probe])
                f.write(f"{args.width},{probe},{np.mean(vals):.6e},{np.std(vals):.6e}\n")
        print("Wrote", lsi_path)

    # --- Console: consolidated Rhat & ESS (nll, U_train, U_data) ---
    print("\n" + "=" * 60)
    print(f"CONSOLIDATED (width={args.width}): Rhat & ESS — nll_probe_mean, U_train, U_data")
    print("=" * 60)
    print(f"\n{'Metric':<18} {'Rhat':>10} {'ESS':>10} {'ESS/1e6 steps':>14}")
    print("-" * 56)
    for key in ["nll_probe_mean", "U_train", "U_data"]:
        if key in consolidated:
            c = consolidated[key]
            print(f"{key:<18} {c['rhat']:>10.4f} {c['ess']:>10.1f} {c['ess_per_1e6']:>14.1f}")
    print()

    # --- Console: LSI (all probes including U_train, U_data) ---
    if lsi_rows:
        print("Proxy LSI ρ̂ = E[||∇f||²]/Var(f)")
        print("-" * 60)
        by_probe = {}
        for row in lsi_rows:
            p = row["probe"]
            if p not in by_probe:
                by_probe[p] = []
            by_probe[p].append(row["rho_hat"])
        for probe in LSI_SAMPLE_PROBES + ["U_train", "U_data"]:
            if probe in by_probe:
                v = np.array(by_probe[probe])
                print(f"  {probe:<12} mean={np.mean(v):.4e}  std={np.std(v):.4e}")
    print()


if __name__ == "__main__":
    main()
