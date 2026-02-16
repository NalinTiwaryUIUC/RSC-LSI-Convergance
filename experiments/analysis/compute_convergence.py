"""
Convergence: split-Rhat, bulk ESS, ESS_rate, optional time-to-Rhat. Output: summaries/convergence.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _split_rhat(traces: np.ndarray) -> float:
    """
    traces: (n_chains, n_samples). Split each chain in half -> (2*n_chains, n_samples/2).
    Return Rhat (potential scale reduction factor).
    """
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
    """Bulk ESS from autocorrelation (single chain). ESS = n / (1 + 2*sum(autocorr))."""
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
    # Sum until first negative or all
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def compute_convergence(
    run_dirs: list[Path],
    B: int,
    S: int,
    probe_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    run_dirs: one per chain (same w, h). Read samples_metrics.npz from each.
    Returns dataframe with columns width, h, probe, rhat, ess_bulk, ess_rate, ess_rate_1e6, t_rhat_105 (optional).
    """
    if probe_names is None:
        probe_names = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
    traces = {}
    steps_per_chain = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        steps_per_chain.append(len(steps))
        for p in probe_names:
            key = p
            if key not in data:
                continue
            vals = data[key]
            if p not in traces:
                traces[p] = []
            traces[p].append(vals)
    if not steps_per_chain:
        return pd.DataFrame()
    n_post = min(steps_per_chain)  # use same length for Rhat
    T_post = (np.array(steps_per_chain) * S).mean()  # approximate post-burn-in steps
    rows = []
    for p in probe_names:
        if p not in traces or not traces[p]:
            continue
        # For Rhat: if single chain, split into 2 halves to get 2 "chains"
        chain_list = traces[p]
        if len(chain_list) == 1:
            t = chain_list[0][:n_post]
            half = len(t) // 2
            arr = np.array([t[:half], t[half : 2 * half]])
        else:
            arr = np.array([t[:n_post] for t in chain_list])
        rhat = _split_rhat(arr)
        ess_list = [_ess_bulk(t) for t in traces[p]]
        ess_bulk = np.nanmean(ess_list)
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


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="Run dirs (one per chain, same w,h)")
    p.add_argument("--B", type=int, default=50_000)
    p.add_argument("--S", type=int, default=200)
    p.add_argument("-o", "--out", default="experiments/summaries/convergence.csv")
    args = p.parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    df = compute_convergence(run_dirs, args.B, args.S)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
