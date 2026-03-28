"""
Proxy LSI: rho_hat_f = E[||∇f||^2] / Var(f). Output: summaries/lsi_proxy.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_lsi_proxy(
    run_dirs: list[Path],
    B: int,
    G: int,
    S: int,
    probe_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pool post-burn-in samples across chains. For each probe with grad_norm_sq:
    rho_hat = mean(grad_norm_sq) / var(f_values).
    Returns dataframe with probe, rho_hat_mean, rho_hat_std, rho_hat_se (std/sqrt(K)).
    """
    if probe_names is None:
        probe_names = ["f_nll", "f_margin", "f_pc1", "f_proj1", "f_dist"]
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        post_burn = steps > B
        for p in probe_names:
            fkey = p
            gkey = f"grad_norm_sq__{p}"
            if fkey not in data or gkey not in data:
                continue
            f_vals = data[fkey][post_burn]
            g_vals = data[gkey]
            # grad_norm_sq is subsampled (every G saved); align by index
            n_saved = np.sum(post_burn)
            n_grad = len(g_vals)
            if n_grad == 0 or n_saved == 0:
                continue
            # Use same-length slice: f_vals for all post_burn, g_vals every G
            idx = np.arange(n_saved)
            grad_idx = idx[idx % G == 0][:n_grad]
            if len(grad_idx) == 0:
                continue
            f_sub = f_vals[grad_idx]
            var_f = np.var(f_sub)
            if var_f <= 0:
                continue
            rho = np.mean(g_vals[: len(grad_idx)]) / var_f
            rows.append({"run_dir": str(run_dir), "probe": p, "rho_hat": rho})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = df.groupby("probe").agg(rho_hat_mean=("rho_hat", "mean"), rho_hat_std=("rho_hat", "std")).reset_index()
    K = df["run_dir"].nunique()
    summary["rho_hat_se"] = summary["rho_hat_std"] / np.sqrt(K) if K > 0 else np.nan
    return summary


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="Run dirs (one per chain)")
    p.add_argument("--exclude-chain", type=int, action="append", default=None, metavar="N",
                   help="Exclude run dirs containing 'chainN' (e.g. --exclude-chain 2)")
    p.add_argument("--B", type=int, default=50_000)
    p.add_argument("--G", type=int, default=5)
    p.add_argument("--S", type=int, default=200)
    p.add_argument("-o", "--out", default="experiments/summaries/lsi_proxy.csv")
    args = p.parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    if args.exclude_chain:
        exclude = set(args.exclude_chain)
        run_dirs = [d for d in run_dirs if not any(f"chain{n}" in str(d) for n in exclude)]
    df = compute_lsi_proxy(
        run_dirs, args.B, args.G, args.S
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
