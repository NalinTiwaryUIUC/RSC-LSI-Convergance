"""
Generate figures from summaries (C1, C2, L1, L2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FIG_DIR = Path("experiments/figures")


def plot_C1_ess_rate_vs_width(convergence_csv: Path, out_path: Path) -> None:
    """C1: ESS per 1e6 grad-evals vs width (curves per probe)."""
    if not convergence_csv.exists():
        return
    df = pd.read_csv(convergence_csv)
    if df.empty or "ess_rate_1e6" not in df.columns:
        return
    fig, ax = plt.subplots()
    probes = ["f_nll", "f_margin", "f_pc1"]
    for p in probes:
        sub = df[df["probe"] == p]
        if sub.empty:
            continue
        ax.plot(sub.index, sub["ess_rate_1e6"], "o-", label=p)
    ax.set_xlabel("(run index)")
    ax.set_ylabel("ESS per 1e6 grad-evals")
    ax.set_title("Convergence: ESS-rate vs width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_C2_time_to_rhat(convergence_csv: Path, out_path: Path) -> None:
    """C2: time-to-Rhat<=1.05 vs width (optional)."""
    if not convergence_csv.exists():
        return
    df = pd.read_csv(convergence_csv)
    if df.empty or "t_rhat_105" not in df.columns:
        return
    fig, ax = plt.subplots()
    for p in df["probe"].unique():
        sub = df[df["probe"] == p]
        ax.plot(sub.index, sub["t_rhat_105"], "o-", label=p)
    ax.set_xlabel("(run index)")
    ax.set_ylabel("grad-evals to Rhat <= 1.05")
    ax.set_title("Time to Rhat threshold vs width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_L1_rho_hat_vs_width(lsi_csv: Path, out_path: Path) -> None:
    """L1: rho_hat vs width (per probe, error bars)."""
    if not lsi_csv.exists():
        return
    try:
        df = pd.read_csv(lsi_csv)
    except pd.errors.EmptyDataError:
        return
    if df.empty or "rho_hat_mean" not in df.columns:
        return
    fig, ax = plt.subplots()
    x = range(len(df))
    ax.errorbar(
        x,
        df["rho_hat_mean"],
        yerr=df.get("rho_hat_se", 0),
        fmt="o-",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["probe"])
    ax.set_xlabel("probe")
    ax.set_ylabel("rho_hat (proxy LSI)")
    ax.set_title("Proxy LSI: rho_hat vs probe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_L2_rho_hat_min_vs_width(lsi_csv: Path, out_path: Path) -> None:
    """L2: rho_hat_min vs width."""
    if not lsi_csv.exists():
        return
    try:
        df = pd.read_csv(lsi_csv)
    except pd.errors.EmptyDataError:
        return
    if df.empty or "rho_hat_mean" not in df.columns:
        return
    rho_min = df["rho_hat_mean"].min()
    fig, ax = plt.subplots()
    ax.bar([0], [rho_min], label="min_f rho_hat")
    ax.set_ylabel("rho_hat_min")
    ax.set_xlabel("(summary)")
    ax.set_title("Proxy LSI: rho_hat_min")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--convergence-csv", default="experiments/summaries/convergence.csv")
    p.add_argument("--lsi-csv", default="experiments/summaries/lsi_proxy.csv")
    p.add_argument("-o", "--out-dir", default="experiments/figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_C1_ess_rate_vs_width(Path(args.convergence_csv), out_dir / "ess_rate_vs_width.png")
    plot_C2_time_to_rhat(Path(args.convergence_csv), out_dir / "time_to_rhat_vs_width.png")
    plot_L1_rho_hat_vs_width(Path(args.lsi_csv), out_dir / "rho_hat_vs_width.png")
    plot_L2_rho_hat_min_vs_width(Path(args.lsi_csv), out_dir / "rho_hat_min_vs_width.png")
    print("Figures written to", out_dir)


if __name__ == "__main__":
    main()
