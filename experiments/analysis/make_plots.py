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


def plot_C0_rhat_vs_width(convergence_csv: Path, out_path: Path) -> None:
    """R-hat vs width (curves per probe)."""
    if not convergence_csv.exists():
        return
    df = pd.read_csv(convergence_csv)
    if df.empty or "rhat" not in df.columns:
        return
    fig, ax = plt.subplots()
    if "width" in df.columns:
        probes = df["probe"].unique()
        for p in probes:
            sub = df[(df["probe"] == p)].sort_values("width")
            if sub.empty:
                continue
            ax.plot(sub["width"], sub["rhat"], "o-", label=p)
        ax.set_xlabel("width")
        ax.axhline(1.05, color="gray", linestyle="--", alpha=0.7, label="Rhat=1.05")
    else:
        x = range(len(df))
        ax.plot(x, df["rhat"], "o-")
        ax.set_xticks(x)
        ax.set_xticklabels(df["probe"])
        ax.set_xlabel("probe")
    ax.set_ylabel("R-hat")
    ax.set_title("R-hat vs width (w=1 vs w=0.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_C1_ess_rate_vs_width(convergence_csv: Path, out_path: Path) -> None:
    """C1: ESS per 1e6 grad-evals vs width (curves per probe)."""
    if not convergence_csv.exists():
        return
    df = pd.read_csv(convergence_csv)
    if df.empty or "ess_rate_1e6" not in df.columns:
        return
    fig, ax = plt.subplots()
    if "width" in df.columns:
        probes = ["f_nll", "f_margin", "f_pc1"] + [c for c in df["probe"].unique() if c not in ("f_nll", "f_margin", "f_pc1")]
        probes = list(dict.fromkeys(probes))
        for p in probes:
            sub = df[(df["probe"] == p)].sort_values("width")
            if sub.empty:
                continue
            ax.plot(sub["width"], sub["ess_rate_1e6"], "o-", label=p)
        ax.set_xlabel("width")
    else:
        probes = ["f_nll", "f_margin", "f_pc1"]
        for p in probes:
            sub = df[df["probe"] == p]
            if sub.empty:
                continue
            ax.plot(sub.index, sub["ess_rate_1e6"], "o-", label=p)
        ax.set_xlabel("(run index)")
    ax.set_ylabel("ESS per 1e6 grad-evals")
    ax.set_title("ESS-rate vs width (w=1 vs w=0.1)")
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
    if "width" in df.columns:
        probes = df["probe"].unique()
        for p in probes:
            sub = df[(df["probe"] == p)].sort_values("width")
            if sub.empty:
                continue
            ax.errorbar(
                sub["width"],
                sub["rho_hat_mean"],
                yerr=sub.get("rho_hat_se", 0),
                fmt="o-",
                capsize=3,
                label=p,
            )
        ax.set_xlabel("width")
        ax.set_title("Proxy LSI: rho_hat vs width (w=1 vs w=0.1)")
    else:
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
        ax.set_title("Proxy LSI: rho_hat vs probe")
    ax.set_ylabel("rho_hat (proxy LSI)")
    if "width" in df.columns:
        ax.legend()
    ax.set_yscale("log")
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
    fig, ax = plt.subplots()
    if "width" in df.columns:
        widths = sorted(df["width"].unique())
        rho_mins = [df[df["width"] == w]["rho_hat_mean"].min() for w in widths]
        ax.bar(range(len(widths)), rho_mins, tick_label=[f"w={w}" for w in widths])
        ax.set_xlabel("width")
        ax.set_title("Proxy LSI: rho_hat_min vs width")
    else:
        rho_min = df["rho_hat_mean"].min()
        ax.bar([0], [rho_min], label="min_f rho_hat")
        ax.set_xlabel("(summary)")
        ax.set_title("Proxy LSI: rho_hat_min")
    ax.set_ylabel("rho_hat_min")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--convergence-csv",
        default="experiments/summaries/convergence_by_width.csv",
        help="Use convergence_by_width.csv for w=1 vs w=0.1 comparison",
    )
    p.add_argument(
        "--lsi-csv",
        default="experiments/summaries/lsi_proxy_by_width.csv",
        help="Use lsi_proxy_by_width.csv for w=1 vs w=0.1 comparison",
    )
    p.add_argument("-o", "--out-dir", default="experiments/figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_C0_rhat_vs_width(Path(args.convergence_csv), out_dir / "rhat_vs_width.png")
    plot_C1_ess_rate_vs_width(Path(args.convergence_csv), out_dir / "ess_rate_vs_width.png")
    plot_C2_time_to_rhat(Path(args.convergence_csv), out_dir / "time_to_rhat_vs_width.png")
    plot_L1_rho_hat_vs_width(Path(args.lsi_csv), out_dir / "rho_hat_vs_width.png")
    plot_L2_rho_hat_min_vs_width(Path(args.lsi_csv), out_dir / "rho_hat_min_vs_width.png")
    print("Figures written to", out_dir)


if __name__ == "__main__":
    main()
