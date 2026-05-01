#!/usr/bin/env python3
"""
Aggregate Bayesian linear LSI width runs across seeds and write §12 PDF plots.

Requires completed ``bayeslin_lsi_width_convergence.py`` outputs (e.g. ``*_seed*`` dirs).
Uses matplotlib Agg backend.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

def _load_run_dirs(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not paths:
        raise FileNotFoundError(f"No directories matched glob: {pattern!r}")
    return paths


def _widths_from_summary(run_dir: Path) -> list[int]:
    p = run_dir / "width_summary.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    return sorted(int(r["width"]) for r in rows)


def _interp_log_metric(run_dir: Path, width: int, col: str, t_grid: np.ndarray) -> np.ndarray:
    p = run_dir / f"convergence_metrics_width{width}.csv"
    times: list[float] = []
    vals: list[float] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            times.append(float(r["time"]))
            v = float(r[col])
            vals.append(float(np.log(max(v, 1e-300))))
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    order = np.argsort(t)
    t, y = t[order], y[order]
    return np.interp(t_grid, t, y, left=np.nan, right=np.nan)


def _width_summary_row(run_dir: Path, width: int) -> dict[str, float]:
    with (run_dir / "width_summary.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["width"]) == width:
                return {k: float(r[k]) for k in r if k not in ("width", "stable")}
    raise KeyError(width)


def _max_time(run_dir: Path, width: int) -> float:
    with (run_dir / f"convergence_metrics_width{width}.csv").open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["time"])


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Plot aggregated BayesLin LSI width results.")
    ap.add_argument(
        "--run-glob",
        type=str,
        required=True,
        help='Glob matching seed run dirs, e.g. "experiments/bayeslin_lsi_width/pilot_seed*"',
    )
    ap.add_argument(
        "--plot-out-dir",
        type=str,
        required=True,
        help="Directory where plots/ subdirectory will be created.",
    )
    args = ap.parse_args()

    run_dirs = _load_run_dirs(args.run_glob)
    widths = _widths_from_summary(run_dirs[0])
    t_max = min(_max_time(rd, widths[-1]) for rd in run_dirs)
    t_grid = np.linspace(0.0, t_max, 200)

    plot_root = Path(args.plot_out_dir) / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    def band_plot(metric_col: str, log_label: str, fname: str) -> None:
        plt.figure(figsize=(7, 4))
        for m in widths:
            curves = []
            for rd in run_dirs:
                curves.append(_interp_log_metric(rd, m, metric_col, t_grid))
            mat = np.vstack(curves)
            mean = np.nanmean(mat, axis=0)
            se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(min=1))
            c = plt.plot(t_grid, mean, label=f"m={m}")[0].get_color()
            plt.fill_between(t_grid, mean - se, mean + se, alpha=0.2, color=c)
        plt.xlabel("time")
        plt.ylabel(log_label)
        plt.legend()
        plt.title(fname.replace(".pdf", "").replace("_", " "))
        plt.tight_layout()
        plt.savefig(plot_root / fname)
        plt.close()

    band_plot("e_euc", "log e_euc", "log_e_euc_vs_time_by_width.pdf")
    band_plot("e_H", "log e_H", "log_e_H_vs_time_by_width.pdf")
    band_plot("e_pred", "log e_pred", "log_e_pred_vs_time_by_width.pdf")

    # rate vs width with lambda_min
    rho_h = []
    lam_min = []
    for m in widths:
        rs = [_width_summary_row(rd, m)["rho_H_early"] for rd in run_dirs]
        ls = [_width_summary_row(rd, m)["lambda_min_H"] for rd in run_dirs]
        rho_h.append(float(np.nanmean(rs)))
        lam_min.append(float(np.nanmean(ls)))
    plt.figure(figsize=(6, 4))
    plt.plot(widths, rho_h, "o-", label="mean rho_H_early")
    ax2 = plt.gca().twinx()
    ax2.plot(widths, lam_min, "s--", color="tab:orange", label="mean lambda_min")
    plt.xlabel("width m")
    plt.ylabel("rho_H_early")
    ax2.set_ylabel("lambda_min(H)")
    lines, labels = plt.gca().get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    plt.legend(lines + l2, labels + lab2, loc="best")
    plt.title("rate vs width with lambda_min")
    plt.tight_layout()
    plt.savefig(plot_root / "rate_vs_width_with_lambda_min.pdf")
    plt.close()

    # C_LSI vs width
    plt.figure(figsize=(6, 4))
    c_lsi = [float(np.nanmean([_width_summary_row(rd, m)["C_LSI"] for rd in run_dirs])) for m in widths]
    plt.plot(widths, c_lsi, "o-")
    plt.xlabel("width m")
    plt.ylabel("mean C_LSI")
    plt.title("C_LSI vs width")
    plt.tight_layout()
    plt.savefig(plot_root / "C_LSI_vs_width.pdf")
    plt.close()

    # rho_H vs lambda_min scatter
    plt.figure(figsize=(5, 4))
    plt.scatter(lam_min, rho_h, c=widths, cmap="viridis", s=80)
    for i, m in enumerate(widths):
        plt.annotate(str(m), (lam_min[i], rho_h[i]), textcoords="offset points", xytext=(4, 2))
    plt.xlabel("lambda_min(H)")
    plt.ylabel("rho_H_early")
    plt.colorbar(label="m")
    plt.title("rate vs inv_C_LSI proxy (lambda_min)")
    plt.tight_layout()
    plt.savefig(plot_root / "rate_vs_inv_C_LSI.pdf")
    plt.close()

    # tau_H vs width
    plt.figure(figsize=(6, 4))
    t05 = [float(np.nanmean([_width_summary_row(rd, m)["tau_H_0p5"] for rd in run_dirs])) for m in widths]
    t01 = [float(np.nanmean([_width_summary_row(rd, m)["tau_H_0p1"] for rd in run_dirs])) for m in widths]
    plt.plot(widths, t05, "o-", label="mean tau_H(0.5)")
    plt.plot(widths, t01, "s-", label="mean tau_H(0.1)")
    plt.xlabel("width m")
    plt.ylabel("time")
    plt.legend()
    plt.title("tau_H vs width")
    plt.tight_layout()
    plt.savefig(plot_root / "tau_H_vs_width.pdf")
    plt.close()

    # spectrum violin-ish: boxplot of eigenvalues per width (pool seeds)
    plt.figure(figsize=(7, 4))
    data = []
    labels = []
    for m in widths:
        pooled: list[float] = []
        for rd in run_dirs:
            z = np.load(rd / f"spectrum_width{m}.npz")
            pooled.extend(z["eigenvalues_H"].astype(np.float64).tolist())
        data.append(pooled)
        labels.append(str(m))
    pos = np.arange(1, len(data) + 1)
    plt.boxplot(data, positions=pos.tolist())
    plt.xticks(pos, labels)
    plt.xlabel("width m")
    plt.ylabel("eigenvalue of H")
    plt.title("H spectrum by width (pooled seeds)")
    plt.tight_layout()
    plt.savefig(plot_root / "H_spectrum_by_width.pdf")
    plt.close()

    print(f"Wrote plots under {plot_root}")


if __name__ == "__main__":
    main()
