#!/usr/bin/env python3
"""
Aggregate negative-curvature runs across seeds and write the four PDFs from
``minimal_negative_curvature_experiment.md`` (sections "Plots" 1-4):

    1. ``gamma_emp_vs_width.pdf``                negative curvature scale.
    2. ``r_eff_over_p_vs_width.pdf``             effective negative rank ratio.
    3. ``E_iso_vs_E_aniso_vs_width.pdf``         exponent proxy comparison
                                                 (log y).
    4. ``cumulative_neg_trace_C_of_k.pdf``       cumulative negative trace
                                                 fraction C(k), one curve
                                                 per width.

Reads ``curvature_summary.csv`` and ``negative_eigs_*`` CSVs produced by
``scripts/run_neg_curvature.py`` from each ``*_seed*`` directory matching
``--run-glob``. Aggregates across seeds for the requested checkpoint.
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


def _load_summary_rows(run_dirs: list[Path], checkpoint: str) -> list[dict]:
    rows: list[dict] = []
    for rd in run_dirs:
        p = rd / "curvature_summary.csv"
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("checkpoint", "") != checkpoint:
                    continue
                rows.append({**r, "_run_dir": str(rd)})
    if not rows:
        raise RuntimeError(f"No curvature_summary rows found for checkpoint={checkpoint!r}")
    return rows


def _aggregate_by_width(rows: list[dict]) -> tuple[list[int], dict[str, dict[int, np.ndarray]]]:
    widths = sorted({int(r["width"]) for r in rows})
    keys = [
        "m", "p", "gamma_emp", "T_neg_top20", "T_neg_SLQ", "T_neg_used",
        "r_eff_neg", "r_eff_over_p", "r_eff_over_sqrt_m",
        "E_iso", "E_aniso", "E_aniso_over_E_iso",
        "sqrt_m_gamma_emp",
    ]
    out: dict[str, dict[int, np.ndarray]] = {k: {w: [] for w in widths} for k in keys}
    for r in rows:
        w = int(r["width"])
        for k in keys:
            v = r.get(k, "")
            try:
                out[k][w].append(float(v))
            except (TypeError, ValueError):
                out[k][w].append(np.nan)
    out_arr: dict[str, dict[int, np.ndarray]] = {}
    for k in keys:
        out_arr[k] = {w: np.asarray(out[k][w], dtype=np.float64) for w in widths}
    return widths, out_arr


def _mean_se(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return float(arr.mean()), se


def _load_eig_rows(run_dirs: list[Path], width: int, checkpoint: str) -> list[np.ndarray]:
    """Return list (one per run) of arrays of (-lambda)_+ sorted descending."""
    out: list[np.ndarray] = []
    pattern = f"negative_eigs_width{width}_seed*_{checkpoint}.csv"
    for rd in run_dirs:
        for p in sorted(rd.glob(pattern)):
            with p.open() as f:
                lams = [float(r["lambda"]) for r in csv.DictReader(f)]
            if not lams:
                continue
            eta = np.maximum(0.0, -np.asarray(lams, dtype=np.float64))
            eta_sorted = np.sort(eta)[::-1]
            out.append(eta_sorted)
    return out


def _build_C_of_k(per_seed: list[np.ndarray], T_neg_per_seed: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (k array, mean C(k) across seeds), padded to common max length.

    C(k) is computed per-seed using the seed's own ``T_neg`` denominator (when
    > 0) or NaN. Then we average across seeds at each k where the value is
    finite.
    """
    if not per_seed:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    K = max(arr.size for arr in per_seed)
    if K == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    mat = np.full((len(per_seed), K), np.nan, dtype=np.float64)
    for i, arr in enumerate(per_seed):
        if arr.size == 0:
            continue
        T = T_neg_per_seed[i] if np.isfinite(T_neg_per_seed[i]) and T_neg_per_seed[i] > 0.0 else float(arr.sum())
        if T <= 0.0:
            continue
        cum = np.cumsum(arr) / T
        mat[i, : arr.size] = cum
    mean = np.nanmean(mat, axis=0)
    ks = np.arange(1, K + 1, dtype=np.int64)
    return ks, mean


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Plot aggregated negative-curvature results.")
    ap.add_argument(
        "--run-glob", type=str, required=True,
        help='Glob matching seed run dirs, e.g. "experiments/neg_curv/main_seed*"',
    )
    ap.add_argument("--plot-out-dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="final",
                    help="Checkpoint to plot (init|mid|final).")
    ap.add_argument("--no-sqrt-m-ref", dest="sqrt_m_ref", action="store_false", default=True,
                    help="Disable c/sqrt(m) reference line in gamma_emp plot.")
    args = ap.parse_args()

    run_dirs = _load_run_dirs(args.run_glob)
    rows = _load_summary_rows(run_dirs, args.checkpoint)
    widths, agg = _aggregate_by_width(rows)

    plot_root = Path(args.plot_out_dir) / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    # Plot 1: gamma_emp vs width
    plt.figure(figsize=(6, 4))
    means, ses = [], []
    ms = []
    for w in widths:
        mu, se = _mean_se(agg["gamma_emp"][w])
        means.append(mu); ses.append(se)
        m_arr = agg["m"][w]
        m_arr = m_arr[np.isfinite(m_arr)]
        ms.append(float(m_arr[0]) if m_arr.size > 0 else float(64 * w))
    plt.errorbar(ms, means, yerr=ses, fmt="o-", label="mean gamma_emp", capsize=3)
    if args.sqrt_m_ref:
        finite = [(m, mu) for m, mu in zip(ms, means) if np.isfinite(mu) and mu > 0]
        if finite:
            m0, mu0 = finite[0]
            ref = [mu0 * float(np.sqrt(m0 / m)) for m in ms]
            plt.plot(ms, ref, "--", color="grey", label=r"$c/\sqrt{m}$ ref")
    plt.xlabel("width m (hidden)")
    plt.ylabel("gamma_emp = max(0, -lambda_min(N))")
    plt.title(f"gamma_emp vs width (checkpoint={args.checkpoint})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "gamma_emp_vs_width.pdf")
    plt.close()

    # Plot 2: r_eff / p vs width
    plt.figure(figsize=(6, 4))
    means, ses = [], []
    for w in widths:
        mu, se = _mean_se(agg["r_eff_over_p"][w])
        means.append(mu); ses.append(se)
    plt.errorbar(ms, means, yerr=ses, fmt="o-", capsize=3)
    plt.xlabel("width m (hidden)")
    plt.ylabel("r_eff_neg / p")
    plt.title(f"effective negative rank / p (checkpoint={args.checkpoint})")
    plt.tight_layout()
    plt.savefig(plot_root / "r_eff_over_p_vs_width.pdf")
    plt.close()

    # Plot 3: E_iso vs E_aniso (log y)
    plt.figure(figsize=(6, 4))
    iso_mu = []
    iso_se = []
    aniso_mu = []
    aniso_se = []
    for w in widths:
        m1, s1 = _mean_se(agg["E_iso"][w])
        m2, s2 = _mean_se(agg["E_aniso"][w])
        iso_mu.append(m1); iso_se.append(s1)
        aniso_mu.append(m2); aniso_se.append(s2)
    plt.errorbar(ms, iso_mu, yerr=iso_se, fmt="o-", label="E_iso = gamma_emp * p", capsize=3)
    plt.errorbar(ms, aniso_mu, yerr=aniso_se, fmt="s-", label="E_aniso = T_neg", capsize=3)
    plt.yscale("log")
    plt.xlabel("width m (hidden)")
    plt.ylabel("exponent proxy (log scale)")
    plt.title(f"E_iso vs E_aniso (checkpoint={args.checkpoint})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "E_iso_vs_E_aniso_vs_width.pdf")
    plt.close()

    # Plot 4: cumulative neg trace C(k), one curve per width
    plt.figure(figsize=(6, 4))
    for w in widths:
        per_seed = _load_eig_rows(run_dirs, w, args.checkpoint)
        T_per_seed = []
        for r in rows:
            if int(r["width"]) != w:
                continue
            try:
                t = float(r["T_neg_used"])
            except (TypeError, ValueError):
                t = float("nan")
            T_per_seed.append(t)
        T_per_seed = T_per_seed[: len(per_seed)] if per_seed else []
        ks, mean_C = _build_C_of_k(per_seed, T_per_seed)
        if ks.size == 0:
            continue
        plt.plot(ks, mean_C, "o-", label=f"m={int(64 * w)}")
    plt.xlabel("k")
    plt.ylabel("C(k) = cumulative neg trace fraction")
    plt.ylim(0.0, 1.05)
    plt.title(f"cumulative negative trace (checkpoint={args.checkpoint})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_root / "cumulative_neg_trace_C_of_k.pdf")
    plt.close()

    print(f"Wrote plots under {plot_root}")


if __name__ == "__main__":
    main()
