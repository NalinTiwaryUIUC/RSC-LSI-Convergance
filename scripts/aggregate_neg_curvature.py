#!/usr/bin/env python3
"""
Aggregate ``curvature_summary.csv`` rows across seed directories (glob).

Typical use after a multi-seed pilot::

    python3 scripts/aggregate_neg_curvature.py \\
        --run-glob 'experiments/neg_curv/pilot_seed*' \\
        --checkpoint final \\
        --out-csv experiments/neg_curv/pilot_aggregate_final.csv

Prints mean ± sample std for paper-facing columns and, when SLQ is present,
``T_neg_SLQ / T_neg_top20`` per width.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def _run_dirs(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not paths:
        raise FileNotFoundError(f"No directories matched: {pattern!r}")
    return paths


def _collect_rows(run_dirs: list[Path], checkpoint: str) -> list[dict]:
    rows: list[dict] = []
    for rd in run_dirs:
        p = rd / "curvature_summary.csv"
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("checkpoint", "") != checkpoint:
                    continue
                rows.append(dict(r))
    if not rows:
        raise RuntimeError(f"No rows for checkpoint={checkpoint!r}")
    return rows


def _f(x: str) -> float:
    if x is None or x == "" or str(x).lower() == "nan":
        return float("nan")
    return float(x)


def _metric_value(r: dict, c: str) -> float:
    """Read metric from row, with fallbacks for older CSVs missing derived columns."""
    raw = r.get(c)
    if raw not in (None, "") and str(raw).lower() != "nan":
        v = _f(str(raw))
        if math.isfinite(v):
            return v
    g = _f(r.get("gamma_emp", ""))
    t20 = _f(r.get("T_neg_top20", ""))
    p = _f(r.get("p", ""))
    if c == "r_eff_top20" and math.isfinite(g) and g > 0.0 and math.isfinite(t20):
        return t20 / g
    if c == "r_eff_top20_over_p" and math.isfinite(g) and g > 0.0 and math.isfinite(t20) and math.isfinite(p) and p > 0:
        return t20 / (g * p)
    if c == "E_aniso_top20_over_E_iso":
        eiso = _f(r.get("E_iso", ""))
        if math.isfinite(t20) and math.isfinite(eiso) and eiso > 0.0:
            return t20 / eiso
    if c == "r_eff_neg":
        t_used = _f(r.get("T_neg_used", ""))
        if not math.isfinite(t_used):
            tslq = _f(r.get("T_neg_SLQ", ""))
            t_used = tslq if math.isfinite(tslq) else t20
        if math.isfinite(g) and g > 0.0 and math.isfinite(t_used):
            return t_used / g
    if c == "r_eff_over_p" and raw in (None, "", "nan"):
        t_used = _f(r.get("T_neg_used", ""))
        if not math.isfinite(t_used):
            tslq = _f(r.get("T_neg_SLQ", ""))
            t_used = tslq if math.isfinite(tslq) else t20
        if math.isfinite(g) and g > 0.0 and math.isfinite(t_used) and math.isfinite(p) and p > 0:
            return (t_used / g) / p
    if c == "r_eff_over_sqrt_m":
        m = _f(r.get("m", ""))
        t_used = _f(r.get("T_neg_used", ""))
        if not math.isfinite(t_used):
            tslq = _f(r.get("T_neg_SLQ", ""))
            t_used = tslq if math.isfinite(tslq) else t20
        if math.isfinite(g) and g > 0.0 and math.isfinite(t_used) and math.isfinite(m) and m > 0.0:
            return (t_used / g) / math.sqrt(m)
    if c == "E_iso" and not math.isfinite(_f(str(raw))):
        if math.isfinite(g) and math.isfinite(p) and p > 0.0:
            return g * p
    if c == "E_aniso" and not math.isfinite(_f(str(raw))):
        t_used = _f(r.get("T_neg_used", ""))
        if not math.isfinite(t_used):
            tslq = _f(r.get("T_neg_SLQ", ""))
            t_used = tslq if math.isfinite(tslq) else t20
        if math.isfinite(t_used):
            return t_used
    if c == "E_aniso_over_E_iso" and not math.isfinite(_f(str(raw))):
        eiso = _metric_value(r, "E_iso")
        ean = _metric_value(r, "E_aniso")
        if math.isfinite(eiso) and eiso > 0.0 and math.isfinite(ean):
            return ean / eiso
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate neg-curvature CSVs across seeds.")
    ap.add_argument("--run-glob", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="final")
    ap.add_argument("--out-csv", type=str, default="", help="Optional path to write aggregate table.")
    args = ap.parse_args()

    run_dirs = _run_dirs(args.run_glob)
    rows = _collect_rows(run_dirs, args.checkpoint)
    widths = sorted({int(r["width"]) for r in rows})

    cols = [
        "p",
        "train_loss",
        "train_acc",
        "curv_loss",
        "curv_acc",
        "gamma_emp",
        "sqrt_m_gamma_emp",
        "T_neg_top20",
        "T_neg_SLQ",
        "r_eff_neg",
        "r_eff_top20",
        "r_eff_over_p",
        "r_eff_over_sqrt_m",
        "r_eff_top20_over_p",
        "E_iso",
        "E_aniso",
        "E_aniso_top20_over_E_iso",
        "E_aniso_over_E_iso",
    ]

    out_lines: list[str] = []
    header = (
        f"# Aggregated over {len(run_dirs)} dirs, checkpoint={args.checkpoint}, "
        f"n_seeds_per_width varies; rows={len(rows)}\n"
        "# width  metric_mean  metric_std  (sample std over seeds)\n"
    )
    print(header, end="")
    out_lines.append(header.rstrip("\n"))

    for w in widths:
        block = [r for r in rows if int(r["width"]) == w]
        print(f"\n## width={w}  (n={len(block)} rows)\n")
        out_lines.append(f"## width={w}  (n={len(block)} rows)")
        for c in cols:
            vals = [_metric_value(r, c) for r in block]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                line = f"  {c}: (no finite values)"
            elif len(vals) == 1:
                line = f"  {c}: {vals[0]:.6g}"
            else:
                m = mean(vals)
                s = stdev(vals)
                line = f"  {c}: {m:.6g} ± {s:.4g}"
            print(line)
            out_lines.append(line)

        # SLQ vs top-20 ratio (per seed, then summarize)
        ratios = []
        for r in block:
            t20 = _f(r.get("T_neg_top20", ""))
            tslq = _f(r.get("T_neg_SLQ", ""))
            if math.isfinite(tslq) and math.isfinite(t20) and t20 > 0.0:
                ratios.append(tslq / t20)
        if ratios:
            if len(ratios) == 1:
                rline = f"  T_neg_SLQ / T_neg_top20: {ratios[0]:.4g}"
            else:
                rline = f"  T_neg_SLQ / T_neg_top20: {mean(ratios):.4g} ± {stdev(ratios):.4g}"
            print(rline)
            out_lines.append(rline)
        else:
            msg = "  T_neg_SLQ / T_neg_top20: (SLQ not run or all NaN)"
            print(msg)
            out_lines.append(msg)

    if args.out_csv.strip():
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # Wide one-row-per-width CSV
        fieldnames = ["width", "n_seeds", "checkpoint"] + [
            f"{c}_mean" for c in cols
        ] + [f"{c}_std" for c in cols] + ["T_SLQ_over_T_top20_mean", "T_SLQ_over_T_top20_std"]
        agg_rows: list[dict] = []
        for w in widths:
            block = [r for r in rows if int(r["width"]) == w]
            rec: dict[str, str | int | float] = {
                "width": w,
                "n_seeds": len(block),
                "checkpoint": args.checkpoint,
            }
            for c in cols:
                vals = [_metric_value(r, c) for r in block]
                vals = [v for v in vals if math.isfinite(v)]
                if not vals:
                    rec[f"{c}_mean"] = float("nan")
                    rec[f"{c}_std"] = float("nan")
                elif len(vals) == 1:
                    rec[f"{c}_mean"] = vals[0]
                    rec[f"{c}_std"] = 0.0
                else:
                    rec[f"{c}_mean"] = mean(vals)
                    rec[f"{c}_std"] = stdev(vals)
            ratios = []
            for r in block:
                t20 = _f(r.get("T_neg_top20", ""))
                tslq = _f(r.get("T_neg_SLQ", ""))
                if math.isfinite(tslq) and math.isfinite(t20) and t20 > 0.0:
                    ratios.append(tslq / t20)
            if len(ratios) == 0:
                rec["T_SLQ_over_T_top20_mean"] = float("nan")
                rec["T_SLQ_over_T_top20_std"] = float("nan")
            elif len(ratios) == 1:
                rec["T_SLQ_over_T_top20_mean"] = ratios[0]
                rec["T_SLQ_over_T_top20_std"] = 0.0
            else:
                rec["T_SLQ_over_T_top20_mean"] = mean(ratios)
                rec["T_SLQ_over_T_top20_std"] = stdev(ratios)
            agg_rows.append(rec)
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(agg_rows)
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
