#!/usr/bin/env python3
"""
Aggregate escape-diagnostic runs (I1 / I2 / I3 / …) for side-by-side comparison on the cluster.

Reads iter_metrics.jsonl from each run dir (under a glob), computes:
  - Pooled early / mid / late means for primary + secondary keys (same spirit as report_chain_convergence)
  - Late-window means of U_prior/U_data and U_prior/U_train (prior–vs–likelihood scale; skip non-finite)

Run from project root on the cluster, e.g.:

  python scripts/summarize_escape_init_comparison.py \\
    --runs-dir experiments/runs \\
    --glob 'w1_n512_h5e-06_T100000*_ul_initI*_chain*' \\
    --out-md experiments/summaries/escape_w1_init_comparison.md \\
    --out-csv experiments/summaries/escape_w1_init_comparison.csv

Adjust --glob to match your h, T, and n_train. Use a tighter glob if you have other w1 runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_chain_convergence import (  # noqa: E402
    PRIMARY_ITER_KEYS,
    SECONDARY_ITER_KEYS,
    discover_runs,
    summarize_series,
)


def load_iter(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for k, v in list(rec.items()):
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = float("nan")
        out.append(rec)
    return out


def init_tag_from_run_name(name: str) -> str:
    """Extract initI1 / initI2_step1600 / initI3_sigma0p02 from run directory name."""
    # Stop _sigma value before _chain (underscore is \\w, so do not use \\w+ after _sigma).
    m = re.search(r"(initI[123](?:_step\d+|_sigma[^_]+)?)", name)
    if m:
        return m.group(1)
    return "unknown"


def chain_sort_key(p: Path) -> tuple[int, str]:
    m = re.search(r"chain(\d+)", p.name)
    cid = int(m.group(1)) if m else 999
    return (cid, p.name)


def ratio_stats(
    recs: list[dict[str, Any]],
    num: str,
    den: str,
    late_frac: float = 0.25,
) -> dict[str, float]:
    """Mean ratio num/den over records in the last `late_frac` fraction of steps (by sorted step)."""
    recs = sorted(recs, key=lambda r: int(r.get("step", 0)))
    if not recs:
        return {"mean": float("nan"), "n": 0.0}
    n = len(recs)
    start = max(0, int(math.floor(n * (1.0 - late_frac))))
    window = recs[start:]
    vals: list[float] = []
    for r in window:
        a, b = r.get(num), r.get(den)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        af, bf = float(a), float(b)
        if not math.isfinite(af) or not math.isfinite(bf) or abs(bf) < 1e-30:
            continue
        vals.append(af / bf)
    if not vals:
        return {"mean": float("nan"), "n": 0.0}
    return {"mean": float(np.mean(vals)), "n": float(len(vals))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare escape init regimes from iter_metrics.jsonl")
    ap.add_argument("--runs-dir", type=str, default="experiments/runs")
    ap.add_argument(
        "--glob",
        type=str,
        default="w1_*_ul_initI*_chain*",
        help="Glob under runs-dir; quote in shell. Tighten with h,T,n if needed.",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="experiments/summaries/escape_init_comparison.md",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="experiments/summaries/escape_init_comparison.csv",
    )
    ap.add_argument(
        "--late-frac",
        type=float,
        default=0.25,
        help="Fraction of iter rows (per chain, then pooled) for late-window ratio stats",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir

    paths = discover_runs(runs_dir, args.glob)
    if not paths:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    by_init: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_init[init_tag_from_run_name(p.name)].append(p)
    for tag in by_init:
        by_init[tag].sort(key=chain_sort_key)

    md_lines = [
        "# Escape init comparison (from `iter_metrics.jsonl`)",
        "",
        f"Runs dir: `{runs_dir}` glob: `{args.glob}`",
        f"Init groups: **{len(by_init)}**",
        "",
        "Ratios **U_prior/U_data** and **U_prior/U_train** use the **last {pct}%** of logged rows per chain, pooled within each init group.".format(
            pct=int(round(100 * args.late_frac))
        ),
        "",
    ]

    csv_rows: list[dict[str, Any]] = []
    all_keys = tuple(PRIMARY_ITER_KEYS) + tuple(SECONDARY_ITER_KEYS)

    for tag in sorted(by_init.keys()):
        gpaths = by_init[tag]
        md_lines.append(f"## `{tag}` ({len(gpaths)} chains)")
        md_lines.append("")
        all_recs: list[dict[str, Any]] = []
        for p in gpaths:
            recs = load_iter(p / "iter_metrics.jsonl")
            all_recs.extend(recs)
            md_lines.append(f"- `{p.name}`: **{len(recs)}** iter rows")

        r1 = ratio_stats(all_recs, "U_prior", "U_data", args.late_frac)
        r2 = ratio_stats(all_recs, "U_prior", "U_train", args.late_frac)
        md_lines.append("")
        md_lines.append(
            f"- **Late mean U_prior/U_data** ≈ {r1['mean']:.6g} (n={int(r1['n'])} finite ratios)"
        )
        md_lines.append(
            f"- **Late mean U_prior/U_train** ≈ {r2['mean']:.6g} (n={int(r2['n'])} finite ratios)"
        )
        md_lines.append("")

        md_lines.append("### Primary iter keys (pooled all chains)")
        md_lines.append(
            "| key | n | mean | std | early | mid | late | Δ(2nd−1st half) |"
        )
        md_lines.append("|-----|---|------|-----|-------|-----|------|----------------|")
        for key in PRIMARY_ITER_KEYS:
            s = summarize_series(all_recs, key)
            if s["n"] == 0:
                md_lines.append(f"| {key} | 0 | — | — | — | — | — | — |")
                continue
            md_lines.append(
                f"| {key} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | "
                f"{s['delta_2nd_minus_1st']:.6g} |"
            )
            csv_rows.append(
                {
                    "init_tag": tag,
                    "table": "primary",
                    "key": key,
                    "n": s["n"],
                    "mean": s["mean"],
                    "std": s["std"],
                    "early_mean": s["early_mean"],
                    "mid_mean": s["mid_mean"],
                    "late_mean": s["late_mean"],
                    "delta_2nd_minus_1st": s["delta_2nd_minus_1st"],
                }
            )

        md_lines.append("")
        md_lines.append("### Secondary iter keys (pooled)")
        md_lines.append(
            "| key | n | mean | std | early | mid | late | Δ(2nd−1st half) |"
        )
        md_lines.append("|-----|---|------|-----|-------|-----|------|----------------|")
        for key in SECONDARY_ITER_KEYS:
            s = summarize_series(all_recs, key)
            if s["n"] == 0:
                md_lines.append(f"| {key} | 0 | — | — | — | — | — | — |")
                continue
            md_lines.append(
                f"| {key} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | "
                f"{s['delta_2nd_minus_1st']:.6g} |"
            )
            csv_rows.append(
                {
                    "init_tag": tag,
                    "table": "secondary",
                    "key": key,
                    "n": s["n"],
                    "mean": s["mean"],
                    "std": s["std"],
                    "early_mean": s["early_mean"],
                    "mid_mean": s["mid_mean"],
                    "late_mean": s["late_mean"],
                    "delta_2nd_minus_1st": s["delta_2nd_minus_1st"],
                }
            )

        csv_rows.append(
            {
                "init_tag": tag,
                "table": "ratio_late",
                "key": "U_prior_over_U_data",
                "n": r1["n"],
                "mean": r1["mean"],
                "std": "",
                "early_mean": "",
                "mid_mean": "",
                "late_mean": "",
                "delta_2nd_minus_1st": "",
            }
        )
        csv_rows.append(
            {
                "init_tag": tag,
                "table": "ratio_late",
                "key": "U_prior_over_U_train",
                "n": r2["n"],
                "mean": r2["mean"],
                "std": "",
                "early_mean": "",
                "mid_mean": "",
                "late_mean": "",
                "delta_2nd_minus_1st": "",
            }
        )
        md_lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote", out_md)

    out_csv = Path(args.out_csv)
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print("Wrote", out_csv)


if __name__ == "__main__":
    main()
