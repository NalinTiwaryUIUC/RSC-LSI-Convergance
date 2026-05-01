#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze RF Gaussian calibration outputs.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--normalize-by-dim", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob("gaussian_errors_width*.csv"))
    if not files:
        raise FileNotFoundError(f"No gaussian_errors_width*.csv under {run_dir}")

    summary_rows = []
    for f in files:
        width = int(f.stem.replace("gaussian_errors_width", ""))
        rows = []
        with f.open() as fh:
            rd = csv.DictReader(fh)
            for r in rd:
                e_mu = float(r["E_mu"])
                e_sigma = float(r["E_sigma"])
                w2 = float(r["W2"])
                rows.append(
                    {
                        "width": width,
                        "step": int(r["step"]),
                        "time": float(r["time"]),
                        "E_mu": e_mu,
                        "E_sigma": e_sigma,
                        "W2": w2,
                        "E_mu_norm": e_mu / np.sqrt(width),
                        "E_sigma_norm": e_sigma,  # already dimension-normalized by construction
                        "W2_norm": w2 / np.sqrt(width),
                    }
                )
        out_trace = out_dir / f"gaussian_errors_width{width}_normalized.csv"
        with out_trace.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "width",
                    "step",
                    "time",
                    "E_mu",
                    "E_sigma",
                    "W2",
                    "E_mu_norm",
                    "E_sigma_norm",
                    "W2_norm",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        last = rows[-1]
        summary_rows.append(
            {
                "width": width,
                "final_E_mu": last["E_mu"],
                "final_E_sigma": last["E_sigma"],
                "final_W2": last["W2"],
                "final_E_mu_norm": last["E_mu_norm"],
                "final_E_sigma_norm": last["E_sigma_norm"],
                "final_W2_norm": last["W2_norm"],
            }
        )

    summary_rows = sorted(summary_rows, key=lambda r: int(r["width"]))
    out_summary = out_dir / "gaussian_width_summary_normalized.csv"
    with out_summary.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    out_md = out_dir / "gaussian_width_summary_normalized.md"
    lines = [
        "# Gaussian calibration normalized summary",
        "",
        "| width | final E_mu | final E_mu/sqrt(m) | final E_sigma_norm | final W2 | final W2/sqrt(m) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['width']} | {r['final_E_mu']:.4g} | {r['final_E_mu_norm']:.4g} | "
            f"{r['final_E_sigma_norm']:.4g} | {r['final_W2']:.4g} | {r['final_W2_norm']:.4g} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

