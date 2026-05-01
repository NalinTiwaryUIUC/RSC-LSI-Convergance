#!/usr/bin/env python3
"""
Fast confirmatory runs for the RF Pilot-2 stack.

Full Pilot-2 batches (~hours) are overkill for validating that coupling → analysis
(spectrum + quadratic + contraction summaries + init modes) still works after edits.

This script runs two presets:

  ci          — tiny problem; ~1 second total; same artifact types as production.
  prod-check  — modest n/T_phys so kappa/tau curves are less degenerate; ~10–30 s.

Both presets execute:
  - Hessian-init coupling + analyze_rf_width_contraction
  - Logit-init coupling + analyze_rf_width_contraction

Optional third leg (Gaussian calibration + normalized reanalysis), same as pilots.

Examples:

  python3 scripts/run_rf_pilot2_confirmatory.py --preset ci
  python3 scripts/run_rf_pilot2_confirmatory.py --preset prod-check --include-gaussian
  python3 scripts/run_rf_pilot2_confirmatory.py --preset prod-check --out-root experiments/rf_logistic_coupling/confirmatory_prod_check
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


PRESETS: dict[str, dict[str, object]] = {
    # Minimal dimensions to stress identical code paths / CSV schemas.
    "ci": {
        "n": 80,
        "p": 8,
        "widths": "12,24",
        "m_max": 24,
        "pairs": 4,
        "T_phys": 1.0,
        "h_factor": 0.05,
        "log_dt": 0.05,
        "seed": 0,
    },
    # Still cheap (~few thousand ULA steps); closer pilot geometry (nested widths, more pairs).
    "prod-check": {
        "n": 256,
        "p": 12,
        "widths": "16,32",
        "m_max": 32,
        "pairs": 8,
        "T_phys": 3.0,
        "h_factor": 0.05,
        "log_dt": 0.02,
        "seed": 0,
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _validate_coupling_run(run_dir: Path, *, init_mode: str, alpha: float) -> None:
    need = [
        "config.yaml",
        "pair_summary_all_widths.csv",
        "width_summary.csv",
        "hessian_spectrum_summary.csv",
        "quadratic_contraction_summary.csv",
        "rf_contraction_summary.csv",
    ]
    for rel in need:
        p = run_dir / rel
        assert p.is_file(), f"Missing {p}"

    # Spectrum: prior floor uses prior alpha from config, not lambda_min.
    with (run_dir / "hessian_spectrum_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty hessian_spectrum_summary"
    for r in rows:
        assert abs(float(r["prior_alpha"]) - alpha) < 1e-9
        for col in ("frac_prior_0p05", "frac_prior_0p10", "frac_prior_0p25"):
            x = float(r[col])
            assert 0.0 <= x <= 1.0

    with (run_dir / "pair_summary_all_widths.csv").open() as f:
        pairs = list(csv.DictReader(f))
    assert pairs
    r0 = pairs[0]
    assert r0["init_mode"] == init_mode
    for col in ("kappa_logit_early", "tau_logit_0p5", "init_D_logit", "kappa_H_early", "final_R_H"):
        assert col in r0 and r0[col] != "", f"missing {col}"
        assert _is_num(r0[col]), f"non-numeric {col}={r0[col]!r}"
    # Short confirmatory runs may not yield finite tau/kappa (sparse logs); init distances should be sane.
    assert np.isfinite(float(r0["init_D_logit"]))
    assert float(r0["init_D_logit"]) > 0.0
    assert np.isfinite(float(r0["final_R_H"]))


def _validate_gaussian(run_dir: Path) -> None:
    p = run_dir / "gaussian_width_summary_normalized.csv"
    assert p.is_file()
    rows = list(csv.DictReader(p.open()))
    assert rows
    for r in rows:
        m = float(r["width"])
        emu = float(r["final_E_mu"])
        assert abs(float(r["final_E_mu_norm"]) - emu / (m ** 0.5)) < 1e-8


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast RF Pilot-2 confirmatory harness.")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="prod-check")
    ap.add_argument("--out-root", type=str, default=str(ROOT / "experiments/rf_logistic_coupling/confirmatory_quick"))
    ap.add_argument("--include-gaussian", action="store_true")
    ap.add_argument("--keep", action="store_true", help="Do not delete out-root before running.")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    alpha = 0.3
    out_root = Path(args.out_root)
    if out_root.exists() and not args.keep:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(ROOT / "scripts" / "rf_logistic_coupling.py"),
        "--n",
        str(cfg["n"]),
        "--p",
        str(cfg["p"]),
        "--widths",
        str(cfg["widths"]),
        "--m-max",
        str(cfg["m_max"]),
        "--pairs",
        str(cfg["pairs"]),
        "--T-phys",
        str(cfg["T_phys"]),
        "--h-factor",
        str(cfg["h_factor"]),
        "--log-dt",
        str(cfg["log_dt"]),
        "--seed",
        str(cfg["seed"]),
        "--alpha",
        str(alpha),
    ]

    hess_dir = out_root / "coupling_hessian"
    logit_dir = out_root / "coupling_logit"

    _run(common + ["--init-mode", "hessian", "--out-dir", str(hess_dir)], ROOT)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_rf_width_contraction.py"),
            "--run-dir",
            str(hess_dir),
        ],
        ROOT,
    )
    _validate_coupling_run(hess_dir, init_mode="hessian", alpha=alpha)

    _run(
        common
        + [
            "--init-mode",
            "logit",
            "--init-logit-radius",
            "1.0",
            "--init-logit-ridge",
            "1e-4",
            "--out-dir",
            str(logit_dir),
        ],
        ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_rf_width_contraction.py"),
            "--run-dir",
            str(logit_dir),
        ],
        ROOT,
    )
    _validate_coupling_run(logit_dir, init_mode="logit", alpha=alpha)

    if args.include_gaussian:
        gdir = out_root / "gaussian"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rf_gaussian_calibration.py"),
                "--n",
                str(min(int(cfg["n"]), 160)),
                "--p",
                str(cfg["p"]),
                "--widths",
                str(cfg["widths"]),
                "--m-max",
                str(cfg["m_max"]),
                "--chains",
                "16",
                "--T-phys",
                str(min(float(cfg["T_phys"]), 2.0)),
                "--h-factor",
                str(cfg["h_factor"]),
                "--log-dt",
                str(cfg["log_dt"]),
                "--seed",
                str(cfg["seed"]),
                "--out-dir",
                str(gdir),
            ],
            ROOT,
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_rf_gaussian_calibration.py"),
                "--run-dir",
                str(gdir),
            ],
            ROOT,
        )
        _validate_gaussian(gdir)

    print("\nOK — confirmatory Pilot-2 stack validated.")
    print(f"Artifacts under: {out_root}")


if __name__ == "__main__":
    main()
