#!/usr/bin/env python3
"""
Bayesian linear regression width experiment: deterministic GD on exact Gaussian posterior.

Multi-seed layout: if ``--seeds`` is set (comma-separated), each run is written to
``{parent(out_dir)}/{stem(out_dir)}_seed{s}`` (e.g. ``.../pilot_seed0``). If only
``--seed`` is used, outputs go directly to ``--out-dir``.

Pilot / main examples::

    python3 scripts/bayeslin_lsi_width_convergence.py \\
      --widths 32,64,128,256 --n-over-m 4 --alpha 0.3 --sigma 1.0 \\
      --teacher-scale 1.0 --h-factor 0.05 --T-phys 10.0 --log-dt 0.02 \\
      --seeds 0,1,2 --out-dir experiments/bayeslin_lsi_width/pilot

    python3 scripts/bayeslin_lsi_width_convergence.py \\
      --widths 32,64,128,256,512 --n-over-m 4 --alpha 0.3 --sigma 1.0 \\
      --teacher-scale 1.0 --h-factor 0.05 --T-phys 20.0 --log-dt 0.02 \\
      --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir experiments/bayeslin_lsi_width/main
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from bayeslin_lsi_common import (
    DEFAULT_RATE_WINDOWS,
    THRESHOLDS,
    build_posterior_precision,
    build_rhs_b,
    convergence_metrics_row,
    first_time_leq,
    fit_rate,
    generate_linear_regression_data,
    global_step_size,
    grad_U,
    potential_U,
    spectrum_summary,
    theta_star_from_normal_eqs,
)


def _parse_widths(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("width list is empty")
    return out


def _parse_seeds(s: str) -> list[int] | None:
    s = s.strip()
    if not s:
        return None
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    return out or None


def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-300)))


def _clip_window(t0: float, t1: float, t_phys: float) -> tuple[float, float]:
    return t0, min(t1, t_phys)


def resolve_run_dirs(out_dir: Path, seeds_csv: str | None, single_seed: int) -> list[tuple[int, Path]]:
    if seeds_csv is None:
        parsed = None
    else:
        parsed = _parse_seeds(seeds_csv)
    if parsed:
        parent = out_dir.parent
        stem = out_dir.name
        return [(s, parent / f"{stem}_seed{s}") for s in parsed]
    return [(single_seed, out_dir)]


CONVERGENCE_FIELDNAMES = [
    "width",
    "step",
    "time",
    "D_euc",
    "e_euc",
    "D_H",
    "e_H",
    "U_gap",
    "e_U",
    "D_pred",
    "e_pred",
    "grad_norm",
    "theta_norm",
    "max_abs_theta",
    "nan_or_inf",
]

WIDTH_SUMMARY_FIELDNAMES = [
    "width",
    "n",
    "alpha",
    "sigma",
    "lambda_min_H",
    "lambda_med_H",
    "lambda_max_H",
    "condition_H",
    "C_LSI",
    "C_PI",
    "h",
    "h_lambda_max",
    "rho_euc_early",
    "rho_H_early",
    "rho_pred_early",
    "rho_euc_mid",
    "rho_H_mid",
    "rho_pred_mid",
    "tau_euc_0p5",
    "tau_euc_0p1",
    "tau_H_0p5",
    "tau_H_0p1",
    "tau_pred_0p5",
    "tau_pred_0p1",
    "final_e_euc",
    "final_e_H",
    "final_e_pred",
    "final_e_U",
    "stable",
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayesian linear LSI width convergence (deterministic GD).")
    p.add_argument("--widths", type=str, default="32,64,128,256,512")
    p.add_argument("--n-over-m", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--teacher-scale", type=float, default=1.0)
    p.add_argument("--h-factor", type=float, default=0.05)
    p.add_argument("--T-phys", type=float, default=20.0)
    p.add_argument("--log-dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0, help="Used when --seeds is not passed.")
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds; if set, writes to {parent(out_dir)}/{stem}_seed{s}/.",
    )
    p.add_argument("--out-dir", type=str, required=True)
    return p


def run_one_seed(
    out_dir: Path,
    seed: int,
    widths: list[int],
    n_over_m: int,
    alpha: float,
    sigma: float,
    teacher_scale: float,
    h_factor: float,
    t_phys: float,
    log_dt: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cfg = {
        "model": "bayesian_linear_regression",
        "widths": widths,
        "n_over_m": n_over_m,
        "alpha": alpha,
        "sigma": sigma,
        "teacher_scale": teacher_scale,
        "h_factor": h_factor,
        "T_phys": t_phys,
        "log_dt": log_dt,
        "seed": seed,
        "init": "zero",
        "dynamics": "gradient_descent",
    }
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    per_width: dict[int, dict[str, object]] = {}
    lam_max_by_width: dict[int, float] = {}

    for m in widths:
        X, y, theta_true = generate_linear_regression_data(
            m=m,
            c=n_over_m,
            alpha=alpha,
            sigma=sigma,
            teacher_scale=teacher_scale,
            rng=rng,
        )
        H = build_posterior_precision(X, alpha, sigma)
        b = build_rhs_b(X, y, sigma)
        theta_star = theta_star_from_normal_eqs(H, b)
        spec = spectrum_summary(H)
        lam_max_by_width[m] = spec["lambda_max"]
        per_width[m] = {
            "X": X,
            "y": y,
            "theta_true": theta_true,
            "theta_star": theta_star,
            "H": H,
            "b": b,
            "spec": spec,
        }

    h = global_step_size(lam_max_by_width, h_factor)
    n_steps = int(np.ceil(t_phys / h))
    s_log = max(1, int(np.floor(log_dt / h)))

    width_summary_rows: list[dict[str, object]] = []

    for m in widths:
        pack = per_width[m]
        X = pack["X"]  # type: ignore[assignment]
        y = pack["y"]  # type: ignore[assignment]
        H = pack["H"]  # type: ignore[assignment]
        b = pack["b"]  # type: ignore[assignment]
        theta_true = pack["theta_true"]  # type: ignore[assignment]
        theta_star = pack["theta_star"]  # type: ignore[assignment]
        spec = pack["spec"]  # type: ignore[assignment]
        n = int(X.shape[0])

        theta = np.zeros(m, dtype=np.float64)
        u_star = potential_U(theta_star, X, y, alpha, sigma)
        d0 = -theta_star
        delta0_euc = float(np.linalg.norm(d0))
        delta0_h = float(np.sqrt(max(d0 @ H @ d0, 0.0)))
        delta0_pred = float(np.linalg.norm(X @ d0) / np.sqrt(n))
        u0 = potential_U(theta, X, y, alpha, sigma)
        u0_gap = float(u0 - u_star)

        times: list[float] = []
        log_e_euc: list[float] = []
        log_e_h: list[float] = []
        log_e_pred: list[float] = []
        log_e_u: list[float] = []
        conv_rows: list[dict[str, object]] = []
        stable = True
        h_lam = h * spec["lambda_max"]
        if h_lam >= 1.0:
            stable = False

        for step in range(n_steps + 1):
            t = min(step * h, t_phys)
            row_m = convergence_metrics_row(
                theta,
                theta_star,
                X,
                H,
                alpha,
                sigma,
                y,
                delta0_euc,
                delta0_h,
                delta0_pred,
                u0_gap,
            )
            if int(row_m["nan_or_inf"]):
                stable = False
            if step % s_log == 0 or step == n_steps:
                row = {
                    "width": m,
                    "step": step,
                    "time": t,
                    **{k: row_m[k] for k in CONVERGENCE_FIELDNAMES[3:]},
                }
                conv_rows.append(row)
                times.append(t)
                log_e_euc.append(_safe_log(row_m["e_euc"]))
                log_e_h.append(_safe_log(row_m["e_H"]))
                log_e_pred.append(_safe_log(row_m["e_pred"]))
                log_e_u.append(_safe_log(row_m["e_U"]))
            if step < n_steps:
                theta = theta - h * grad_U(theta, H, b)

        times_a = np.asarray(times, dtype=np.float64)
        lee = np.asarray(log_e_euc, dtype=np.float64)
        leh = np.asarray(log_e_h, dtype=np.float64)
        lep = np.asarray(log_e_pred, dtype=np.float64)
        leu = np.asarray(log_e_u, dtype=np.float64)

        def rho_for(log_e: np.ndarray, t0: float, t1: float) -> float:
            a, bclip = _clip_window(t0, t1, t_phys)
            return fit_rate(times_a, log_e, a, bclip)

        rho_e_e = rho_for(lee, 0.0, 2.0)
        rho_h_e = rho_for(leh, 0.0, 2.0)
        rho_p_e = rho_for(lep, 0.0, 2.0)
        rho_e_m = rho_for(lee, 2.0, 10.0)
        rho_h_m = rho_for(leh, 2.0, 10.0)
        rho_p_m = rho_for(lep, 2.0, 10.0)

        e_euc_s = np.exp(lee)
        e_h_s = np.exp(leh)
        e_p_s = np.exp(lep)
        e_u_s = np.exp(leu)

        width_summary_rows.append(
            {
                "width": m,
                "n": n,
                "alpha": alpha,
                "sigma": sigma,
                "lambda_min_H": spec["lambda_min"],
                "lambda_med_H": spec["lambda_med"],
                "lambda_max_H": spec["lambda_max"],
                "condition_H": spec["condition"],
                "C_LSI": spec["C_LSI"],
                "C_PI": spec["C_PI"],
                "h": h,
                "h_lambda_max": h_lam,
                "rho_euc_early": rho_e_e,
                "rho_H_early": rho_h_e,
                "rho_pred_early": rho_p_e,
                "rho_euc_mid": rho_e_m,
                "rho_H_mid": rho_h_m,
                "rho_pred_mid": rho_p_m,
                "tau_euc_0p5": first_time_leq(times_a, e_euc_s, 0.5),
                "tau_euc_0p1": first_time_leq(times_a, e_euc_s, 0.1),
                "tau_H_0p5": first_time_leq(times_a, e_h_s, 0.5),
                "tau_H_0p1": first_time_leq(times_a, e_h_s, 0.1),
                "tau_pred_0p5": first_time_leq(times_a, e_p_s, 0.5),
                "tau_pred_0p1": first_time_leq(times_a, e_p_s, 0.1),
                "final_e_euc": float(e_euc_s[-1]),
                "final_e_H": float(e_h_s[-1]),
                "final_e_pred": float(e_p_s[-1]),
                "final_e_U": float(e_u_s[-1]),
                "stable": int(stable and h_lam < 1.0),
            }
        )

        with (out_dir / f"convergence_metrics_width{m}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CONVERGENCE_FIELDNAMES)
            w.writeheader()
            w.writerows(conv_rows)

        evals = np.linalg.eigvalsh(H)
        np.savez(
            out_dir / f"spectrum_width{m}.npz",
            eigenvalues_H=evals,
            lambda_min=np.array([spec["lambda_min"]], dtype=np.float64),
            lambda_med=np.array([spec["lambda_med"]], dtype=np.float64),
            lambda_max=np.array([spec["lambda_max"]], dtype=np.float64),
            condition=np.array([spec["condition"]], dtype=np.float64),
            C_LSI=np.array([spec["C_LSI"]], dtype=np.float64),
        )
        np.savez(
            out_dir / f"posterior_width{m}.npz",
            X=X,
            y=y,
            theta_true=theta_true,
            theta_star=theta_star,
            H=H,
            b=b,
        )

        rate_rows = []
        for wdw in DEFAULT_RATE_WINDOWS:
            t0c, t1c = _clip_window(wdw.t0, wdw.t1, t_phys)
            rate_rows.append(
                {
                    "metric": "e_euc",
                    "window": wdw.name,
                    "t0": t0c,
                    "t1": t1c,
                    "rho_hat": fit_rate(times_a, lee, t0c, t1c),
                }
            )
            rate_rows.append(
                {
                    "metric": "e_H",
                    "window": wdw.name,
                    "t0": t0c,
                    "t1": t1c,
                    "rho_hat": fit_rate(times_a, leh, t0c, t1c),
                }
            )
            rate_rows.append(
                {
                    "metric": "e_pred",
                    "window": wdw.name,
                    "t0": t0c,
                    "t1": t1c,
                    "rho_hat": fit_rate(times_a, lep, t0c, t1c),
                }
            )
            rate_rows.append(
                {
                    "metric": "e_U",
                    "window": wdw.name,
                    "t0": t0c,
                    "t1": t1c,
                    "rho_hat": fit_rate(times_a, leu, t0c, t1c),
                }
            )
        with (out_dir / f"rate_summary_width{m}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "window", "t0", "t1", "rho_hat"])
            w.writeheader()
            w.writerows(rate_rows)

        thr_rows = []
        for thr in THRESHOLDS:
            thr_rows.append(
                {"metric": "e_euc", "threshold": thr, "tau": first_time_leq(times_a, e_euc_s, thr)}
            )
            thr_rows.append(
                {"metric": "e_H", "threshold": thr, "tau": first_time_leq(times_a, e_h_s, thr)}
            )
            thr_rows.append(
                {"metric": "e_pred", "threshold": thr, "tau": first_time_leq(times_a, e_p_s, thr)}
            )
            thr_rows.append(
                {"metric": "e_U", "threshold": thr, "tau": first_time_leq(times_a, e_u_s, thr)}
            )
        with (out_dir / f"threshold_summary_width{m}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "threshold", "tau"])
            w.writeheader()
            w.writerows(thr_rows)

    with (out_dir / "width_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WIDTH_SUMMARY_FIELDNAMES)
        w.writeheader()
        w.writerows(width_summary_rows)

    print(json.dumps({"out_dir": str(out_dir), "seed": seed, "h": h, "n_steps": n_steps, "widths": widths}, indent=2))


def main() -> None:
    args = build_argparser().parse_args()
    widths = _parse_widths(args.widths)
    out_base = Path(args.out_dir)
    seeds_csv = args.seeds.strip() or None
    runs = resolve_run_dirs(out_base, seeds_csv, args.seed)
    for seed_val, run_dir in runs:
        run_one_seed(
            run_dir,
            seed_val,
            widths,
            args.n_over_m,
            args.alpha,
            args.sigma,
            args.teacher_scale,
            args.h_factor,
            args.T_phys,
            args.log_dt,
        )


if __name__ == "__main__":
    main()
