#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from rf_width_common import (
    apply_hinv_sqrt,
    design_matrix_from_bank,
    hessian_spectrum,
    logistic_grad,
    logistic_hessian,
    logistic_posterior_value_grad_hess,
    make_random_feature_bank,
    make_synthetic_binary_data,
    sigmoid,
    smoothness_bound,
    solve_map_newton,
)


COUPLED_FIELDNAMES = [
    "width",
    "pair_id",
    "step",
    "time",
    "D_euc",
    "D_euc_norm",
    "R_euc",
    "D_H",
    "R_H",
    "log_R_H",
    "D_logit",
    "R_logit",
    "D_prob",
    "U_a",
    "U_b",
    "U_gap_a",
    "U_gap_b",
    "grad_norm_a",
    "grad_norm_b",
    "theta_norm_a",
    "theta_norm_b",
    "nan_or_inf",
]

PAIR_SUMMARY_FIELDNAMES = [
    "width",
    "pair_id",
    "kappa_H_early",
    "kappa_H_mid",
    "kappa_H_full",
    "kappa_logit_early",
    "kappa_logit_mid",
    "kappa_logit_full",
    "tau_H_0p5",
    "tau_H_0p1",
    "tau_logit_0p5",
    "tau_logit_0p1",
    "final_R_H",
    "final_R_logit",
    "final_R_prob",
    "init_mode",
    "init_D_H",
    "init_D_euc",
    "init_D_logit",
    "init_D_prob",
    "init_r_H",
    "init_r_logit",
    "init_logit_ridge",
    "stable",
]

WIDTH_SUMMARY_FIELDNAMES = [
    "width",
    "m",
    "lambda_min_H",
    "lambda_med_H",
    "lambda_max_H",
    "condition_H",
    "L_bound",
    "h",
    "hL_bound",
    "median_kappa_H_early",
    "iqr_kappa_H_early",
    "median_kappa_H_mid",
    "iqr_kappa_H_mid",
    "median_kappa_logit_early",
    "iqr_kappa_logit_early",
    "median_kappa_logit_mid",
    "iqr_kappa_logit_mid",
    "median_tau_H_0p5",
    "iqr_tau_H_0p5",
    "median_tau_H_0p1",
    "iqr_tau_H_0p1",
    "median_tau_logit_0p5",
    "iqr_tau_logit_0p5",
    "median_tau_logit_0p1",
    "iqr_tau_logit_0p1",
    "median_final_R_H",
    "iqr_final_R_H",
    "median_final_R_logit",
    "iqr_final_R_logit",
    "num_stable_pairs",
]


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


def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-300)))


def _first_time_leq(times: np.ndarray, vals: np.ndarray, thr: float) -> float:
    idx = np.where(vals <= thr)[0]
    if idx.size == 0:
        return float("nan")
    return float(times[idx[0]])


def _fit_rate(times: np.ndarray, log_rh: np.ndarray, t0: float, t1: float) -> float:
    mask = (times >= t0) & (times <= t1) & np.isfinite(log_rh)
    if mask.sum() < 2:
        return float("nan")
    x = times[mask]
    y = log_rh[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0.0:
        return float("nan")
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(-slope)


def _median_iqr(vals: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return float(q50), float(q75 - q25)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random-feature logistic coupled ULA.")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--p", type=int, default=20)
    p.add_argument("--widths", type=str, default="32,64,128,256,512")
    p.add_argument("--m-max", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--teacher-scale", type=float, default=2.0)
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--T-phys", type=float, default=20.0)
    p.add_argument("--h-factor", type=float, default=0.05)
    p.add_argument("--init-hnorm-radius", type=float, default=5.0)
    p.add_argument("--init-mode", type=str, choices=["hessian", "logit"], default="hessian")
    p.add_argument("--init-logit-radius", type=float, default=1.0)
    p.add_argument("--init-logit-ridge", type=float, default=1e-4)
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dt", type=float, default=0.02)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--map-max-iter", type=int, default=200)
    p.add_argument("--map-tol", type=float, default=1e-8)
    return p


def _make_logit_init_delta(
    phi: np.ndarray,
    rng: np.random.Generator,
    radius: float,
    ridge: float,
) -> np.ndarray:
    n, m = phi.shape
    u = rng.normal(size=n)
    a = phi @ phi.T + ridge * np.eye(n)
    coeff = np.linalg.solve(a, u)
    delta = phi.T @ coeff
    d = np.linalg.norm(phi @ delta) / np.sqrt(n)
    if d <= 0.0:
        delta = rng.normal(size=m)
        d = np.linalg.norm(phi @ delta) / np.sqrt(n)
    return radius * delta / max(d, 1e-12)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    widths = _parse_widths(args.widths)
    if args.m_max < max(widths):
        raise ValueError("--m-max must be >= max(widths)")

    cfg = vars(args).copy()
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data = make_synthetic_binary_data(n=args.n, p=args.p, teacher_scale=args.teacher_scale, seed=args.seed)
    np.savez(
        out_dir / "data.npz",
        x=data.x,
        y=data.y,
        w_teacher=data.w_teacher,
        teacher_scale=np.array([data.teacher_scale], dtype=np.float64),
    )

    bank = make_random_feature_bank(p=args.p, m_max=args.m_max, seed=args.seed + 1)

    # Precompute all widths for global step size.
    phi_by_width: dict[int, np.ndarray] = {}
    lbound_by_width: dict[int, float] = {}
    for w in widths:
        phi = design_matrix_from_bank(data.x, bank, w)
        phi_by_width[w] = phi
        lbound_by_width[w] = smoothness_bound(phi, args.alpha)
        np.savez(out_dir / f"features_width{w}.npz", phi=phi, width=np.array([w], dtype=np.int64))

    lmax = max(lbound_by_width.values())
    h = float(args.h_factor / lmax)
    n_steps = int(np.ceil(args.T_phys / h))
    s_log = max(1, int(np.floor(args.log_dt / h)))

    width_summary_rows: list[dict[str, object]] = []
    master_pair_rows: list[dict[str, object]] = []

    for width in widths:
        phi = phi_by_width[width]
        map_res = solve_map_newton(
            phi=phi,
            y=data.y,
            alpha=args.alpha,
            max_iter=args.map_max_iter,
            tol=args.map_tol,
        )
        h_map = logistic_hessian(map_res.theta_map, phi, args.alpha)
        spec = hessian_spectrum(h_map)
        np.savez(
            out_dir / f"map_width{width}.npz",
            theta_map=map_res.theta_map,
            objective=np.array([map_res.objective], dtype=np.float64),
            grad_norm=np.array([map_res.grad_norm], dtype=np.float64),
            converged=np.array([int(map_res.converged)], dtype=np.int64),
            n_iter=np.array([map_res.n_iter], dtype=np.int64),
        )
        np.savez(
            out_dir / f"hessian_width{width}.npz",
            hessian=h_map,
            lambda_min=np.array([spec.lambda_min], dtype=np.float64),
            lambda_med=np.array([spec.lambda_med], dtype=np.float64),
            lambda_max=np.array([spec.lambda_max], dtype=np.float64),
            condition=np.array([spec.condition], dtype=np.float64),
            l_bound=np.array([lbound_by_width[width]], dtype=np.float64),
        )

        rng = np.random.default_rng(args.seed + 1000 + width)
        coupled_rows: list[dict[str, object]] = []
        pair_rows: list[dict[str, object]] = []
        init_delta_rows = []
        for pair_id in range(args.pairs):
            if args.init_mode == "hessian":
                z = rng.normal(size=width)
                v = apply_hinv_sqrt(h_map, z)
                denom = float(np.sqrt(v @ h_map @ v))
                delta = args.init_hnorm_radius * v / max(denom, 1e-12)
            else:
                delta = _make_logit_init_delta(
                    phi=phi,
                    rng=rng,
                    radius=args.init_logit_radius,
                    ridge=args.init_logit_ridge,
                )
            theta_a = map_res.theta_map + delta
            theta_b = map_res.theta_map - delta

            base_delta = theta_a - theta_b
            d_euc0 = float(np.linalg.norm(base_delta))
            d_h0 = float(np.sqrt(base_delta @ h_map @ base_delta))
            d_logit0 = float(np.linalg.norm(phi @ base_delta) / np.sqrt(phi.shape[0]))
            p_a0 = sigmoid(phi @ theta_a)
            p_b0 = sigmoid(phi @ theta_b)
            d_prob0 = float(np.linalg.norm(p_a0 - p_b0) / np.sqrt(phi.shape[0]))

            init_delta_rows.append({"pair_id": pair_id, **{f"delta_{i}": base_delta[i] for i in range(width)}})

            times, rh_vals, rlogit_vals, rprob_vals = [], [], [], []
            stable = True
            for step in range(n_steps + 1):
                t_phys = min(step * h, args.T_phys)
                delta_ab = theta_a - theta_b
                logits_diff = phi @ delta_ab
                p_a = sigmoid(phi @ theta_a)
                p_b = sigmoid(phi @ theta_b)
                d_euc = float(np.linalg.norm(delta_ab))
                d_h = float(np.sqrt(max(delta_ab @ h_map @ delta_ab, 0.0)))
                d_logit = float(np.linalg.norm(logits_diff) / np.sqrt(phi.shape[0]))
                d_prob = float(np.linalg.norm(p_a - p_b) / np.sqrt(phi.shape[0]))

                u_a, grad_a, _ = logistic_posterior_value_grad_hess(theta_a, phi, data.y, args.alpha)
                u_b, grad_b, _ = logistic_posterior_value_grad_hess(theta_b, phi, data.y, args.alpha)
                nan_or_inf = int(
                    (not np.isfinite(theta_a).all())
                    or (not np.isfinite(theta_b).all())
                    or (not np.isfinite(grad_a).all())
                    or (not np.isfinite(grad_b).all())
                )
                if nan_or_inf:
                    stable = False

                if step % s_log == 0 or step == n_steps:
                    row = {
                        "width": width,
                        "pair_id": pair_id,
                        "step": step,
                        "time": t_phys,
                        "D_euc": d_euc,
                        "D_euc_norm": d_euc / np.sqrt(width),
                        "R_euc": d_euc / max(d_euc0, 1e-12),
                        "D_H": d_h,
                        "R_H": d_h / max(d_h0, 1e-12),
                        "log_R_H": _safe_log(d_h / max(d_h0, 1e-12)),
                        "D_logit": d_logit,
                        "R_logit": d_logit / max(d_logit0, 1e-12),
                        "D_prob": d_prob,
                        "U_a": u_a,
                        "U_b": u_b,
                        "U_gap_a": u_a - map_res.objective,
                        "U_gap_b": u_b - map_res.objective,
                        "grad_norm_a": float(np.linalg.norm(grad_a)),
                        "grad_norm_b": float(np.linalg.norm(grad_b)),
                        "theta_norm_a": float(np.linalg.norm(theta_a)),
                        "theta_norm_b": float(np.linalg.norm(theta_b)),
                        "nan_or_inf": nan_or_inf,
                    }
                    coupled_rows.append(row)
                    times.append(t_phys)
                    rh_vals.append(row["R_H"])
                    rlogit_vals.append(row["R_logit"])
                    rprob_vals.append(row["D_prob"] / max(d_prob0, 1e-12))

                if step < n_steps:
                    xi = rng.normal(size=width)
                    theta_a = theta_a - h * logistic_grad(theta_a, phi, data.y, args.alpha) + np.sqrt(2.0 * h) * xi
                    theta_b = theta_b - h * logistic_grad(theta_b, phi, data.y, args.alpha) + np.sqrt(2.0 * h) * xi

            times_a = np.asarray(times, dtype=np.float64)
            rh_a = np.asarray(rh_vals, dtype=np.float64)
            lrh_a = np.log(np.maximum(rh_a, 1e-300))
            rlogit_a = np.asarray(rlogit_vals, dtype=np.float64)
            lrlogit_a = np.log(np.maximum(rlogit_a, 1e-300))
            row_pair = {
                "width": width,
                "pair_id": pair_id,
                "kappa_H_early": _fit_rate(times_a, lrh_a, 0.0, min(2.0, args.T_phys)),
                "kappa_H_mid": _fit_rate(times_a, lrh_a, 2.0, min(10.0, args.T_phys)),
                "kappa_H_full": _fit_rate(times_a, lrh_a, 0.0, min(10.0, args.T_phys)),
                "kappa_logit_early": _fit_rate(times_a, lrlogit_a, 0.0, min(2.0, args.T_phys)),
                "kappa_logit_mid": _fit_rate(times_a, lrlogit_a, 2.0, min(10.0, args.T_phys)),
                "kappa_logit_full": _fit_rate(times_a, lrlogit_a, 0.0, min(10.0, args.T_phys)),
                "tau_H_0p5": _first_time_leq(times_a, rh_a, 0.5),
                "tau_H_0p1": _first_time_leq(times_a, rh_a, 0.1),
                "tau_logit_0p5": _first_time_leq(times_a, rlogit_a, 0.5),
                "tau_logit_0p1": _first_time_leq(times_a, rlogit_a, 0.1),
                "final_R_H": float(rh_a[-1]) if rh_a.size else float("nan"),
                "final_R_logit": float(rlogit_a[-1]) if rlogit_vals else float("nan"),
                "final_R_prob": float(np.asarray(rprob_vals)[-1]) if rprob_vals else float("nan"),
                "init_mode": args.init_mode,
                "init_D_H": d_h0,
                "init_D_euc": d_euc0,
                "init_D_logit": d_logit0,
                "init_D_prob": d_prob0,
                "init_r_H": args.init_hnorm_radius if args.init_mode == "hessian" else float("nan"),
                "init_r_logit": args.init_logit_radius if args.init_mode == "logit" else float("nan"),
                "init_logit_ridge": args.init_logit_ridge if args.init_mode == "logit" else float("nan"),
                "stable": int(stable),
            }
            pair_rows.append(row_pair)
            master_pair_rows.append(row_pair)

        np.savez(
            out_dir / f"init_deltas_width{width}.npz",
            deltas=np.array([[r[f"delta_{i}"] for i in range(width)] for r in init_delta_rows], dtype=np.float64),
            pair_ids=np.array([r["pair_id"] for r in init_delta_rows], dtype=np.int64),
            h=np.array([h], dtype=np.float64),
        )

        with (out_dir / f"coupled_metrics_width{width}.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=COUPLED_FIELDNAMES)
            wtr.writeheader()
            wtr.writerows(coupled_rows)
        with (out_dir / f"pair_summary_width{width}.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=PAIR_SUMMARY_FIELDNAMES)
            wtr.writeheader()
            wtr.writerows(pair_rows)

        med_ke, iqr_ke = _median_iqr([r["kappa_H_early"] for r in pair_rows])
        med_km, iqr_km = _median_iqr([r["kappa_H_mid"] for r in pair_rows])
        med_kle, iqr_kle = _median_iqr([r["kappa_logit_early"] for r in pair_rows])
        med_klm, iqr_klm = _median_iqr([r["kappa_logit_mid"] for r in pair_rows])
        med_t5, iqr_t5 = _median_iqr([r["tau_H_0p5"] for r in pair_rows])
        med_t1, iqr_t1 = _median_iqr([r["tau_H_0p1"] for r in pair_rows])
        med_tl5, iqr_tl5 = _median_iqr([r["tau_logit_0p5"] for r in pair_rows])
        med_tl1, iqr_tl1 = _median_iqr([r["tau_logit_0p1"] for r in pair_rows])
        med_frh, iqr_frh = _median_iqr([r["final_R_H"] for r in pair_rows])
        med_frl, iqr_frl = _median_iqr([r["final_R_logit"] for r in pair_rows])
        width_summary_rows.append(
            {
                "width": width,
                "m": width,
                "lambda_min_H": spec.lambda_min,
                "lambda_med_H": spec.lambda_med,
                "lambda_max_H": spec.lambda_max,
                "condition_H": spec.condition,
                "L_bound": lbound_by_width[width],
                "h": h,
                "hL_bound": h * lbound_by_width[width],
                "median_kappa_H_early": med_ke,
                "iqr_kappa_H_early": iqr_ke,
                "median_kappa_H_mid": med_km,
                "iqr_kappa_H_mid": iqr_km,
                "median_kappa_logit_early": med_kle,
                "iqr_kappa_logit_early": iqr_kle,
                "median_kappa_logit_mid": med_klm,
                "iqr_kappa_logit_mid": iqr_klm,
                "median_tau_H_0p5": med_t5,
                "iqr_tau_H_0p5": iqr_t5,
                "median_tau_H_0p1": med_t1,
                "iqr_tau_H_0p1": iqr_t1,
                "median_tau_logit_0p5": med_tl5,
                "iqr_tau_logit_0p5": iqr_tl5,
                "median_tau_logit_0p1": med_tl1,
                "iqr_tau_logit_0p1": iqr_tl1,
                "median_final_R_H": med_frh,
                "iqr_final_R_H": iqr_frh,
                "median_final_R_logit": med_frl,
                "iqr_final_R_logit": iqr_frl,
                "num_stable_pairs": int(sum(int(r["stable"]) for r in pair_rows)),
            }
        )

    with (out_dir / "width_summary.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=WIDTH_SUMMARY_FIELDNAMES)
        wtr.writeheader()
        wtr.writerows(width_summary_rows)

    # Lightweight markdown summary.
    md_lines = [
        "# RF logistic coupling summary",
        "",
        f"- n={args.n}, p={args.p}, alpha={args.alpha}, pairs={args.pairs}",
        f"- widths={widths}",
        f"- init_mode={args.init_mode}",
        f"- T_phys={args.T_phys}, h={h:.6g}, n_steps={n_steps}, s_log={s_log}",
        "",
        "## Width summary",
        "",
        "| width | hL_bound | med_kappa_H_early | med_kappa_logit_early | med_tau_H_0p5 | med_tau_logit_0p5 | med_final_R_H | med_final_R_logit | stable_pairs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in width_summary_rows:
        md_lines.append(
            "| {width} | {hL_bound:.4g} | {median_kappa_H_early:.4g} | {median_kappa_logit_early:.4g} | "
            "{median_tau_H_0p5:.4g} | {median_tau_logit_0p5:.4g} | {median_final_R_H:.4g} | "
            "{median_final_R_logit:.4g} | {num_stable_pairs} |".format(**row)
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    # Aggregate pair summary for convenience.
    with (out_dir / "pair_summary_all_widths.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=PAIR_SUMMARY_FIELDNAMES)
        wtr.writeheader()
        wtr.writerows(master_pair_rows)

    print(json.dumps({"out_dir": str(out_dir), "widths": widths, "h": h, "n_steps": n_steps}, indent=2))


if __name__ == "__main__":
    main()

