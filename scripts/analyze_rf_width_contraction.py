#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def _median_iqr(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return float(q50), float(q75 - q25)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze RF coupling outputs.")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--runs-dir", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--out-md", type=str, default="")
    return p


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xm, ym = x[mask], y[mask]
    sx, sy = float(np.std(xm)), float(np.std(ym))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(xm, ym)[0, 1])


def _quadratic_log_rh(hess: np.ndarray, delta0: np.ndarray, h: float, times: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(hess)
    evals = np.maximum(evals, 1e-14)
    y = evecs.T @ delta0
    k = np.maximum(np.round(times / h).astype(np.int64), 0)
    d0 = float(np.sqrt(np.sum(evals * (y**2))))
    out = np.empty_like(times, dtype=np.float64)
    base = np.maximum(np.abs(1.0 - h * evals), 1e-14)
    for i, ki in enumerate(k):
        fac2k = base ** (2 * ki)
        d = np.sqrt(np.sum(evals * fac2k * (y**2)))
        out[i] = np.log(max(d / max(d0, 1e-300), 1e-300))
    return out


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = Path(args.run_dir or args.runs_dir)
    if not run_dir.as_posix():
        raise ValueError("Provide --run-dir (or --runs-dir for backward compatibility).")
    pair_path = run_dir / "pair_summary_all_widths.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing {pair_path}")

    rows = []
    with pair_path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)

    widths = sorted({int(r["width"]) for r in rows})
    cfg_path = run_dir / "config.yaml"
    prior_alpha = 0.3
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict) and "alpha" in cfg:
            prior_alpha = float(cfg["alpha"])

    out_rows = []
    hess_rows = []
    quad_summary_rows = []
    for w in widths:
        sub = [r for r in rows if int(r["width"]) == w]
        k_early = np.array([float(r["kappa_H_early"]) for r in sub], dtype=np.float64)
        k_mid = np.array([float(r["kappa_H_mid"]) for r in sub], dtype=np.float64)
        kl_early = np.array([float(r.get("kappa_logit_early", "nan")) for r in sub], dtype=np.float64)
        kl_mid = np.array([float(r.get("kappa_logit_mid", "nan")) for r in sub], dtype=np.float64)
        t05 = np.array([float(r["tau_H_0p5"]) for r in sub], dtype=np.float64)
        t01 = np.array([float(r["tau_H_0p1"]) for r in sub], dtype=np.float64)
        tl05 = np.array([float(r.get("tau_logit_0p5", "nan")) for r in sub], dtype=np.float64)
        tl01 = np.array([float(r.get("tau_logit_0p1", "nan")) for r in sub], dtype=np.float64)
        frh = np.array([float(r["final_R_H"]) for r in sub], dtype=np.float64)
        frl = np.array([float(r["final_R_logit"]) for r in sub], dtype=np.float64)
        stable = sum(int(r["stable"]) for r in sub)
        med_ke, iqr_ke = _median_iqr(k_early)
        med_km, iqr_km = _median_iqr(k_mid)
        med_kle, iqr_kle = _median_iqr(kl_early)
        med_klm, iqr_klm = _median_iqr(kl_mid)
        med_t05, iqr_t05 = _median_iqr(t05)
        med_t01, iqr_t01 = _median_iqr(t01)
        med_tl05, iqr_tl05 = _median_iqr(tl05)
        med_tl01, iqr_tl01 = _median_iqr(tl01)
        med_frh, iqr_frh = _median_iqr(frh)
        med_frl, iqr_frl = _median_iqr(frl)
        out_rows.append(
            {
                "width": w,
                "pairs": len(sub),
                "stable_pairs": stable,
                "median_kappa_H_early": med_ke,
                "iqr_kappa_H_early": iqr_ke,
                "median_kappa_H_mid": med_km,
                "iqr_kappa_H_mid": iqr_km,
                "median_kappa_logit_early": med_kle,
                "iqr_kappa_logit_early": iqr_kle,
                "median_kappa_logit_mid": med_klm,
                "iqr_kappa_logit_mid": iqr_klm,
                "median_tau_H_0p5": med_t05,
                "iqr_tau_H_0p5": iqr_t05,
                "median_tau_H_0p1": med_t01,
                "iqr_tau_H_0p1": iqr_t01,
                "median_tau_logit_0p5": med_tl05,
                "iqr_tau_logit_0p5": iqr_tl05,
                "median_tau_logit_0p1": med_tl01,
                "iqr_tau_logit_0p1": iqr_tl01,
                "median_final_R_H": med_frh,
                "iqr_final_R_H": iqr_frh,
                "median_final_R_logit": med_frl,
                "iqr_final_R_logit": iqr_frl,
            }
        )

        # Hessian spectrum summary.
        hess_path = run_dir / f"hessian_width{w}.npz"
        if hess_path.exists():
            dat = np.load(hess_path)
            hess = dat["hessian"]
            evals = np.linalg.eigvalsh(hess)
            hess_rows.append(
                {
                    "seed": run_dir.name,
                    "width": w,
                    "m": w,
                    "lambda_min": float(np.min(evals)),
                    "lambda_p05": float(np.quantile(evals, 0.05)),
                    "lambda_p25": float(np.quantile(evals, 0.25)),
                    "lambda_median": float(np.median(evals)),
                    "lambda_p75": float(np.quantile(evals, 0.75)),
                    "lambda_p95": float(np.quantile(evals, 0.95)),
                    "lambda_max": float(np.max(evals)),
                    "trace_over_m": float(np.trace(hess) / w),
                    "condition_number": float(np.max(evals) / max(np.min(evals), 1e-14)),
                    "prior_alpha": prior_alpha,
                    "frac_prior_0p05": float(np.mean(evals <= prior_alpha * 1.05)),
                    "frac_prior_0p10": float(np.mean(evals <= prior_alpha * 1.10)),
                    "frac_prior_0p25": float(np.mean(evals <= prior_alpha * 1.25)),
                }
            )

        # Quadratic-vs-empirical contraction.
        init_path = run_dir / f"init_deltas_width{w}.npz"
        cm_path = run_dir / f"coupled_metrics_width{w}.csv"
        if hess_path.exists() and init_path.exists() and cm_path.exists():
            hess = np.load(hess_path)["hessian"]
            init = np.load(init_path)
            deltas = init["deltas"]
            h_step = float(init["h"][0])
            by_pair = {}
            with cm_path.open() as f:
                rd = csv.DictReader(f)
                for r in rd:
                    pid = int(r["pair_id"])
                    by_pair.setdefault(pid, {"time": [], "log_rh_emp": []})
                    by_pair[pid]["time"].append(float(r["time"]))
                    by_pair[pid]["log_rh_emp"].append(float(np.log(max(float(r["R_H"]), 1e-300))))
            qrows = []
            med_abs, fin_abs, max_abs, corrs = [], [], [], []
            for pid, rec in sorted(by_pair.items()):
                t = np.asarray(rec["time"], dtype=np.float64)
                emp = np.asarray(rec["log_rh_emp"], dtype=np.float64)
                order = np.argsort(t)
                t, emp = t[order], emp[order]
                quad = _quadratic_log_rh(hess, deltas[pid], h_step, t)
                delta = emp - quad
                med_abs.append(float(np.median(np.abs(delta))))
                fin_abs.append(float(np.abs(delta[-1])))
                max_abs.append(float(np.max(np.abs(delta))))
                corrs.append(_corr(emp, quad))
                for ti, ei, qi, di in zip(t, emp, quad, delta):
                    qrows.append(
                        {
                            "width": w,
                            "pair_id": pid,
                            "time": ti,
                            "log_R_H_emp": ei,
                            "log_R_H_quad": qi,
                            "delta_quad": di,
                        }
                    )
            qf = run_dir / f"quadratic_contraction_width{w}.csv"
            with qf.open("w", newline="") as f:
                wtr = csv.DictWriter(
                    f, fieldnames=["width", "pair_id", "time", "log_R_H_emp", "log_R_H_quad", "delta_quad"]
                )
                wtr.writeheader()
                wtr.writerows(qrows)
            quad_summary_rows.append(
                {
                    "width": w,
                    "median_abs_delta_quad": float(np.nanmedian(med_abs)),
                    "final_abs_delta_quad": float(np.nanmedian(fin_abs)),
                    "max_abs_delta_quad": float(np.nanmedian(max_abs)),
                    "corr_emp_quad_log_R_H": float(np.nanmedian(np.asarray(corrs, dtype=np.float64))),
                }
            )

    base_out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_csv = Path(args.out_csv) if args.out_csv else (base_out_dir / "rf_contraction_summary.csv")
    out_md = Path(args.out_md) if args.out_md else (base_out_dir / "rf_contraction_summary.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    lines = [
        "# RF contraction summary",
        "",
        "| width | stable/pairs | median kappa early | median kappa mid | median tau 0.5 | median tau 0.1 | median final R_H | median final R_logit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        lines.append(
            f"| {r['width']} | {r['stable_pairs']}/{r['pairs']} | {r['median_kappa_H_early']:.4g} | "
            f"{r['median_kappa_H_mid']:.4g} | {r['median_tau_H_0p5']:.4g} | {r['median_tau_H_0p1']:.4g} | "
            f"{r['median_final_R_H']:.4g} | {r['median_final_R_logit']:.4g} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")

    if hess_rows:
        hess_csv = base_out_dir / "hessian_spectrum_summary.csv"
        with hess_csv.open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(hess_rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(hess_rows)
        print(f"Wrote {hess_csv}")

    if quad_summary_rows:
        quad_csv = base_out_dir / "quadratic_contraction_summary.csv"
        with quad_csv.open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(quad_summary_rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(quad_summary_rows)
        print(f"Wrote {quad_csv}")


if __name__ == "__main__":
    main()

