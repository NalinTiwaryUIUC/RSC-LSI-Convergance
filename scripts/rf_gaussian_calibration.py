#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from rf_width_common import design_matrix_from_bank, make_random_feature_bank


def _parse_widths(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _sqrtm_spd(a: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(a)
    evals = np.maximum(evals, 1e-14)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


def _inv_sqrtm_spd(a: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(a)
    evals = np.maximum(evals, 1e-14)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def _w2_gaussian(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    dm = mu1 - mu2
    s2sqrt = _sqrtm_spd(s2)
    mid = s2sqrt @ s1 @ s2sqrt
    mid_sqrt = _sqrtm_spd(mid)
    return float(np.dot(dm, dm) + np.trace(s1 + s2 - 2.0 * mid_sqrt))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RF Gaussian calibration experiment.")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--p", type=int, default=20)
    p.add_argument("--widths", type=str, default="32,64,128,256,512")
    p.add_argument("--m-max", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--T-phys", type=float, default=20.0)
    p.add_argument("--h-factor", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dt", type=float, default=0.02)
    p.add_argument("--out-dir", type=str, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    widths = _parse_widths(args.widths)
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.n, args.p))
    bank = make_random_feature_bank(args.p, args.m_max, args.seed + 1)
    y = None

    rows = []
    for width in widths:
        phi = design_matrix_from_bank(x, bank, width)
        if y is None:
            theta_true = rng.normal(size=width)
            y = phi @ theta_true + args.sigma * rng.normal(size=args.n)
        alpha_i = args.alpha * np.eye(width)
        precision = alpha_i + (phi.T @ phi) / (args.sigma ** 2)
        sigma_post = np.linalg.inv(precision)
        mu_post = sigma_post @ (phi.T @ y) / (args.sigma ** 2)
        lbound = args.alpha + np.linalg.eigvalsh(precision)[-1]
        h = args.h_factor / lbound
        n_steps = int(np.ceil(args.T_phys / h))
        s_log = max(1, int(np.floor(args.log_dt / h)))

        # Independent chains
        chains = rng.normal(size=(args.chains, width))
        save_steps = []
        mu_errs, sigma_errs, w2s, times = [], [], [], []
        inv_sqrt = _inv_sqrtm_spd(sigma_post)
        for step in range(n_steps + 1):
            if step % s_log == 0 or step == n_steps:
                mu_hat = chains.mean(axis=0)
                xc = chains - mu_hat
                cov_hat = (xc.T @ xc) / max(args.chains - 1, 1)
                e_mu = float(np.linalg.norm(inv_sqrt @ (mu_hat - mu_post)))
                norm_cov = inv_sqrt @ cov_hat @ inv_sqrt
                e_sigma = float(np.linalg.norm(norm_cov - np.eye(width), ord="fro") / np.sqrt(width))
                w2 = _w2_gaussian(mu_hat, cov_hat, mu_post, sigma_post)
                mu_errs.append(e_mu)
                sigma_errs.append(e_sigma)
                w2s.append(w2)
                save_steps.append(step)
                times.append(min(step * h, args.T_phys))
            if step < n_steps:
                grad = (chains @ precision) - (phi.T @ y / (args.sigma ** 2))
                noise = rng.normal(size=chains.shape)
                chains = chains - h * grad + np.sqrt(2.0 * h) * noise

        with (out_dir / f"gaussian_errors_width{width}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["width", "step", "time", "E_mu", "E_sigma", "W2"])
            for st, tm, em, es, w2 in zip(save_steps, times, mu_errs, sigma_errs, w2s):
                w.writerow([width, st, tm, em, es, w2])
        rows.append(
            {
                "width": width,
                "final_E_mu": mu_errs[-1],
                "final_E_sigma": sigma_errs[-1],
                "final_W2": w2s[-1],
            }
        )

    with (out_dir / "gaussian_width_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["width", "final_E_mu", "final_E_sigma", "final_W2"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

