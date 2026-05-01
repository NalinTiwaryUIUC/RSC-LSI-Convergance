import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bayeslin_lsi_common import (  # noqa: E402
    build_posterior_precision,
    build_rhs_b,
    convergence_metrics_row,
    first_time_leq,
    fit_rate,
    generate_linear_regression_data,
    global_step_size,
    grad_U,
    grad_U_via_star,
    iterate_gd_closed_form,
    potential_U,
    spectrum_summary,
    theta_star_from_normal_eqs,
)


class TestPosteriorAndGradient(unittest.TestCase):
    def test_H_and_theta_star_residual(self):
        rng = np.random.default_rng(0)
        m, n, alpha, sigma = 6, 24, 0.3, 1.0
        X = rng.normal(size=(n, m)).astype(np.float64)
        theta_true = rng.normal(size=m)
        y = X @ theta_true + sigma * rng.normal(size=n)
        H = build_posterior_precision(X, alpha, sigma)
        H_ref = (1.0 / sigma**2) * (X.T @ X) + alpha * np.eye(m)
        np.testing.assert_allclose(H, H_ref, rtol=1e-12)
        b = build_rhs_b(X, y, sigma)
        ts = theta_star_from_normal_eqs(H, b)
        res = np.linalg.norm(H @ ts - b)
        self.assertLess(res, 1e-10)

    def test_gradient_identity(self):
        rng = np.random.default_rng(1)
        m, n = 5, 20
        X = rng.normal(size=(n, m))
        y = rng.normal(size=n)
        H = build_posterior_precision(X, 0.3, 1.0)
        b = build_rhs_b(X, y, 1.0)
        ts = theta_star_from_normal_eqs(H, b)
        theta = rng.normal(size=m)
        g1 = grad_U(theta, H, b)
        g2 = grad_U_via_star(theta, ts, H)
        np.testing.assert_allclose(g1, g2, rtol=1e-12)

    def test_gd_matches_linear_iteration_diagonal(self):
        m = 4
        evals = np.array([1.0, 2.0, 3.0, 4.0])
        H = np.diag(evals)
        b = np.array([1.0, 0.0, 0.0, 0.0])
        ts = theta_star_from_normal_eqs(H, b)
        theta0 = np.zeros(m)
        h = 0.01
        n_steps = 50
        theta = theta0.copy()
        for _ in range(n_steps):
            theta = theta - h * grad_U(theta, H, b)
        theta_cf = iterate_gd_closed_form(theta0, ts, H, h, n_steps)
        np.testing.assert_allclose(theta, theta_cf, rtol=1e-10, atol=1e-10)

    def test_gd_matches_linear_iteration_spd(self):
        rng = np.random.default_rng(2)
        m = 5
        A = rng.normal(size=(m, m))
        H = A.T @ A + 0.5 * np.eye(m)
        b = rng.normal(size=m)
        ts = theta_star_from_normal_eqs(H, b)
        theta0 = rng.normal(size=m)
        h = 0.02 / np.linalg.eigvalsh(H)[-1]
        n_steps = 30
        theta = theta0.copy()
        for _ in range(n_steps):
            theta = theta - h * grad_U(theta, H, b)
        theta_cf = iterate_gd_closed_form(theta0, ts, H, h, n_steps)
        np.testing.assert_allclose(theta, theta_cf, rtol=1e-9, atol=1e-9)

    def test_objective_gap_quadratic(self):
        rng = np.random.default_rng(3)
        m, n, alpha, sigma = 7, 28, 0.3, 1.0
        X = rng.normal(size=(n, m))
        y = rng.normal(size=n)
        H = build_posterior_precision(X, alpha, sigma)
        b = build_rhs_b(X, y, sigma)
        ts = theta_star_from_normal_eqs(H, b)
        theta = rng.normal(size=m)
        gap = potential_U(theta, X, y, alpha, sigma) - potential_U(ts, X, y, alpha, sigma)
        d = theta - ts
        quad = 0.5 * float(d @ H @ d)
        self.assertAlmostEqual(gap, quad, places=8)

    def test_prediction_metric(self):
        rng = np.random.default_rng(4)
        m, n, alpha, sigma = 4, 16, 0.3, 1.0
        X, y, _ = generate_linear_regression_data(m, 4, alpha, sigma, 1.0, rng)
        H = build_posterior_precision(X, alpha, sigma)
        b = build_rhs_b(X, y, sigma)
        ts = theta_star_from_normal_eqs(H, b)
        theta = rng.normal(size=m)
        d = theta - ts
        d_pred = float(np.linalg.norm(X @ d) / np.sqrt(n))
        delta0 = -ts
        d0e = float(np.linalg.norm(delta0))
        d0h = float(np.sqrt(max(delta0 @ H @ delta0, 0.0)))
        d0p = float(np.linalg.norm(X @ delta0) / np.sqrt(n))
        u0g = float(potential_U(np.zeros(m), X, y, alpha, sigma) - potential_U(ts, X, y, alpha, sigma))
        row = convergence_metrics_row(theta, ts, X, H, alpha, sigma, y, d0e, d0h, d0p, u0g)
        e_pred = d_pred / max(d0p, 1e-300)
        self.assertAlmostEqual(row["D_pred"], d_pred, places=10)
        self.assertAlmostEqual(row["e_pred"], e_pred, places=10)

    def test_global_h_bound(self):
        lam = {8: 10.0, 16: 4.0}
        h = global_step_size(lam, h_factor=0.05)
        for w, lv in lam.items():
            self.assertLessEqual(h * lv, 0.05 + 1e-12)

    def test_fit_rate_on_exponential(self):
        rho_true = 0.7
        t = np.linspace(0.0, 5.0, 50)
        log_e = -rho_true * t + 0.1
        r = fit_rate(t, log_e, 0.0, 5.0)
        self.assertAlmostEqual(r, rho_true, places=2)

    def test_first_time_leq(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([1.0, 0.8, 0.4, 0.09])
        self.assertAlmostEqual(first_time_leq(t, v, 0.5), 2.0)
        self.assertAlmostEqual(first_time_leq(t, v, 0.1), 3.0)


class TestDataGeneration(unittest.TestCase):
    def test_spectrum_keys(self):
        rng = np.random.default_rng(0)
        X, y, _ = generate_linear_regression_data(10, 4, 0.3, 1.0, 1.0, rng)
        H = build_posterior_precision(X, 0.3, 1.0)
        s = spectrum_summary(H)
        self.assertIn("C_LSI", s)
        self.assertEqual(s["C_LSI"], s["C_PI"])


class TestRunnerSmoke(unittest.TestCase):
    def test_convergence_runner_outputs(self):
        tmp = Path(tempfile.mkdtemp(prefix="bayeslin_smoke_"))
        out = tmp / "run"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bayeslin_lsi_width_convergence.py"),
                "--widths",
                "8,16",
                "--n-over-m",
                "4",
                "--T-phys",
                "0.5",
                "--log-dt",
                "0.05",
                "--seed",
                "7",
                "--out-dir",
                str(out),
            ],
            cwd=str(ROOT),
            check=True,
        )
        for m in (8, 16):
            self.assertTrue((out / f"convergence_metrics_width{m}.csv").is_file())
            self.assertTrue((out / f"spectrum_width{m}.npz").is_file())
            self.assertTrue((out / f"posterior_width{m}.npz").is_file())
            self.assertTrue((out / f"rate_summary_width{m}.csv").is_file())
            self.assertTrue((out / f"threshold_summary_width{m}.csv").is_file())
        with (out / "width_summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(int(r["stable"]), 1)
            self.assertLess(float(r["h_lambda_max"]), 1.0)

    def test_multi_seed_directories(self):
        tmp = Path(tempfile.mkdtemp(prefix="bayeslin_multi_"))
        base = tmp / "pilot"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bayeslin_lsi_width_convergence.py"),
                "--widths",
                "8",
                "--T-phys",
                "0.2",
                "--log-dt",
                "0.04",
                "--seeds",
                "0,1",
                "--out-dir",
                str(base),
            ],
            cwd=str(ROOT),
            check=True,
        )
        self.assertTrue((tmp / "pilot_seed0" / "width_summary.csv").is_file())
        self.assertTrue((tmp / "pilot_seed1" / "width_summary.csv").is_file())


class TestPlotSmoke(unittest.TestCase):
    def test_plot_script_writes_pdfs(self):
        tmp = Path(tempfile.mkdtemp(prefix="bayeslin_plot_"))
        for seed in (0, 1):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "bayeslin_lsi_width_convergence.py"),
                    "--widths",
                    "8,16",
                    "--T-phys",
                    "0.6",
                    "--log-dt",
                    "0.05",
                    "--seed",
                    str(seed),
                    "--out-dir",
                    str(tmp / f"run_seed{seed}"),
                ],
                cwd=str(ROOT),
                check=True,
            )
        # glob must match both dirs
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_bayeslin_lsi_width.py"),
                "--run-glob",
                str(tmp / "run_seed*"),
                "--plot-out-dir",
                str(tmp / "plots_out"),
            ],
            cwd=str(ROOT),
            check=True,
        )
        plot_dir = tmp / "plots_out" / "plots"
        for name in (
            "log_e_euc_vs_time_by_width.pdf",
            "log_e_H_vs_time_by_width.pdf",
            "log_e_pred_vs_time_by_width.pdf",
            "rate_vs_width_with_lambda_min.pdf",
            "C_LSI_vs_width.pdf",
            "rate_vs_inv_C_LSI.pdf",
            "tau_H_vs_width.pdf",
            "H_spectrum_by_width.pdf",
        ):
            self.assertTrue((plot_dir / name).is_file(), msg=f"missing {name}")


if __name__ == "__main__":
    unittest.main()
