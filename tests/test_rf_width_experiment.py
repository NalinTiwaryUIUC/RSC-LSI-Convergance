import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rf_width_common import (  # noqa: E402
    apply_hinv_sqrt,
    design_matrix_from_bank,
    logistic_grad,
    logistic_posterior_value_grad_hess,
    make_random_feature_bank,
    make_synthetic_binary_data,
    smoothness_bound,
    solve_map_newton,
)


class TestRFCommon(unittest.TestCase):
    def test_nested_features_prefix(self):
        data = make_synthetic_binary_data(n=32, p=8, teacher_scale=2.0, seed=1)
        bank = make_random_feature_bank(p=8, m_max=16, seed=2)
        phi16 = design_matrix_from_bank(data.x, bank, 16)
        phi8 = design_matrix_from_bank(data.x, bank, 8)
        np.testing.assert_allclose(phi16[:, :8], phi8 * np.sqrt(8.0 / 16.0), atol=1e-10)

    def test_gradient_matches_finite_difference(self):
        data = make_synthetic_binary_data(n=20, p=6, teacher_scale=1.5, seed=7)
        bank = make_random_feature_bank(p=6, m_max=12, seed=8)
        phi = design_matrix_from_bank(data.x, bank, 10)
        theta = np.linspace(-0.2, 0.3, 10)
        alpha = 0.3
        u, grad, _ = logistic_posterior_value_grad_hess(theta, phi, data.y, alpha)
        self.assertTrue(np.isfinite(u))
        eps = 1e-6
        fd = np.zeros_like(theta)
        for i in range(theta.size):
            e = np.zeros_like(theta)
            e[i] = eps
            u_p, _, _ = logistic_posterior_value_grad_hess(theta + e, phi, data.y, alpha)
            u_m, _, _ = logistic_posterior_value_grad_hess(theta - e, phi, data.y, alpha)
            fd[i] = (u_p - u_m) / (2.0 * eps)
        np.testing.assert_allclose(grad, fd, rtol=5e-4, atol=5e-4)

    def test_map_hessian_and_hinv(self):
        data = make_synthetic_binary_data(n=40, p=10, teacher_scale=2.0, seed=3)
        bank = make_random_feature_bank(p=10, m_max=20, seed=4)
        phi = design_matrix_from_bank(data.x, bank, 16)
        res = solve_map_newton(phi=phi, y=data.y, alpha=0.3, max_iter=80, tol=1e-7)
        self.assertTrue(np.isfinite(res.grad_norm))
        self.assertLess(res.grad_norm, 1e-4)
        _, _, h = logistic_posterior_value_grad_hess(res.theta_map, phi, data.y, 0.3)
        evals = np.linalg.eigvalsh(h)
        self.assertGreater(float(evals.min()), 0.0)
        z = np.random.default_rng(0).normal(size=16)
        v = apply_hinv_sqrt(h, z)
        self.assertTrue(np.isfinite(v).all())
        self.assertAlmostEqual(float(v @ h @ v), float(z @ z), places=5)
        self.assertGreater(smoothness_bound(phi, 0.3), 0.3)

    def test_same_noise_coupling_delta_identity(self):
        data = make_synthetic_binary_data(n=24, p=6, teacher_scale=2.0, seed=11)
        bank = make_random_feature_bank(p=6, m_max=12, seed=12)
        phi = design_matrix_from_bank(data.x, bank, 8)
        alpha = 0.3
        h = 1e-3
        rng = np.random.default_rng(0)
        a = rng.normal(size=8)
        b = rng.normal(size=8)
        xi = rng.normal(size=8)
        a_next = a - h * logistic_grad(a, phi, data.y, alpha) + np.sqrt(2 * h) * xi
        b_next = b - h * logistic_grad(b, phi, data.y, alpha) + np.sqrt(2 * h) * xi
        lhs = a_next - b_next
        rhs = (a - b) - h * (logistic_grad(a, phi, data.y, alpha) - logistic_grad(b, phi, data.y, alpha))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


class TestRFRunnerSmoke(unittest.TestCase):
    def test_smoke_run_outputs_and_schema(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_width_smoke_"))
        out = tmp / "run"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "rf_logistic_coupling.py"),
            "--n",
            "64",
            "--p",
            "10",
            "--widths",
            "8,16",
            "--m-max",
            "16",
            "--pairs",
            "2",
            "--T-phys",
            "0.2",
            "--h-factor",
            "0.05",
            "--log-dt",
            "0.05",
            "--seed",
            "0",
            "--out-dir",
            str(out),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        expected = [
            "config.yaml",
            "data.npz",
            "features_width8.npz",
            "features_width16.npz",
            "map_width8.npz",
            "map_width16.npz",
            "hessian_width8.npz",
            "hessian_width16.npz",
            "coupled_metrics_width8.csv",
            "coupled_metrics_width16.csv",
            "pair_summary_width8.csv",
            "pair_summary_width16.csv",
            "pair_summary_all_widths.csv",
            "width_summary.csv",
            "summary.md",
        ]
        for rel in expected:
            self.assertTrue((out / rel).is_file(), msg=f"missing {rel}")

        with (out / "coupled_metrics_width8.csv").open() as f:
            rd = csv.DictReader(f)
            rows = list(rd)
        self.assertGreater(len(rows), 0)
        needed_cols = {"width", "pair_id", "step", "time", "R_H", "R_logit", "nan_or_inf"}
        self.assertTrue(needed_cols.issubset(set(rows[0].keys())))
        # Simple e2e check: at least one pair has some contraction at end.
        by_pair = {}
        for r in rows:
            by_pair.setdefault((r["width"], r["pair_id"]), []).append(r)
        found_contract = False
        for pr in by_pair.values():
            pr_sorted = sorted(pr, key=lambda x: int(x["step"]))
            if float(pr_sorted[-1]["R_H"]) < float(pr_sorted[0]["R_H"]):
                found_contract = True
                break
        self.assertTrue(found_contract, msg="Expected at least one pair to contract in smoke run.")

    def test_logit_init_mode_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_width_logit_init_"))
        out = tmp / "run"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rf_logistic_coupling.py"),
                "--n",
                "48",
                "--p",
                "8",
                "--widths",
                "8,16",
                "--m-max",
                "16",
                "--pairs",
                "2",
                "--T-phys",
                "0.12",
                "--h-factor",
                "0.05",
                "--log-dt",
                "0.04",
                "--init-mode",
                "logit",
                "--init-logit-radius",
                "1.0",
                "--init-logit-ridge",
                "1e-4",
                "--seed",
                "3",
                "--out-dir",
                str(out),
            ],
            cwd=str(ROOT),
            check=True,
        )
        with (out / "pair_summary_width8.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        r0 = rows[0]
        self.assertEqual(r0["init_mode"], "logit")
        self.assertTrue(np.isfinite(float(r0["init_D_logit"])))
        self.assertGreater(float(r0["init_D_logit"]), 0.0)
        # For pair separation (a-b)=2*delta, target init_D_logit ≈ 2*radius.
        self.assertLess(abs(float(r0["init_D_logit"]) - 2.0), 0.2)
        for k in ["kappa_logit_early", "tau_logit_0p5", "init_logit_ridge"]:
            self.assertIn(k, r0)

    def test_analyze_script_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_width_analyze_"))
        out = tmp / "run"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rf_logistic_coupling.py"),
                "--n",
                "48",
                "--p",
                "8",
                "--widths",
                "8,16",
                "--m-max",
                "16",
                "--pairs",
                "2",
                "--T-phys",
                "0.1",
                "--h-factor",
                "0.05",
                "--log-dt",
                "0.05",
                "--seed",
                "1",
                "--out-dir",
                str(out),
            ],
            cwd=str(ROOT),
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_rf_width_contraction.py"),
                "--run-dir",
                str(out),
            ],
            cwd=str(ROOT),
            check=True,
        )
        self.assertTrue((out / "rf_contraction_summary.csv").is_file())
        self.assertTrue((out / "rf_contraction_summary.md").is_file())
        self.assertTrue((out / "hessian_spectrum_summary.csv").is_file())
        self.assertTrue((out / "quadratic_contraction_summary.csv").is_file())
        self.assertTrue((out / "quadratic_contraction_width8.csv").is_file())
        with (out / "hessian_spectrum_summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        for r in rows:
            for key in ["frac_prior_0p05", "frac_prior_0p10", "frac_prior_0p25"]:
                x = float(r[key])
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 1.0)

    def test_gaussian_reanalysis_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_gauss_rean_"))
        run = tmp / "gauss"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rf_gaussian_calibration.py"),
                "--n",
                "64",
                "--p",
                "8",
                "--widths",
                "8,16",
                "--m-max",
                "16",
                "--chains",
                "8",
                "--T-phys",
                "0.2",
                "--h-factor",
                "0.05",
                "--log-dt",
                "0.05",
                "--seed",
                "0",
                "--out-dir",
                str(run),
            ],
            cwd=str(ROOT),
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_rf_gaussian_calibration.py"),
                "--run-dir",
                str(run),
                "--normalize-by-dim",
            ],
            cwd=str(ROOT),
            check=True,
        )
        self.assertTrue((run / "gaussian_width_summary_normalized.csv").is_file())
        with (run / "gaussian_width_summary_normalized.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        for r in rows:
            m = float(r["width"])
            e_mu = float(r["final_E_mu"])
            e_mu_norm = float(r["final_E_mu_norm"])
            self.assertAlmostEqual(e_mu_norm, e_mu / np.sqrt(m), places=8)

    def test_pilot2_confirmatory_harness_ci(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_pilot2_confirm_"))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_rf_pilot2_confirmatory.py"),
                "--preset",
                "ci",
                "--out-root",
                str(tmp),
            ],
            cwd=str(ROOT),
            check=True,
        )
        self.assertTrue((tmp / "coupling_hessian" / "hessian_spectrum_summary.csv").is_file())
        self.assertTrue((tmp / "coupling_logit" / "quadratic_contraction_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()

