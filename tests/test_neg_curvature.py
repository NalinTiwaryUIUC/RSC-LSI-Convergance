"""
Unit tests for the negative-curvature / NME experiment math primitives, plus
end-to-end smoke tests for the runner and plotter.

**Quick check (math + linop only, a few seconds — recommended for CI / pre-push):**

    ./scripts/run_neg_curvature_tests_quick.sh

Equivalent::

    python3 -m unittest -v \\
      tests.test_neg_curvature.TestHVP \\
      tests.test_neg_curvature.TestGGNCE \\
      tests.test_neg_curvature.TestNMEZeroForLinearModel \\
      tests.test_neg_curvature.TestTopKSmallest \\
      tests.test_neg_curvature.TestSLQ \\
      tests.test_neg_curvature.TestCumulativeMetrics \\
      tests.test_neg_curvature.TestWidthMapping \\
      tests.test_neg_curvature.TestLinop \\
      tests.test_neg_curvature.TestFlatGrad

**Full module (adds ResNet runner smoke + matplotlib plot subprocess — slower):**

    python3 -m unittest tests.test_neg_curvature -v

**After multi-seed pilot, mean ± std table:**

    MODE=table_pilot sbatch scripts/run_neg_curvature.sh
    # or: python3 scripts/aggregate_neg_curvature.py --run-glob 'experiments/neg_curv/pilot_seed*' --checkpoint final --out-csv ...
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from curvature_common import (  # noqa: E402
    cumulative_eta_metrics,
    e_aniso,
    e_iso,
    flat_grad,
    ggn_ce_vp,
    hvp_full,
    make_torch_linop,
    nme_vp,
    r_eff_neg,
    slq_trace_neg,
    top_k_smallest_eigs,
)


def _functional_logits(model: nn.Module, theta_flat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    params = {}
    i = 0
    for name, p in model.named_parameters():
        n = p.numel()
        params[name] = theta_flat[i : i + n].view_as(p)
        i += n
    return functional_call(model, params, (x,))


# ---------------------------------------------------------------------------
# 1. HVP correctness against dense Hessian via functional_call.
# ---------------------------------------------------------------------------

class TestHVP(unittest.TestCase):
    def test_hvp_linear_mse(self):
        torch.manual_seed(0)
        d, k, B = 4, 2, 6
        model = nn.Linear(d, k, bias=False).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randn(B, k, dtype=torch.float64)

        def mse(z, y_):
            return F.mse_loss(z, y_, reduction="mean")

        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

        def loss_of_theta(theta_flat):
            z = _functional_logits(model, theta_flat, x)
            return mse(z, y)

        H_dense = torch.autograd.functional.hessian(loss_of_theta, theta0)

        gen = torch.Generator().manual_seed(7)
        for _ in range(3):
            v = torch.randn(theta0.numel(), dtype=torch.float64, generator=gen)
            h_v = hvp_full(model, x, y, v, loss_fn=mse)
            torch.testing.assert_close(h_v, H_dense @ v, atol=1e-9, rtol=1e-9)

    def test_hvp_ce_against_dense(self):
        torch.manual_seed(1)
        d, k, B = 5, 3, 7
        model = nn.Linear(d, k, bias=True).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randint(0, k, (B,))

        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

        def loss_of_theta(theta_flat):
            z = _functional_logits(model, theta_flat, x)
            return F.cross_entropy(z, y, reduction="mean")

        H_dense = torch.autograd.functional.hessian(loss_of_theta, theta0)

        gen = torch.Generator().manual_seed(11)
        for _ in range(3):
            v = torch.randn(theta0.numel(), dtype=torch.float64, generator=gen)
            h_v = hvp_full(model, x, y, v)
            torch.testing.assert_close(h_v, H_dense @ v, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. GGN-CE correctness against (1/B) sum_i J_i^T S_i J_i.
# ---------------------------------------------------------------------------

class TestGGNCE(unittest.TestCase):
    def test_ggn_ce_matches_dense(self):
        torch.manual_seed(2)
        d, k, B = 4, 3, 5
        model = nn.Linear(d, k, bias=True).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randint(0, k, (B,))

        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

        def logits_of_theta(theta_flat):
            return _functional_logits(model, theta_flat, x)

        J = torch.autograd.functional.jacobian(logits_of_theta, theta0)  # [B, k, p]
        z0 = logits_of_theta(theta0)
        s = torch.softmax(z0.detach(), dim=-1)  # [B, k]
        p_total = theta0.numel()
        G = torch.zeros(p_total, p_total, dtype=torch.float64)
        for i in range(B):
            Ji = J[i]  # [k, p]
            Si = torch.diag(s[i]) - torch.outer(s[i], s[i])
            G = G + Ji.T @ Si @ Ji
        G = G / B

        gen = torch.Generator().manual_seed(13)
        for _ in range(3):
            v = torch.randn(p_total, dtype=torch.float64, generator=gen)
            gv = ggn_ce_vp(model, x, y, v)
            torch.testing.assert_close(gv, G @ v, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 3. For a model linear in theta, N = H_L - G must be exactly zero.
# ---------------------------------------------------------------------------

class TestNMEZeroForLinearModel(unittest.TestCase):
    def test_nme_linear_model_is_zero(self):
        torch.manual_seed(3)
        d, k, B = 4, 3, 6
        model = nn.Linear(d, k, bias=True).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randint(0, k, (B,))
        p_total = sum(p.numel() for p in model.parameters())
        gen = torch.Generator().manual_seed(17)
        for _ in range(3):
            v = torch.randn(p_total, dtype=torch.float64, generator=gen)
            nv = nme_vp(model, x, y, v)
            self.assertLess(float(torch.linalg.norm(nv)), 1e-8)


# ---------------------------------------------------------------------------
# 4. Lanczos top-k smallest matches dense eigh.
# ---------------------------------------------------------------------------

class TestTopKSmallest(unittest.TestCase):
    def test_top_k_smallest_eigs_matches_eigh(self):
        rng = np.random.default_rng(42)
        p = 64
        A = rng.normal(size=(p, p))
        M = (A + A.T) / 2.0
        M -= 2.0 * np.eye(p)

        def matvec(v_np):
            return (M @ np.asarray(v_np, dtype=np.float64)).astype(np.float64)

        linop = sla.LinearOperator((p, p), matvec=matvec, rmatvec=matvec, dtype=np.float64)
        eigs = top_k_smallest_eigs(linop, k=10, ncv=4 * 10 + 1, maxiter=500, tol=0.0)
        eigs_full = np.linalg.eigvalsh(M)[:10]
        np.testing.assert_allclose(eigs, eigs_full, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. SLQ converges (within tolerance) to the true negative trace.
# ---------------------------------------------------------------------------

class TestSLQ(unittest.TestCase):
    def test_slq_neg_trace_close_to_truth(self):
        rng = np.random.default_rng(0)
        p = 32
        eigvals = np.concatenate(
            [np.array([-3.0, -2.0, -1.0, -0.5]),
             np.linspace(1.0, 5.0, p - 4)]
        )
        Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
        M = Q @ np.diag(eigvals) @ Q.T
        M = 0.5 * (M + M.T)

        def matvec(v_np):
            return (M @ np.asarray(v_np, dtype=np.float64)).astype(np.float64)

        truth = float(np.maximum(0.0, -eigvals).sum())
        rng_probe = np.random.default_rng(7)
        # Fewer probes than the production SLQ default keeps the full test file fast.
        est = slq_trace_neg(matvec, p, num_probes=24, lanczos_steps=16, rng=rng_probe)
        self.assertGreater(est, 0.0)
        self.assertLess(abs(est - truth) / max(truth, 1e-12), 0.28)


# ---------------------------------------------------------------------------
# 6. Cumulative metrics k50/k80/k90 closed form.
# ---------------------------------------------------------------------------

class TestCumulativeMetrics(unittest.TestCase):
    def test_cumulative_eta_metrics_simple(self):
        eigs = np.array([-3.0, -2.0, -1.0, 0.5])
        m = cumulative_eta_metrics(eigs)
        self.assertAlmostEqual(m["gamma_emp"], 3.0)
        self.assertAlmostEqual(m["T_neg_top"], 6.0)
        # eta_sorted desc = [3, 2, 1] -> cum / 6 = [0.5, 0.833, 1.0]
        self.assertEqual(m["k50"], 1)
        self.assertEqual(m["k80"], 2)
        self.assertEqual(m["k90"], 3)

    def test_cumulative_no_negatives(self):
        eigs = np.array([0.1, 0.2, 0.3])
        m = cumulative_eta_metrics(eigs)
        self.assertEqual(m["gamma_emp"], 0.0)
        self.assertEqual(m["T_neg_top"], 0.0)
        self.assertEqual(m["k50"], 0)
        self.assertEqual(m["k80"], 0)
        self.assertEqual(m["k90"], 0)

    def test_summary_scalars(self):
        self.assertAlmostEqual(r_eff_neg(6.0, 3.0), 2.0)
        self.assertEqual(r_eff_neg(6.0, 0.0), 0.0)
        self.assertAlmostEqual(e_iso(0.5, 100), 50.0)
        self.assertAlmostEqual(e_aniso(7.5), 7.5)


# ---------------------------------------------------------------------------
# 7. Width <-> hidden m mapping for small_resnet_ln.
# ---------------------------------------------------------------------------

class TestWidthMapping(unittest.TestCase):
    def test_width_to_hidden_m(self):
        from models import create_model
        from models.params import param_count

        counts: list[int] = []
        for w in (1, 2, 4):
            model = create_model(width_multiplier=float(w),
                                 arch="small_resnet_ln", num_blocks=1)
            first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
            self.assertEqual(first_conv.out_channels, 64 * w)
            counts.append(param_count(model))
        self.assertGreater(counts[1], counts[0])
        self.assertGreater(counts[2], counts[1])


# ---------------------------------------------------------------------------
# 8 + 9. End-to-end smoke for runner and plotter (no CIFAR; monkey-patched).
# ---------------------------------------------------------------------------

def _synthetic_subset(device, torch_dtype, n_train=8, n_curv=4):
    torch.manual_seed(123)
    x_train = torch.randn(n_train, 3, 32, 32, dtype=torch_dtype, device=device)
    y_train = torch.randint(0, 10, (n_train,), device=device)
    x_curv = x_train[:n_curv]
    y_curv = y_train[:n_curv]
    return x_train, y_train, x_curv, y_curv


class TestRunnerSmoke(unittest.TestCase):
    """Drive `_run_one_seed` directly with synthetic data."""

    def test_runner_smoke(self):
        import run_neg_curvature as runner

        original_loader = runner._load_fixed_subset
        try:
            runner._load_fixed_subset = lambda *_, device, torch_dtype, **__: _synthetic_subset(device, torch_dtype)
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "smoke"
                ns = argparse.Namespace(
                    widths="1", seed=0, seeds="",
                    arch="small_resnet_ln", num_blocks=1,
                    n_train=8, n_curv=4,
                    dataset_seed=42, data_dir=str(out / "data"), root=str(out / "data"),
                    lr=0.05, momentum=0.9, weight_decay=0.0,
                    max_steps=2, mid_step=1,
                    checkpoints="final",
                    curvature_mode="legacy",
                    snapshot_steps="",
                    match_backup="closest_acc",
                    match_target_loss=float("nan"),
                    matched_label="",
                    save_ckpts=False,
                    # eigsh needs enough iterations for ARPACK on ~100k-dim N;
                    # 80 matches production default so the first attempt converges.
                    num_neg=2, lanczos_steps=80, ncv=9, lanczos_tol=0.0,
                    slq=False, num_probes=0, slq_steps=0,
                    local_check=False, num_local=0, eps_rel=0.01,
                    num_local_neg=0, local_lanczos_steps=10,
                    dtype="float32", device="cpu",
                    out_dir=str(out),
                    matched_train_acc=None,
                )
                runner._run_one_seed(out, 0, ns)

                self.assertTrue((out / "config.yaml").exists())
                csv_path = out / "curvature_summary.csv"
                self.assertTrue(csv_path.exists())
                with csv_path.open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                row = rows[0]
                for col in [
                    "width", "m", "seed", "checkpoint", "p",
                    "train_loss", "train_acc", "curv_loss", "curv_acc",
                    "gamma_emp", "sqrt_m_gamma_emp",
                    "T_neg_top20", "T_neg_SLQ",
                    "r_eff_top20", "r_eff_neg", "r_eff_over_p", "r_eff_top20_over_p",
                    "r_eff_over_sqrt_m",
                    "E_iso", "E_aniso", "E_aniso_over_E_iso", "E_aniso_top20_over_E_iso",
                    "k50", "k80", "k90",
                ]:
                    self.assertIn(col, row, msg=f"missing column {col!r}")
                self.assertEqual(row["checkpoint"], "final")
                self.assertEqual(int(row["m"]), 64)
                self.assertGreater(int(row["p"]), 0)

                eig_files = list(out.glob("negative_eigs_width1_seed0_final.csv"))
                self.assertEqual(len(eig_files), 1)
                with eig_files[0].open() as f:
                    eig_rows = list(csv.DictReader(f))
                self.assertEqual(len(eig_rows), 2)
                for col in ["rank", "lambda", "eta", "cum_eta", "cum_eta_over_Tneg"]:
                    self.assertIn(col, eig_rows[0])
        finally:
            runner._load_fixed_subset = original_loader


# ---------------------------------------------------------------------------
# 10. Plotter smoke test using synthetic per-seed CSVs.
# ---------------------------------------------------------------------------

def _write_synthetic_seed_dir(seed_dir: Path, *, widths=(1, 2, 4), seed: int = 0):
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for w in widths:
        m_hidden = 64 * w
        gamma = 0.1 / float(np.sqrt(m_hidden))
        eigs = np.array([-gamma, -gamma * 0.5, -gamma * 0.25, 0.1, 0.2])
        T_neg = float(np.maximum(0.0, -eigs).sum())
        eig_path = seed_dir / f"negative_eigs_width{w}_seed{seed}_final.csv"
        with eig_path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["rank", "lambda", "eta", "cum_eta", "cum_eta_over_Tneg"])
            wr.writeheader()
            cum = 0.0
            for rank, lam in enumerate(np.sort(eigs), start=1):
                eta_v = max(0.0, -float(lam))
                cum += eta_v
                wr.writerow({"rank": rank, "lambda": float(lam), "eta": eta_v,
                             "cum_eta": cum,
                             "cum_eta_over_Tneg": (cum / T_neg) if T_neg > 0 else float("nan")})
        p = 1000 * w
        r_top = T_neg / gamma if gamma > 0 else 0.0
        summary_rows.append({
            "width": w, "m": m_hidden, "seed": seed, "checkpoint": "final",
            "step": 1000, "p": p,
            "train_loss": 0.5, "train_acc": 90.0, "curv_loss": 0.5, "curv_acc": 90.0,
            "gamma_emp": gamma, "sqrt_m_gamma_emp": np.sqrt(m_hidden) * gamma,
            "T_neg_top20": T_neg, "T_neg_SLQ": float("nan"), "T_neg_used": T_neg,
            "r_eff_top20": r_top,
            "r_eff_neg": r_top,
            "r_eff_over_p": (T_neg / gamma) / p if gamma > 0 else 0.0,
            "r_eff_top20_over_p": r_top / p if gamma > 0 else 0.0,
            "r_eff_over_sqrt_m": (T_neg / gamma) / np.sqrt(m_hidden) if gamma > 0 else 0.0,
            "E_iso": gamma * p, "E_aniso": T_neg,
            "E_aniso_over_E_iso": T_neg / (gamma * p) if gamma > 0 else 0.0,
            "E_aniso_top20_over_E_iso": T_neg / (gamma * p) if gamma > 0 else 0.0,
            "k50": 1, "k80": 2, "k90": 3,
            "local_gamma_max": float("nan"), "local_gamma_mean": float("nan"),
            "local_gamma_std": float("nan"),
        })
    from run_neg_curvature import SUMMARY_FIELDNAMES  # type: ignore
    with (seed_dir / "curvature_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        wr.writeheader()
        wr.writerows(summary_rows)


class TestPlotSmoke(unittest.TestCase):
    def test_plot_smoke(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for s in (0, 1):
                _write_synthetic_seed_dir(base / f"main_seed{s}", widths=(1, 2, 4), seed=s)
            plot_dir = base / "main_plots"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "plot_neg_curvature.py"),
                "--run-glob", str(base / "main_seed*"),
                "--plot-out-dir", str(plot_dir),
                "--checkpoint", "final",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=f"plotter failed: stdout={r.stdout!r} stderr={r.stderr!r}")
            for fname in [
                "gamma_emp_vs_width.pdf",
                "r_eff_over_p_vs_width.pdf",
                "E_iso_vs_E_aniso_vs_width.pdf",
                "cumulative_neg_trace_C_of_k.pdf",
            ]:
                p = plot_dir / "plots" / fname
                self.assertTrue(p.exists(), msg=f"missing {fname}")
                self.assertGreater(p.stat().st_size, 100, msg=f"{fname} suspiciously small")


# ---------------------------------------------------------------------------
# 11. LinearOperator bridge sanity (numpy <-> torch).
# ---------------------------------------------------------------------------

class TestBackupMatchSteps(unittest.TestCase):
    def test_closest_acc_tie_prefers_earlier(self):
        import run_neg_curvature as runner

        probes = [(250, 1.0, 85.0), (500, 1.0, 95.0), (750, 1.0, 85.0)]
        st, reason = runner._pick_backup_match_step(
            probes,
            backup="closest_acc",
            acc_target=90.0,
            target_loss=float("nan"),
            max_steps=2000,
        )
        self.assertEqual(st, 250)
        self.assertEqual(reason, "closest_acc")

    def test_closest_loss(self):
        import run_neg_curvature as runner

        probes = [(100, 0.5, 50.0), (200, 0.35, 60.0)]
        st, _ = runner._pick_backup_match_step(
            probes,
            backup="closest_loss",
            acc_target=90.0,
            target_loss=0.4,
            max_steps=2000,
        )
        self.assertEqual(st, 200)


class TestLinop(unittest.TestCase):
    def test_make_torch_linop_roundtrip(self):
        device = torch.device("cpu")
        torch_dtype = torch.float64
        p = 7
        M = torch.randn(p, p, dtype=torch_dtype)
        M = 0.5 * (M + M.T)

        def matvec_torch(v):
            return M @ v

        linop = make_torch_linop(matvec_torch, p, device=device,
                                 torch_dtype=torch_dtype, np_dtype=np.float64)
        v = np.random.default_rng(0).normal(size=p)
        out = linop.matvec(v)
        truth = (M @ torch.as_tensor(v, dtype=torch_dtype)).numpy()
        np.testing.assert_allclose(out, truth, atol=1e-12)


# ---------------------------------------------------------------------------
# 12. flat_grad and ce_mean_loss helpers.
# ---------------------------------------------------------------------------

class TestCELossScaling(unittest.TestCase):
    """Full loss Hessian H_L scales linearly when CE is multiplied by a constant."""

    def test_hvp_scales_with_ce_multiplier(self):
        torch.manual_seed(21)
        d, k, B = 4, 3, 8
        model = nn.Linear(d, k, bias=True).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randint(0, k, (B,))
        p_total = sum(p.numel() for p in model.parameters())
        v = torch.randn(p_total, dtype=torch.float64)

        def ce_scaled(scale: float):
            def loss_fn(z, y_):
                return scale * F.cross_entropy(z, y_, reduction="mean")

            return loss_fn

        s = 2.5
        h1 = hvp_full(model, x, y, v, loss_fn=ce_scaled(1.0))
        h2 = hvp_full(model, x, y, v, loss_fn=ce_scaled(s))
        torch.testing.assert_close(h2, s * h1, rtol=1e-9, atol=1e-9)


class TestFlatGrad(unittest.TestCase):
    def test_flat_grad_matches_autograd(self):
        torch.manual_seed(5)
        d, k, B = 4, 3, 6
        model = nn.Linear(d, k, bias=True).double()
        x = torch.randn(B, d, dtype=torch.float64)
        y = torch.randint(0, k, (B,))
        g = flat_grad(model, x, y)
        # Compare against direct autograd
        loss = F.cross_entropy(model(x), y, reduction="mean")
        grads = torch.autograd.grad(loss, list(model.parameters()))
        truth = torch.cat([gi.reshape(-1) for gi in grads])
        torch.testing.assert_close(g, truth, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
