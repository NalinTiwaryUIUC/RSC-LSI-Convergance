"""
Unit tests for ULA: potential, step, B_t metrics.
"""
import tempfile
import unittest

import torch

from data.cifar import get_train_loader
from models import create_model, flatten_params, param_count
from ula.domain_bt import compute_bt_metrics, get_rho2
from ula.potential import compute_U
from ula.step import ula_step


def _get_small_loaders():
    data_dir = tempfile.mkdtemp(prefix="lsi_test_ula_")
    train = get_train_loader(
        n=64, batch_size=64, dataset_seed=42, data_dir=data_dir, root="./data"
    )
    return train, data_dir


class TestPotential(unittest.TestCase):
    def test_compute_U_scalar(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        U = compute_U(model, train, alpha=1e-2, device=device)
        self.assertTrue(U.dim() == 0 or U.numel() == 1)
        self.assertTrue(torch.isfinite(U).item())

    def test_compute_U_gradient_flow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        U = compute_U(model, train, alpha=1e-2, device=device)
        U.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())


class TestDomainBt(unittest.TestCase):
    def test_inside_bt(self):
        theta0 = torch.randn(100)
        theta = theta0 + 0.001 * torch.randn(100)
        rho2 = get_rho2(theta0, factor=0.05)
        theta_dist, bt_margin, inside_bt = compute_bt_metrics(theta, theta0, rho2)
        self.assertIsInstance(theta_dist, float)
        self.assertIsInstance(bt_margin, float)
        self.assertIsInstance(inside_bt, bool)
        self.assertLess(theta_dist, rho2 * 2)  # small perturbation
        self.assertTrue(inside_bt)


class TestULAStep(unittest.TestCase):
    def test_one_step_changes_theta(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        theta0 = flatten_params(model).clone()
        rho2 = get_rho2(theta0, factor=0.05)
        out = ula_step(
            model, train, alpha=1e-2, h=1e-5, theta0_flat=theta0,
            rho2=rho2, device=device, return_U=True,
        )
        theta_after = flatten_params(model)
        self.assertFalse(torch.allclose(theta_after, theta0))
        self.assertIn("theta_dist", out)
        self.assertIn("bt_margin", out)
        self.assertIn("inside_bt", out)
        self.assertIn("U", out)
        self.assertIsInstance(out["theta_dist"], float)
        self.assertIsInstance(out["bt_margin"], float)
        self.assertTrue(torch.isfinite(torch.tensor(out["U"])))
