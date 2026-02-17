"""
Unit tests for ULA: potential, step.
"""
import tempfile
import unittest

import torch

from data.cifar import get_train_loader
from models import create_model, flatten_params, param_count
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


class TestULAStep(unittest.TestCase):
    def test_one_step_changes_theta(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        theta0 = flatten_params(model).clone()
        out = ula_step(
            model, train, alpha=1e-2, h=1e-5, device=device, return_U=True,
        )
        theta_after = flatten_params(model)
        self.assertFalse(torch.allclose(theta_after, theta0))
        self.assertIn("U", out)
        self.assertTrue(torch.isfinite(torch.tensor(out["U"])))
