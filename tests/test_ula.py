"""
Unit tests for ULA: potential, step.
"""
import tempfile
import unittest

import torch

from data.cifar import get_train_loader
from models import create_model, flatten_params, param_count, unflatten_like
from ula.potential import compute_U
from ula.step import ula_step


def _get_small_loaders():
    data_dir = tempfile.mkdtemp(prefix="lsi_test_ula_")
    train = get_train_loader(
        n=64, batch_size=64, dataset_seed=42, data_dir=data_dir, root="./data"
    )
    return train, data_dir


def _get_n1024_loader():
    """Full-batch loader with n=1024 for SNR test at production-like scale."""
    data_dir = tempfile.mkdtemp(prefix="lsi_test_ula_n1024_")
    train = get_train_loader(
        n=1024, batch_size=1024, dataset_seed=42, data_dir=data_dir, root="./data"
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
        # U = sum CE + (alpha/2)*||theta||^2 is positive and not exploded
        u_val = U.item()
        self.assertGreater(u_val, 0.0, msg="U should be positive")
        self.assertLess(u_val, 1e7, msg="U should not explode on small setup")

    def test_compute_U_gradient_flow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        U = compute_U(model, train, alpha=1e-2, device=device)
        U.backward()
        grads = []
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            grads.append(p.grad.view(-1))
        total_grad = torch.cat(grads)
        self.assertGreater(total_grad.norm().item(), 0.0, msg="Gradient should be non-zero")


class TestULAStep(unittest.TestCase):
    def test_one_step_changes_theta(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        theta0 = flatten_params(model).clone()
        out = ula_step(
            model, train, alpha=1e-2, h=1e-5, device=device,
            noise_scale=0.1, return_U=True,
        )
        theta_after = flatten_params(model)
        self.assertFalse(torch.allclose(theta_after, theta0))
        self.assertIn("U", out)
        u_val = out["U"]
        self.assertTrue(torch.isfinite(torch.tensor(u_val)).item())
        self.assertGreater(u_val, 0.0)
        self.assertLess(u_val, 1e7)
        self.assertIn("grad_norm", out)
        self.assertIn("theta_norm", out)
        self.assertGreater(out["grad_norm"], 0.0)
        self.assertGreater(out["theta_norm"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(out["grad_norm"])).item())
        self.assertTrue(torch.isfinite(torch.tensor(out["theta_norm"])).item())

    def test_ula_step_noise_scale_zero_is_deterministic(self):
        """With noise_scale=0 the step is pure gradient descent; same seed => same outcome."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train, _ = _get_small_loaders()
        x, y = next(iter(train))
        x, y = x.to(device), y.to(device)
        train_batch = (x, y)

        gen = torch.Generator(device=device).manual_seed(12345)
        model1 = create_model(width_multiplier=0.5).to(device)
        theta0 = flatten_params(model1).clone()
        ula_step(model1, train_batch, alpha=1e-2, h=1e-5, device=device, noise_scale=0.0, generator=gen)
        out1 = flatten_params(model1).clone()

        gen2 = torch.Generator(device=device).manual_seed(12345)
        model2 = create_model(width_multiplier=0.5).to(device)
        unflatten_like(theta0.clone(), model2)
        ula_step(model2, train_batch, alpha=1e-2, h=1e-5, device=device, noise_scale=0.0, generator=gen2)
        out2 = flatten_params(model2)
        torch.testing.assert_close(out1, out2, msg="noise_scale=0 should be deterministic")


class TestSignalToNoise(unittest.TestCase):
    def test_snr_finite_and_positive(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_small_loaders()
        # Compute gradient of U
        model.zero_grad(set_to_none=True)
        U = compute_U(model, train, alpha=1e-2, device=device)
        U.backward()
        grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
        d = grads.numel()
        h = 1e-5
        noise_scale = 0.1
        signal = h * grads.norm().item()
        noise = (2.0 * h * d) ** 0.5 * noise_scale
        snr = signal / noise if noise > 0 else float("nan")
        # With reduced noise_scale=0.1, SNR should be small but not purely random-walk.
        self.assertTrue(torch.isfinite(torch.tensor(snr)).item())
        self.assertGreater(snr, 1e-3)
        self.assertLess(snr, 5e-2)

    def test_snr_n1024(self):
        """SNR at n=1024 with default noise_scale=0.03 should be in a reasonable band."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(width_multiplier=0.5).to(device)
        train, _ = _get_n1024_loader()
        model.zero_grad(set_to_none=True)
        U = compute_U(model, train, alpha=1e-2, device=device)
        U.backward()
        grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
        d = grads.numel()
        h = 1e-5
        noise_scale = 0.03  # default from config
        signal = h * grads.norm().item()
        noise = (2.0 * h * d) ** 0.5 * noise_scale
        snr = signal / noise if noise > 0 else float("nan")
        self.assertTrue(torch.isfinite(torch.tensor(snr)).item(), msg=f"SNR should be finite, got {snr}")
        self.assertGreater(snr, 1e-3, msg=f"SNR too low at n=1024: {snr}")
        self.assertLess(snr, 0.2, msg=f"SNR too high at n=1024: {snr}")
