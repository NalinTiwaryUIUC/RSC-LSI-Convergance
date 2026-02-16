"""
Unit tests for probes and gradient norms.
"""
import tempfile
import unittest

import torch

from data.cifar import get_probe_loader, get_train_loader
from models import create_model, flatten_params, param_count
from probes import (
    PROBES_FOR_GRAD_NORM,
    compute_grad_norm_sq,
    evaluate_probes,
    get_probe_value_for_grad,
    get_or_create_logit_projection,
    get_or_create_param_projections,
)


def _data_dir():
    return tempfile.mkdtemp(prefix="lsi_test_probes_")


class TestEvaluateProbes(unittest.TestCase):
    def test_evaluate_probes_shapes_and_finite(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = _data_dir()
        model = create_model(width_multiplier=0.5).to(device)
        train_loader = get_train_loader(64, batch_size=64, data_dir=data_dir, root="./data")
        probe_loader = get_probe_loader(32, data_dir=data_dir, root="./data")
        theta0 = flatten_params(model).detach()
        d = theta0.numel()
        logit_dim = 32 * 10
        v1, v2 = get_or_create_param_projections(d, data_dir=data_dir)
        logit_proj = get_or_create_logit_projection(logit_dim, data_dir=data_dir)
        values = evaluate_probes(
            model, probe_loader, theta0, v1, v2, logit_proj, device
        )
        for k, v in values.items():
            self.assertIn(k, ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"])
            self.assertIsInstance(v, float)
            self.assertTrue(torch.isfinite(torch.tensor(v)).item(), msg=k)

    def test_grad_norm_sq_one_probe(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = _data_dir()
        model = create_model(width_multiplier=0.5).to(device)
        probe_loader = get_probe_loader(32, data_dir=data_dir, root="./data")
        theta0 = flatten_params(model).detach().to(device)
        d = theta0.numel()
        v1, v2 = get_or_create_param_projections(d, data_dir=data_dir)
        logit_dim = 32 * 10
        logit_proj = get_or_create_logit_projection(logit_dim, data_dir=data_dir)
        f_scalar = get_probe_value_for_grad(
            model, probe_loader, theta0, v1, v2, logit_proj, "f_nll", device
        )
        self.assertTrue(f_scalar.requires_grad or f_scalar.grad_fn is not None)
        grad_sq = compute_grad_norm_sq(f_scalar, model.parameters())
        self.assertIsInstance(grad_sq, float)
        self.assertGreaterEqual(grad_sq, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(grad_sq)).item())
