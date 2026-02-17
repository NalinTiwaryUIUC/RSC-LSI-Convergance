"""
Unit tests for model: ResNet CIFAR and param helpers.
"""
import unittest

import torch

from models.params import flatten_params, param_count, unflatten_like
from models.resnet_cifar import create_model


class TestResNetCIFAR(unittest.TestCase):
    def test_forward_shape(self):
        model = create_model(width_multiplier=0.5, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_param_count_scales_with_width(self):
        m_small = create_model(width_multiplier=0.5)
        m_large = create_model(width_multiplier=2.0)
        d_small = param_count(m_small)
        d_large = param_count(m_large)
        self.assertGreater(d_large, d_small)
        # Width 2.0 vs 0.5 => channel counts scale 4x; param count should scale noticeably (> 2x)
        ratio = d_large / d_small
        self.assertGreaterEqual(ratio, 2.0, msg="Wider model should have at least 2x params")
        self.assertLess(ratio, 50.0, msg="Param scaling should be plausible, not exploded")


class TestParams(unittest.TestCase):
    def test_flatten_unflatten_roundtrip(self):
        model = create_model(width_multiplier=0.5)
        original = [p.data.clone() for p in model.parameters()]
        flat = flatten_params(model)
        self.assertEqual(flat.numel(), sum(p.numel() for p in model.parameters()))
        # Perturb
        flat = flat + 0.1 * torch.randn_like(flat)
        unflatten_like(flat, model)
        flat2 = flatten_params(model)
        torch.testing.assert_close(flat2, flat)

    def test_param_count(self):
        model = create_model(width_multiplier=1.0)
        d = param_count(model)
        flat = flatten_params(model)
        self.assertEqual(d, flat.numel())
