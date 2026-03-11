"""
Unit tests for model: ResNet CIFAR (resnet18) and small LayerNorm ResNet, plus param helpers.
"""
import unittest

import torch

from models import create_model
from models.params import flatten_params, param_count, unflatten_like


class TestResNetCIFAR(unittest.TestCase):
    def test_forward_shape(self):
        model = create_model(width_multiplier=0.5, num_classes=10, arch="resnet18")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_param_count_scales_with_width(self):
        m_small = create_model(width_multiplier=0.5, arch="resnet18")
        m_large = create_model(width_multiplier=2.0, arch="resnet18")
        d_small = param_count(m_small)
        d_large = param_count(m_large)
        self.assertGreater(d_large, d_small)
        # Width 2.0 vs 0.5 => channel counts scale 4x; param count should scale noticeably (> 2x)
        ratio = d_large / d_small
        self.assertGreaterEqual(ratio, 2.0, msg="Wider model should have at least 2x params")
        self.assertLess(ratio, 50.0, msg="Param scaling should be plausible, not exploded")


class TestSmallResNetLN(unittest.TestCase):
    def test_forward_shape_small_resnet_ln(self):
        model = create_model(
            width_multiplier=0.5,
            num_classes=10,
            arch="small_resnet_ln",
            num_blocks=2,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_small_resnet_ln_has_fewer_params_than_resnet18(self):
        m_small_ln = create_model(
            width_multiplier=0.5,
            arch="small_resnet_ln",
            num_blocks=2,
        )
        m_resnet18 = create_model(
            width_multiplier=0.5,
            arch="resnet18",
        )
        d_small_ln = param_count(m_small_ln)
        d_resnet18 = param_count(m_resnet18)
        self.assertLess(
            d_small_ln,
            d_resnet18,
            msg="small_resnet_ln should have fewer parameters than resnet18 at same width",
        )


class TestParams(unittest.TestCase):
    def test_flatten_unflatten_roundtrip_resnet18(self):
        model = create_model(width_multiplier=0.5, arch="resnet18")
        original = [p.data.clone() for p in model.parameters()]
        flat = flatten_params(model)
        self.assertEqual(flat.numel(), sum(p.numel() for p in model.parameters()))
        # Perturb
        flat = flat + 0.1 * torch.randn_like(flat)
        unflatten_like(flat, model)
        flat2 = flatten_params(model)
        torch.testing.assert_close(flat2, flat)

    def test_flatten_unflatten_roundtrip_small_resnet_ln(self):
        model = create_model(width_multiplier=0.5, arch="small_resnet_ln", num_blocks=2)
        flat = flatten_params(model)
        self.assertEqual(flat.numel(), sum(p.numel() for p in model.parameters()))
        flat_perturbed = flat + 0.1 * torch.randn_like(flat)
        unflatten_like(flat_perturbed, model)
        flat2 = flatten_params(model)
        torch.testing.assert_close(flat2, flat_perturbed)

    def test_param_count_matches_flatten_length_resnet18(self):
        model = create_model(width_multiplier=1.0, arch="resnet18")
        d = param_count(model)
        flat = flatten_params(model)
        self.assertEqual(d, flat.numel())

    def test_param_count_matches_flatten_length_small_resnet_ln(self):
        model = create_model(width_multiplier=1.0, arch="small_resnet_ln", num_blocks=2)
        d = param_count(model)
        flat = flatten_params(model)
        self.assertEqual(d, flat.numel())
