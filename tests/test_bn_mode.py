"""
Unit tests for BN mode utilities.
"""
import unittest

import torch
import torch.nn as nn

from models import create_model
from run.bn_mode import max_bn_running_delta, set_bn_batchstats_freeze_buffers, snapshot_bn_running_stats


class TestBnMode(unittest.TestCase):
    def test_set_bn_batchstats_freeze_buffers(self):
        """After call: all BN layers have momentum==0, BN in train, Dropout in eval."""
        model = create_model(width_multiplier=0.5)
        set_bn_batchstats_freeze_buffers(model)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.assertEqual(m.momentum, 0.0, msg=f"BN {m} should have momentum=0")
                self.assertTrue(m.training, msg=f"BN {m} should be in train mode")
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                self.assertFalse(m.training, msg=f"Dropout {m} should be in eval mode")
        self.assertTrue(model.training, msg="Model should be in train mode")

    def test_bn_buffers_unchanged(self):
        """Snapshot before/after N forward passes; max_bn_running_delta should be ~0."""
        model = create_model(width_multiplier=0.5)
        set_bn_batchstats_freeze_buffers(model)
        x = torch.randn(8, 3, 32, 32)
        stats_before = snapshot_bn_running_stats(model)
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        delta = max_bn_running_delta(stats_before, model)
        self.assertLess(delta, 1e-8, msg="BN running stats should be frozen (delta near 0)")
