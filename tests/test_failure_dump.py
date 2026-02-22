"""
Tests for failure dump and diagnostic guards.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from models import create_model
from run.persistence import dump_failure


class TestFailureDump(unittest.TestCase):
    def test_dump_failure_creates_file(self):
        """dump_failure creates FAIL_step{N}.pt with model state and payload."""
        run_dir = Path(tempfile.mkdtemp(prefix="lsi_fail_dump_"))
        model = create_model(width_multiplier=0.5)
        step = 42
        payload = {"h": 1e-5, "alpha": 0.01, "U_train": float("nan")}
        dump_failure(run_dir, step, model, payload)
        path = run_dir / f"FAIL_step{step}.pt"
        self.assertTrue(path.exists(), msg=f"Expected {path} to exist")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertIn("step", ckpt)
        self.assertEqual(ckpt["step"], step)
        self.assertIn("model", ckpt)
        self.assertIn("h", ckpt)
        self.assertIn("U_train", ckpt)
        self.assertTrue(torch.isnan(torch.tensor(ckpt["U_train"])))
