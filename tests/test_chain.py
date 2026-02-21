"""
Minimal chain test: T=10, B=2, S=2 to verify run_chain and persistence.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import RunConfig
from data import get_probe_loader, get_train_loader
from run.chain import run_chain
from run.persistence import load_run_config


class TestChainPersistence(unittest.TestCase):
    def test_chain_produces_files(self):
        """Short run produces run_config.yaml, iter_metrics.jsonl, samples_metrics.npz."""
        data_dir = tempfile.mkdtemp(prefix="lsi_chain_test_")
        run_dir = Path(data_dir) / "run"
        config = RunConfig(
            n_train=64,
            probe_size=16,
            width_multiplier=0.5,
            h=1e-5,
            T=10,
            B=2,
            S=2,
            pretrain_steps=0,
            data_dir=data_dir,
        )
        train_loader = get_train_loader(
            config.n_train, batch_size=config.n_train, data_dir=data_dir, root="./data"
        )
        probe_loader = get_probe_loader(
            config.probe_size, data_dir=data_dir, root="./data"
        )
        run_chain(config, chain_id=0, run_dir=run_dir, train_loader=train_loader, probe_loader=probe_loader)
        self.assertTrue((run_dir / "run_config.yaml").exists())
        self.assertTrue((run_dir / "iter_metrics.jsonl").exists())
        self.assertTrue((run_dir / "samples_metrics.npz").exists())
        loaded = load_run_config(run_dir)
        self.assertEqual(loaded.T, 10)
        self.assertEqual(loaded.B, 2)
        self.assertEqual(loaded.h, 1e-5)
        self.assertGreater(loaded.noise_scale, 0.0)
        self.assertLessEqual(loaded.noise_scale, 10.0)  # allow diagnostic-tuned values

        # iter_metrics: required keys and finite values
        with open(run_dir / "iter_metrics.jsonl") as f:
            lines = [json.loads(l) for l in f]
        self.assertGreater(len(lines), 0, msg="At least one iter_metrics line")
        required_keys = {"step", "grad_evals"}
        for rec in lines:
            for k in required_keys:
                self.assertIn(k, rec, msg=f"Missing key {k} in iter_metrics")
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    self.assertTrue(np.isfinite(v), msg=f"iter_metrics[{k}]={v} not finite")
        # First line should have full diagnostics (step 1 or first log)
        first = lines[0]
        for key in ("U_train", "grad_norm", "theta_norm", "f_nll", "f_margin", "snr"):
            self.assertIn(key, first, msg=f"First iter_metrics line missing {key}")
            self.assertTrue(np.isfinite(first[key]))
        self.assertGreater(first["U_train"], 0.0)
        self.assertLess(first["U_train"], 1e7)
        self.assertGreater(first["grad_norm"], 0.0)

        # samples_metrics: exact count and plausible probe values
        data = np.load(run_dir / "samples_metrics.npz")
        self.assertIn("step", data)
        self.assertIn("f_nll", data)
        expected_count = (config.T - config.B) // config.S
        self.assertEqual(len(data["step"]), expected_count,
                         msg=f"Expected (T-B)/S = {expected_count} saved samples")
        steps = data["step"]
        self.assertTrue(np.all(np.diff(steps) == config.S), msg="Saved steps should be spaced by S")
        f_nll = data["f_nll"]
        self.assertTrue(np.all(np.isfinite(f_nll)))
        self.assertTrue(np.all(f_nll >= 0.0), msg="f_nll (CE) should be non-negative")
        self.assertTrue(np.all(f_nll < 1e3), msg="f_nll should not explode")
