"""
Minimal chain test: T=10, B=2, S=2 to verify run_chain and persistence.
"""
import tempfile
import unittest
from pathlib import Path

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
        import numpy as np
        data = np.load(run_dir / "samples_metrics.npz")
        self.assertIn("step", data)
        self.assertIn("f_nll", data)
        self.assertGreater(len(data["step"]), 0)
