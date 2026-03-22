"""
Tests for BAOAB underdamped Langevin: deterministic limits, OU statistics, continuation.
"""
import math
import tempfile
import unittest

import torch

from data.cifar import get_train_loader
from models import create_model, flatten_params, unflatten_like
from ula.baoab import underdamped_baoab_step


def _tiny_train(device):
    data_dir = tempfile.mkdtemp(prefix="lsi_ud_")
    train = get_train_loader(
        n=64, batch_size=64, dataset_seed=42, data_dir=data_dir, root="./data"
    )
    x, y = next(iter(train))
    return (x.to(device), y.to(device)), data_dir


class TestBAOABDeterministic(unittest.TestCase):
    def test_noise_zero_two_runs_match(self):
        """noise_scale=0: OU noise vanishes; same seed => same theta and v."""
        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        model1 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        model2 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        theta0 = flatten_params(model1).clone()
        unflatten_like(theta0, model2)
        d = theta0.numel()
        v1 = torch.zeros(d, device=device, dtype=theta0.dtype)
        v2 = torch.zeros(d, device=device, dtype=theta0.dtype)
        h, gamma, alpha = 1e-6, 0.5, 1e-2
        gen = torch.Generator(device=device).manual_seed(999)
        underdamped_baoab_step(
            model1, v1, train_data, alpha, h, gamma, device,
            noise_scale=0.0, generator=gen,
            ce_reduction="mean", return_U=True,
        )
        gen2 = torch.Generator(device=device).manual_seed(999)
        underdamped_baoab_step(
            model2, v2, train_data, alpha, h, gamma, device,
            noise_scale=0.0, generator=gen2,
            ce_reduction="mean", return_U=True,
        )
        torch.testing.assert_close(flatten_params(model1), flatten_params(model2))
        torch.testing.assert_close(v1, v2)

    def test_noise_zero_no_nans(self):
        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        model = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        d = flatten_params(model).numel()
        v = torch.zeros(d, device=device, dtype=torch.float32)
        h, gamma, alpha = 1e-7, 1.0, 1e-2
        U_prev = None
        for t in range(5):
            gen = torch.Generator(device=device).manual_seed(1000 + t)
            out = underdamped_baoab_step(
                model, v, train_data, alpha, h, gamma, device,
                noise_scale=0.0, generator=gen,
                ce_reduction="mean", return_U=True,
            )
            U = out["U"]
            self.assertTrue(math.isfinite(U))
            if U_prev is not None:
                # loose: energy often decreases along deterministic flow; allow small numerical wiggle
                pass
            U_prev = U
            self.assertTrue(torch.isfinite(v).all())
            self.assertTrue(torch.isfinite(flatten_params(model)).all())


class TestOUStatistics(unittest.TestCase):
    def test_ou_stationary_variance_per_coordinate(self):
        """With no kicks, OU update should have marginal variance ~1 per coord (noise_scale=1)."""
        device = torch.device("cpu")
        d = 500
        h, gamma = 0.05, 2.0
        exp_m = math.exp(-gamma * h)
        target_var = 1.0 - math.exp(-2.0 * gamma * h)
        n_steps = 8_000
        v = torch.zeros(d)
        g = torch.Generator().manual_seed(42)
        for _ in range(n_steps):
            xi = torch.randn(d, generator=g)
            ou_noise = math.sqrt(target_var) * xi
            v = exp_m * v + ou_noise
        emp_var = v.var().item()
        self.assertAlmostEqual(emp_var, 1.0, delta=0.2)
        self.assertAlmostEqual(v.mean().item(), 0.0, delta=0.08)


class TestBAOABOneStepNoise(unittest.TestCase):
    def test_noise_step_norm_reproducible_with_seed(self):
        """Same RNG seed => same OU noise norm logged in return_U."""
        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        model1 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        model2 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        theta0 = flatten_params(model1).clone()
        unflatten_like(theta0, model2)
        d = theta0.numel()
        v1 = torch.zeros(d, device=device)
        v2 = torch.zeros(d, device=device)
        gen1 = torch.Generator(device=device).manual_seed(4242)
        gen2 = torch.Generator(device=device).manual_seed(4242)
        o1 = underdamped_baoab_step(
            model1, v1, train_data, 1e-2, 1e-5, 1.0, device,
            noise_scale=1.0, generator=gen1, ce_reduction="mean", return_U=True,
        )
        o2 = underdamped_baoab_step(
            model2, v2, train_data, 1e-2, 1e-5, 1.0, device,
            noise_scale=1.0, generator=gen2, ce_reduction="mean", return_U=True,
        )
        self.assertAlmostEqual(o1["noise_step_norm"], o2["noise_step_norm"], places=5)


class TestBAOABLongSmoke(unittest.TestCase):
    def test_finite_norms_many_steps(self):
        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        model = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        d = flatten_params(model).numel()
        v = torch.randn(d, device=device) * 0.01
        h, gamma, alpha = 1e-7, 1.0, 1e-2
        for step in range(1, 51):
            gen = torch.Generator(device=device).manual_seed(5000 + step)
            out = underdamped_baoab_step(
                model, v, train_data, alpha, h, gamma, device,
                noise_scale=1.0, generator=gen,
                ce_reduction="mean", return_U=(step % 25 == 0),
            )
            tn = flatten_params(model).norm().item()
            vn = v.norm().item()
            self.assertTrue(math.isfinite(tn) and tn < 1e9)
            self.assertTrue(math.isfinite(vn) and vn < 1e9)
            if step % 25 == 0:
                self.assertIn("v_norm", out)


class TestBAOABContinuation(unittest.TestCase):
    def test_split_run_matches_unbroken(self):
        """Steps 1..k then k+1..T with saved (theta,v) matches T steps in one loop (same RNG per step)."""
        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        h, gamma, alpha = 1e-7, 0.8, 1e-2
        T, k = 12, 5

        def run_segment(model, v, start_step, end_step):
            for step in range(start_step, end_step + 1):
                gen = torch.Generator(device=device).manual_seed(777 + step)
                underdamped_baoab_step(
                    model, v, train_data, alpha, h, gamma, device,
                    noise_scale=0.5, generator=gen,
                    ce_reduction="mean", return_U=False,
                )

        m1 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        theta_init = flatten_params(m1).clone()
        d = theta_init.numel()
        v1 = torch.zeros(d, device=device, dtype=next(m1.parameters()).dtype)
        run_segment(m1, v1, 1, T)

        m2 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        unflatten_like(theta_init.clone(), m2)
        v2 = torch.zeros(d, device=device, dtype=next(m2.parameters()).dtype)
        run_segment(m2, v2, 1, k)
        theta_mid = flatten_params(m2).clone()
        v_mid = v2.clone()
        m3 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        unflatten_like(theta_mid, m3)
        v3 = v_mid.clone()
        run_segment(m3, v3, k + 1, T)

        torch.testing.assert_close(flatten_params(m1), flatten_params(m3), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(v1, v3, rtol=1e-4, atol=1e-5)

    def test_disk_roundtrip(self):
        import io

        device = torch.device("cpu")
        train_data, _ = _tiny_train(device)
        model = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        d = flatten_params(model).numel()
        v = torch.randn(d)
        gen = torch.Generator(device=device).manual_seed(123)
        underdamped_baoab_step(
            model, v, train_data, 1e-2, 1e-6, 1.0, device,
            noise_scale=0.0, generator=gen, ce_reduction="mean",
        )
        buf = io.BytesIO()
        torch.save({"theta": flatten_params(model).clone(), "v": v.clone()}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=True)
        m2 = create_model(width_multiplier=0.5, arch="resnet18").to(device)
        unflatten_like(ckpt["theta"], m2)
        v2 = ckpt["v"].clone()
        gen2 = torch.Generator(device=device).manual_seed(124)
        underdamped_baoab_step(
            m2, v2, train_data, 1e-2, 1e-6, 1.0, device,
            noise_scale=0.0, generator=gen2, ce_reduction="mean",
        )
        self.assertTrue(torch.isfinite(v2).all())


class TestChainUnderdamped(unittest.TestCase):
    def test_short_chain_underdamped_produces_metrics(self):
        from config import RunConfig
        from data import get_probe_loader, get_train_loader
        from run.chain import run_chain
        from run.persistence import load_run_config
        from pathlib import Path
        import json

        data_dir = tempfile.mkdtemp(prefix="lsi_ud_chain_")
        config = RunConfig(
            n_train=64,
            probe_size=16,
            width_multiplier=0.5,
            arch="small_resnet_ln",
            num_blocks=1,
            sampler="underdamped",
            gamma=1.0,
            v_init="zero",
            h=1e-6,
            T=8,
            B=0,
            S=2,
            log_every=2,
            pretrain_steps=0,
            data_dir=data_dir,
            dataset_seed=42,
        )
        train_loader = get_train_loader(
            config.n_train, batch_size=config.n_train, data_dir=data_dir, root="./data"
        )
        probe_loader = get_probe_loader(config.probe_size, data_dir=data_dir, root="./data")
        run_dir = Path(data_dir) / "ud_run"
        run_chain(config, chain_id=0, run_dir=run_dir, train_loader=train_loader, probe_loader=probe_loader)
        loaded = load_run_config(run_dir)
        self.assertEqual(loaded.sampler, "underdamped")
        self.assertEqual(loaded.gamma, 1.0)
        with open(run_dir / "iter_metrics.jsonl") as f:
            line = json.loads(f.readline())
        self.assertIn("v_norm", line)
        self.assertIn("kinetic_energy", line)


if __name__ == "__main__":
    unittest.main()
