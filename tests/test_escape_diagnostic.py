"""
Smoke tests for escape diagnostic: pretrain snapshots and I3 init perturbation scale.
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from config import RunConfig
from data import get_probe_loader, get_train_loader
from run.chain import run_chain

ROOT = Path(__file__).resolve().parents[1]


class TestPretrainSnapshots(unittest.TestCase):
    def test_snapshot_steps_write_files(self):
        """--snapshot-steps creates *_step*.pt under --snapshot-dir."""
        data_dir = tempfile.mkdtemp(prefix="lsi_snap_")
        snap_dir = Path(data_dir) / "snap"
        snap_dir.mkdir()
        final_pt = Path(data_dir) / "final.pt"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/pretrain.py"),
            "--width",
            "0.5",
            "--n_train",
            "64",
            "--pretrain-steps",
            "4",
            "--snapshot-steps",
            "2,4",
            "--snapshot-dir",
            str(snap_dir),
            "-o",
            str(final_pt),
            "--data_dir",
            data_dir,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        w_str = "0.5"
        self.assertTrue((snap_dir / f"pretrain_w{w_str}_n64_nb2_step2.pt").is_file())
        self.assertTrue((snap_dir / f"pretrain_w{w_str}_n64_nb2_step4.pt").is_file())
        self.assertTrue(final_pt.is_file())
        payload = __import__("torch").load(final_pt, map_location="cpu", weights_only=True)
        self.assertIn("state_dict", payload)


class TestInitPerturbI3(unittest.TestCase):
    def test_init_perturb_sigma_scales_dist_to_ref(self):
        """With loaded checkpoint, dist_to_ref_over_sqrt_d after step 1 is O(sigma) when drift/noise are tiny."""
        data_dir = tempfile.mkdtemp(prefix="lsi_i3_")
        snap_dir = Path(data_dir) / "snap"
        snap_dir.mkdir()
        final_pt = Path(data_dir) / "final.pt"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/pretrain.py"),
                "--width",
                "0.5",
                "--n_train",
                "64",
                "--pretrain-steps",
                "2",
                "--data_dir",
                data_dir,
                "-o",
                str(final_pt),
            ],
            cwd=str(ROOT),
            check=True,
        )

        run_dir = Path(data_dir) / "run"
        config = RunConfig(
            n_train=64,
            probe_size=16,
            width_multiplier=0.5,
            h=1e-15,
            T=1,
            B=0,
            S=1,
            log_every=1,
            pretrain_steps=0,
            data_dir=data_dir,
            noise_scale=0.0,
            chain_seed=12345,
            init_perturb_sigma=0.02,
            init_perturb_reference="checkpoint",
        )
        train_loader = get_train_loader(
            config.n_train,
            batch_size=config.n_train,
            data_dir=data_dir,
            root="./data",
        )
        probe_loader = get_probe_loader(config.probe_size, data_dir=data_dir, root="./data")
        run_chain(
            config,
            chain_id=0,
            run_dir=run_dir,
            train_loader=train_loader,
            probe_loader=probe_loader,
            pretrain_path=final_pt,
        )
        with open(run_dir / "iter_metrics.jsonl") as f:
            first = json.loads(f.readline())
        d = float(first["dist_to_ref_over_sqrt_d"])
        self.assertTrue(math.isfinite(d))
        # E[||σ ξ||/√d] ≈ σ; one ULA step adds negligible drift at h=1e-15, noise_scale=0
        self.assertGreater(d, 0.005)
        self.assertLess(d, 0.06)


class TestAnalyzeEscapeDiagnostic(unittest.TestCase):
    def test_chain_prefix_strips_chain_id(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_escape_diagnostic import chain_prefix  # noqa: E402

        self.assertEqual(chain_prefix("w1_n64_h1e-5_T10_a0.3_b1p0_initI1_chain3"), "w1_n64_h1e-5_T10_a0.3_b1p0_initI1")

    def test_preset_threshold_grid_crossings(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_escape_diagnostic import compute_preset_taus_for_chain  # noqa: E402

        recs = [
            {
                "step": 0,
                "dist_to_ref_over_sqrt_d": 0.01,
                "dist_to_ref_over_ou_radius": 0.01,
                "nll_probe_mean": 2.0,
                "f_margin": -0.1,
            },
            {
                "step": 100,
                "dist_to_ref_over_sqrt_d": 0.12,
                "dist_to_ref_over_ou_radius": 0.07,
                "nll_probe_mean": 2.4,
                "f_margin": -0.8,
            },
        ]
        out = compute_preset_taus_for_chain(recs, fill_missing_tau="none")
        self.assertEqual(out["d_sqrt_ge_0p05"], 100)
        self.assertEqual(out["d_sqrt_ge_0p1"], 100)
        self.assertIsNone(out["d_sqrt_ge_0p15"])
        self.assertEqual(out["ou_ge_0p03"], 100)
        self.assertEqual(out["nll_ge_init_plus_0p25"], 100)
        self.assertIsNone(out["nll_ge_init_plus_0p5"])
        self.assertEqual(out["f_margin_le_init_minus_0p5"], 100)
        self.assertIsNone(out["f_margin_le_init_minus_1"])

    def test_absolute_predictive_thresholds(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_escape_diagnostic import compute_preset_taus_for_chain  # noqa: E402

        recs = [
            {
                "step": 0,
                "dist_to_ref_over_sqrt_d": 0.01,
                "dist_to_ref_over_ou_radius": 0.01,
                "nll_probe_mean": 1.0,
                "f_margin": 0.0,
            },
            {
                "step": 50,
                "dist_to_ref_over_sqrt_d": 0.01,
                "dist_to_ref_over_ou_radius": 0.01,
                "nll_probe_mean": 1.6,
                "f_margin": -0.1,
            },
            {
                "step": 80,
                "dist_to_ref_over_sqrt_d": 0.01,
                "dist_to_ref_over_ou_radius": 0.01,
                "nll_probe_mean": 1.6,
                "f_margin": -0.4,
            },
        ]
        out = compute_preset_taus_for_chain(
            recs,
            fill_missing_tau="none",
            abs_nll_ge=(1.5,),
            abs_f_margin_le=(-0.25,),
        )
        self.assertEqual(out["nll_abs_ge_1p5"], 50)
        self.assertEqual(out["f_margin_abs_le_m0p25"], 80)

    def test_parse_csv_floats_dedupes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_escape_diagnostic import parse_csv_floats  # noqa: E402

        self.assertEqual(parse_csv_floats("1, 1, 2"), (1.0, 2.0))
        self.assertEqual(parse_csv_floats("-0.2, -0.3"), (-0.2, -0.3))

    def test_post_geom_conditional_tau(self):
        """τ_pred|geom is first step >= τ_geom with predictive crossing; Δτ = τ_pred − τ_geom."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_post_geom_predictive import compute_chain_taus  # noqa: E402

        recs = [
            {
                "step": 10,
                "dist_to_ref_over_sqrt_d": 0.02,
                "nll_probe_mean": 1.0,
                "f_margin": 0.0,
            },
            {
                "step": 100,
                "dist_to_ref_over_sqrt_d": 0.06,
                "nll_probe_mean": 1.2,
                "f_margin": -0.05,
            },
            {
                "step": 500,
                "dist_to_ref_over_sqrt_d": 0.08,
                "nll_probe_mean": 1.5,
                "f_margin": -0.25,
            },
        ]
        out = compute_chain_taus(
            recs,
            geom_d=(0.05,),
            abs_nll_ge=(1.45,),
            abs_f_margin_le=(-0.20,),
        )
        cid_nll = "geom_d0p05_nll_ge_1p45"
        cid_m = "geom_d0p05_f_margin_le_m0p2"
        self.assertEqual(out[cid_nll], (100, 500, 400))
        self.assertEqual(out[cid_m], (100, 500, 400))

    def test_post_geom_same_step_nll(self):
        """If nll threshold holds already at τ_geom row, τ_pred = τ_geom, Δτ = 0."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_post_geom_predictive import compute_chain_taus  # noqa: E402

        recs = [
            {
                "step": 50,
                "dist_to_ref_over_sqrt_d": 0.06,
                "nll_probe_mean": 1.5,
                "f_margin": 0.0,
            },
        ]
        out = compute_chain_taus(
            recs,
            geom_d=(0.05,),
            abs_nll_ge=(1.45,),
            abs_f_margin_le=(),
        )
        self.assertEqual(out["geom_d0p05_nll_ge_1p45"], (50, 50, 0))


if __name__ == "__main__":
    unittest.main()
