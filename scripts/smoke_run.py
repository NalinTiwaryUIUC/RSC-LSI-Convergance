"""
Smoke run: short chain to validate pipeline (T=500, B=100, S=50, n=128, w=0.5, 1 chain).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig, ensure_directories, get_device
from data import get_probe_loader, get_train_loader
from run.chain import run_chain


def main() -> None:
    ensure_directories()
    device = get_device()
    use_gpu = device.type == "cuda"
    config = RunConfig(
        n_train=128,
        probe_size=32,
        width_multiplier=0.5,
        h=1e-5,
        alpha=1e-2,
        T=500,
        B=100,
        S=50,
        K=1,
        chain_seed=0,
        dataset_seed=42,
    )
    train_loader = get_train_loader(
        config.n_train,
        batch_size=config.n_train,
        dataset_seed=config.dataset_seed,
        data_dir=config.data_dir,
        root="./data",
        pin_memory=use_gpu,
    )
    probe_loader = get_probe_loader(
        config.probe_size,
        dataset_seed=config.dataset_seed + 1,
        data_dir=config.data_dir,
        root="./data",
        pin_memory=use_gpu,
    )
    run_dir = Path("experiments/runs") / "smoke_w0.5_n128_h1e-5_chain0"
    run_chain(
        config, chain_id=0, run_dir=run_dir,
        train_loader=train_loader, probe_loader=probe_loader, device=device,
    )
    print("Smoke run done:", run_dir)
    print("Check run_config.yaml, iter_metrics.jsonl, samples_metrics.npz")


if __name__ == "__main__":
    main()
