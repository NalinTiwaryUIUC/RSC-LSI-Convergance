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
from models import create_model
from run.chain import run_chain
from ula.potential import compute_U
import torch


def main() -> None:
    ensure_directories()
    device = get_device()
    use_gpu = device.type == "cuda"
    defaults = RunConfig()
    config = RunConfig(
        n_train=128,
        probe_size=32,
        width_multiplier=0.5,
        h=defaults.h,
        alpha=defaults.alpha,
        noise_scale=defaults.noise_scale,
        T=500,
        B=100,
        S=50,
        K=1,
        pretrain_steps=200,
        pretrain_lr=defaults.pretrain_lr,
        chain_seed=0,
        dataset_seed=defaults.dataset_seed,
        data_dir=defaults.data_dir,
    )
    train_loader = get_train_loader(
        config.n_train,
        batch_size=config.n_train,
        dataset_seed=config.dataset_seed,
        data_dir=config.data_dir,
        root="./data",
        pin_memory=use_gpu,
    )
    # Quick signal-to-noise ratio diagnostic at the starting point
    model = create_model(width_multiplier=config.width_multiplier).to(device)
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    U = compute_U(model, (x_train, y_train), alpha=config.alpha, device=device)
    U.backward()
    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    d = grads.numel()
    h = config.h
    noise_scale = config.noise_scale
    signal = h * grads.norm().item()
    noise = (2.0 * h * d) ** 0.5 * noise_scale
    snr = signal / noise if noise > 0 else float("nan")
    print(
        f"Approx SNR at init (h={h}, noise_scale={noise_scale}): "
        f"{snr:.3e} (signal={signal:.3e}, noise={noise:.3e}, d={d})"
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
