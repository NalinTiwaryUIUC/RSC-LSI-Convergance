"""
ULA chain runner: init, loop (step + log + save samples + grad norms), persist.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

from config import RunConfig, get_device
from models import create_model, flatten_params, param_count, unflatten_like
from probes import (
    PROBES_FOR_GRAD_NORM,
    compute_grad_norm_sq,
    evaluate_probes,
    get_probe_value_for_grad,
    get_or_create_logit_projection,
    get_or_create_param_projections,
)
from run.persistence import write_iter_metrics, write_run_config, write_samples_metrics
from ula.step import ula_step


def _pretrain_model(
    model: torch.nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    device: torch.device,
) -> None:
    """Simple full-batch SGD pretraining before ULA."""
    if steps <= 0:
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if isinstance(train_data, tuple):
        x, y = train_data
    else:
        x, y = next(iter(train_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()
        optimizer.step()


def run_chain(
    config: RunConfig,
    chain_id: int,
    run_dir: str | Path,
    train_loader: torch.utils.data.DataLoader,
    probe_loader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
    log_U_every: int | None = None,
) -> None:
    """
    Run one ULA chain. Writes run_config.yaml, iter_metrics.jsonl, samples_metrics.npz to run_dir.
    """
    if device is None:
        device = get_device()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # GPU optimizations: cuDNN benchmark (fixed input size) and pre-load data to device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)
    x_probe, y_probe = next(iter(probe_loader))
    x_probe = x_probe.to(device, non_blocking=True)
    y_probe = y_probe.to(device, non_blocking=True)
    train_data = (x_train, y_train)
    probe_data = (x_probe, y_probe)

    # Config for this chain
    config.chain_id = chain_id
    config.run_dir = str(run_dir)

    # Model and init
    model = create_model(
        width_multiplier=config.width_multiplier,
        num_classes=config.num_classes,
    ).to(device)
    theta0 = flatten_params(model).clone().detach()
    d = theta0.numel()
    config.param_count = d
    write_run_config(config, run_dir)

    sigma_init = config.sigma_init_scale * (theta0.std().item() + 1e-8)
    g = torch.Generator(device=device).manual_seed(config.chain_seed + chain_id * 1000)
    noise = torch.randn(d, device=device, generator=g) * sigma_init
    unflatten_like(theta0 + noise, model)

    # Optional short pretraining to move closer to a good region before sampling
    _pretrain_model(
        model,
        train_data,
        steps=config.pretrain_steps,
        lr=config.pretrain_lr,
        device=device,
    )

    theta0_flat = flatten_params(model).clone().detach()  # reference for probes

    # Projections (fixed across chain) — move to device once
    v1, v2 = get_or_create_param_projections(
        d, seed=config.probe_projection_seed, data_dir=config.data_dir
    )
    v1, v2 = v1.to(device), v2.to(device)
    probe_size = config.probe_size
    num_classes = config.num_classes
    logit_dim = probe_size * num_classes
    logit_proj = get_or_create_logit_projection(
        logit_dim, seed=config.probe_projection_seed + 1, data_dir=config.data_dir
    ).to(device)

    # Accumulators for saved samples
    steps_saved: List[int] = []
    grad_evals_saved: List[int] = []
    f_values: Dict[str, List[float]] = {
        "f_nll": [], "f_margin": [], "f_pc1": [], "f_pc2": [],
        "f_proj1": [], "f_proj2": [], "f_dist": [],
    }
    grad_norm_sq: Dict[str, List[float]] = {p: [] for p in PROBES_FOR_GRAD_NORM}
    saved_count = 0

    T, B, S, G = config.T, config.B, config.S, config.grad_norm_stride
    log_U_every = config.log_every if log_U_every is None else log_U_every
    U_prev: float | None = None
    for step in range(1, T + 1):
        gen = torch.Generator(device=device).manual_seed(config.chain_seed + chain_id * 1000 + step)
        out = ula_step(
            model,
            train_data,
            config.alpha,
            config.h,
            device,
            noise_scale=config.noise_scale,
            return_U=(step % log_U_every == 0 or step == 1),
            generator=gen,
        )
        if step % config.log_every == 0 or step == 1:
            vals = evaluate_probes(
                model, probe_data, theta0_flat, v1, v2, logit_proj, device
            )
            U_now = out.get("U")
            grad_n = out.get("grad_norm")
            # SNR = (h * ||grad||) / (sqrt(2*h*d) * noise_scale); diagnose drift vs noise
            snr_val = None
            if grad_n is not None and d > 0 and config.noise_scale > 0:
                noise_std = math.sqrt(2.0 * config.h * d) * config.noise_scale
                if noise_std > 0:
                    snr_val = (config.h * grad_n) / noise_std
            delta_U_val = None
            if U_prev is not None and U_now is not None:
                delta_U_val = U_now - U_prev
            U_prev = U_now

            write_iter_metrics(
                step=step,
                grad_evals=step,
                run_dir=run_dir,
                U_train=U_now,
                grad_norm=grad_n,
                theta_norm=out.get("theta_norm"),
                f_nll=vals.get("f_nll"),
                f_margin=vals.get("f_margin"),
                snr=snr_val,
                delta_U=delta_U_val,
            )

        if config.progress_print_every > 0 and step % config.progress_print_every == 0:
            pct = 100 * step / T
            parts = [f"chain {chain_id} step {step}/{T} ({pct:.1f}%)"]
            if "U" in out:
                parts.append(f"U={out['U']:.1f}")
            if "grad_norm" in out:
                parts.append(f"||∇U||={out['grad_norm']:.1f}")
            if "theta_norm" in out:
                parts.append(f"||θ||={out['theta_norm']:.0f}")
            if step % config.log_every == 0 or step == 1:
                parts.append(f"f_nll={vals.get('f_nll', float('nan')):.3f}")
            print(" ".join(parts))

        if step % S == 0 and step > B:
            steps_saved.append(step)
            grad_evals_saved.append(step)
            vals = evaluate_probes(
                model, probe_data, theta0_flat, v1, v2, logit_proj, device
            )
            for k, v in vals.items():
                f_values[k].append(v)
            saved_count += 1
            # Every G-th saved sample: compute grad norms for selected probes
            if saved_count % G == 0:
                for pname in PROBES_FOR_GRAD_NORM:
                    f_scalar = get_probe_value_for_grad(
                        model, probe_data, theta0_flat, v1, v2, logit_proj, pname, device
                    )
                    gs = compute_grad_norm_sq(f_scalar, model.parameters())
                    grad_norm_sq[pname].append(gs)

    write_samples_metrics(
        run_dir,
        steps_saved,
        grad_evals_saved,
        f_values,
        grad_norm_sq,
    )
    return None
