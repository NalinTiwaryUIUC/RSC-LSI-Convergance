"""
ULA chain runner: init, loop (step + log + save samples + grad norms), persist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

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
from ula.domain_bt import get_rho2
from ula.step import ula_step


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

    # Config for this chain
    config.chain_id = chain_id
    config.run_dir = str(run_dir)
    write_run_config(config, run_dir)

    # Model and init
    model = create_model(
        width_multiplier=config.width_multiplier,
        num_classes=config.num_classes,
    ).to(device)
    theta0 = flatten_params(model).clone().detach()
    d = theta0.numel()
    sigma_init = config.sigma_init_scale * (theta0.std().item() + 1e-8)
    g = torch.Generator(device=device).manual_seed(config.chain_seed + chain_id * 1000)
    noise = torch.randn(d, device=device, generator=g) * sigma_init
    unflatten_like(theta0 + noise, model)
    theta0_flat = theta0  # reference for B_t and probes
    rho2 = get_rho2(theta0_flat, factor=config.rho2_factor)

    # Projections (fixed across chain)
    v1, v2 = get_or_create_param_projections(
        d, seed=config.probe_projection_seed, data_dir=config.data_dir
    )
    probe_size = config.probe_size
    num_classes = config.num_classes
    logit_dim = probe_size * num_classes
    logit_proj = get_or_create_logit_projection(
        logit_dim, seed=config.probe_projection_seed + 1, data_dir=config.data_dir
    )

    # Accumulators for saved samples
    steps_saved: List[int] = []
    grad_evals_saved: List[int] = []
    inside_bt_saved: List[bool] = []
    bt_margin_saved: List[float] = []
    f_values: Dict[str, List[float]] = {
        "f_nll": [], "f_margin": [], "f_pc1": [], "f_pc2": [],
        "f_proj1": [], "f_proj2": [], "f_dist": [],
    }
    grad_norm_sq: Dict[str, List[float]] = {p: [] for p in PROBES_FOR_GRAD_NORM}
    saved_count = 0

    T, B, S, G = config.T, config.B, config.S, config.grad_norm_stride
    for step in range(1, T + 1):
        gen = torch.Generator(device=device).manual_seed(config.chain_seed + chain_id * 1000 + step)
        out = ula_step(
            model,
            train_loader,
            config.alpha,
            config.h,
            theta0_flat,
            rho2,
            device,
            return_U=(log_U_every is not None and step % log_U_every == 0),
            generator=gen,
        )
        write_iter_metrics(
            step=step,
            grad_evals=step,
            theta_dist=out["theta_dist"],
            bt_margin=out["bt_margin"],
            inside_bt=out["inside_bt"],
            run_dir=run_dir,
            U_train=out.get("U"),
        )

        if step % S == 0 and step > B:
            steps_saved.append(step)
            grad_evals_saved.append(step)
            inside_bt_saved.append(out["inside_bt"])
            bt_margin_saved.append(out["bt_margin"])
            vals = evaluate_probes(
                model, probe_loader, theta0_flat, v1, v2, logit_proj, device
            )
            for k, v in vals.items():
                f_values[k].append(v)
            saved_count += 1
            # Every G-th saved sample: compute grad norms for selected probes
            if saved_count % G == 0:
                for pname in PROBES_FOR_GRAD_NORM:
                    f_scalar = get_probe_value_for_grad(
                        model, probe_loader, theta0_flat, v1, v2, logit_proj, pname, device
                    )
                    gs = compute_grad_norm_sq(f_scalar, model.parameters())
                    grad_norm_sq[pname].append(gs)

    write_samples_metrics(
        run_dir,
        steps_saved,
        grad_evals_saved,
        inside_bt_saved,
        bt_margin_saved,
        f_values,
        grad_norm_sq,
    )
    return None
