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
from run.diagnostics import (
    bn_buffer_stats,
    grad_vector_stats,
    param_vector_stats,
    probe_metrics,
    register_activation_hooks,
    basic_block_predicate,
)
from run.persistence import dump_failure, write_iter_metrics, write_run_config, write_samples_metrics
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
    pretrain_path: str | Path | None = None,
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
    d = flatten_params(model).numel()
    config.param_count = d
    config.ou_radius_pred = math.sqrt(d / config.alpha)
    write_run_config(config, run_dir)

    if pretrain_path is not None:
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        theta0 = flatten_params(model).clone().detach()
        sigma_init = config.sigma_init_scale * (theta0.std().item() + 1e-8)
        g = torch.Generator(device=device).manual_seed(config.chain_seed + chain_id * 1000)
        noise = torch.randn(d, device=device, generator=g) * sigma_init
        unflatten_like(theta0 + noise, model)
        _pretrain_model(
            model,
            train_data,
            steps=config.pretrain_steps,
            lr=config.pretrain_lr,
            device=device,
        )

    theta0_flat = flatten_params(model).clone().detach()  # reference for probes
    theta0_norm_sq = (theta0_flat.norm().item()) ** 2  # for OU theta_norm_sq_pred
    if config.bn_mode == "eval":
        model.eval()
    elif config.bn_mode == "batchstat_frozen":
        from run.bn_mode import set_bn_batchstats_freeze_buffers

        set_bn_batchstats_freeze_buffers(model)
    else:
        raise ValueError(f"Unknown bn_mode: {config.bn_mode}")

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
    dist_to_ref_at_step1: float | None = None
    nll_probe_at_step1: float | None = None
    act_logger, act_hooks = register_activation_hooks(model, basic_block_predicate)
    try:
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

                # Diagnostics: param/grad stats, probe stability (model has grads after ula_step)
                params = list(model.parameters())
                theta_n, theta_max, finite_params, nan_params = param_vector_stats(params)
                grad_norm_d, grad_max, finite_grad, nan_grads = grad_vector_stats(params)
                pm = probe_metrics(model, x_probe, y_probe)
                finite_loss = U_now is not None and bool(torch.isfinite(torch.tensor(U_now)))

                # Failure guard: dump and return on first non-finite
                if not (finite_loss and finite_params and finite_grad):
                    dump_failure(run_dir, step, model, {
                        "h": config.h, "alpha": config.alpha, "noise_scale": config.noise_scale,
                        "U_train": U_now, "finite_loss": finite_loss, "finite_params": finite_params,
                        "finite_grad": finite_grad, "nan_count_params": nan_params, "nan_count_grads": nan_grads,
                    })
                    write_iter_metrics(
                        step=step, grad_evals=step, run_dir=run_dir,
                        U_train=U_now, grad_norm=grad_n, theta_norm=out.get("theta_norm"),
                        f_nll=vals.get("f_nll"), f_margin=vals.get("f_margin"), snr=snr_val, delta_U=delta_U_val,
                        finite_loss=finite_loss, finite_params=finite_params, finite_grad=finite_grad,
                        nan_count_params=nan_params, nan_count_grads=nan_grads,
                    )
                    write_samples_metrics(
                        run_dir, steps_saved, grad_evals_saved, f_values, grad_norm_sq,
                    )
                    return

                # Capture step-1 baselines for stop flags
                if step == 1:
                    f_dist_val = vals.get("f_dist")
                    if f_dist_val is not None and f_dist_val >= 0:
                        dist_to_ref_at_step1 = math.sqrt(f_dist_val)
                    nll_probe_at_step1 = pm["nll_probe"]

                # A. U decomposition + scale sanity (U uses sum CE)
                theta_norm_val = out.get("theta_norm")
                U_prior = (0.5 * config.alpha * (theta_norm_val**2)) if theta_norm_val is not None else None
                U_data = (U_now - U_prior) if (U_now is not None and U_prior is not None) else None
                ce_sum_train = U_data  # U = sum CE + U_prior
                ce_mean_train = (U_data / config.n_train) if (U_data is not None and config.n_train > 0) else None
                U_data_minus_ce = (U_data - ce_sum_train) if ce_sum_train is not None else None

                # B. Locality relative to pretrained checkpoint
                f_dist_val = vals.get("f_dist")
                dist_to_ref_sq = f_dist_val
                dist_to_ref = math.sqrt(f_dist_val) if f_dist_val is not None and f_dist_val >= 0 else None

                # C. OU "pure prior diffusion" test
                ou_radius_pred = config.ou_radius_pred
                theta_norm_over_ou = (theta_norm_val / ou_radius_pred) if (theta_norm_val is not None and ou_radius_pred) else None
                t = step * config.h
                theta_norm_sq_pred_ou = (
                    math.exp(-2 * config.alpha * t) * theta0_norm_sq
                    + (d / config.alpha) * (1 - math.exp(-2 * config.alpha * t))
                )
                theta_norm_sq_over_pred_ou = (
                    (theta_norm_val**2) / theta_norm_sq_pred_ou
                    if (theta_norm_val is not None and theta_norm_sq_pred_ou > 0)
                    else None
                )

                # D. Stop-early flags
                bad_locality = (
                    dist_to_ref is not None
                    and dist_to_ref_at_step1 is not None
                    and dist_to_ref_at_step1 > 0
                    and dist_to_ref > 5 * dist_to_ref_at_step1
                )
                bad_prediction = (
                    nll_probe_at_step1 is not None
                    and pm["nll_probe"] > 2 * nll_probe_at_step1 + 2.0
                )
                abort_suggested = bad_locality or bad_prediction

                extra: Dict[str, Any] = {
                    "theta_max_abs": theta_max,
                    "finite_params": finite_params,
                    "finite_grad": finite_grad,
                    "finite_loss": finite_loss,
                    "nan_count_params": nan_params,
                    "nan_count_grads": nan_grads,
                    "gradU_max_abs": grad_max,
                    "logit_max_abs": pm["logit_max_abs"],
                    "logsumexp_max": pm["logsumexp_max"],
                    "pmax_mean": pm["pmax_mean"],
                    "nll_probe": pm["nll_probe"],
                    "margin_probe": pm["margin_probe"],
                    "logits_finite": pm["logits_finite"],
                    "drift_step_norm": out.get("drift_step_norm"),
                    "noise_step_norm": out.get("noise_step_norm"),
                    "delta_theta_norm": out.get("delta_theta_norm"),
                    # A. U decomposition
                    "U_prior": U_prior,
                    "U_data": U_data,
                    "ce_mean_train": ce_mean_train,
                    "ce_sum_train": ce_sum_train,
                    "U_data_minus_ce": U_data_minus_ce,
                    # B. Locality
                    "dist_to_ref": dist_to_ref,
                    "dist_to_ref_sq": dist_to_ref_sq,
                    # C. OU test
                    "theta_norm_over_ou": theta_norm_over_ou,
                    "theta_norm_sq_pred_ou": theta_norm_sq_pred_ou,
                    "theta_norm_sq_over_pred_ou": theta_norm_sq_over_pred_ou,
                    # D. Stop flags
                    "bad_locality": bad_locality,
                    "bad_prediction": bad_prediction,
                    "abort_suggested": abort_suggested,
                }
                if step % 500 == 0:
                    bn_st = bn_buffer_stats(model)
                    extra["bn_runmean_maxabs"] = bn_st["bn_runmean_maxabs"]
                    extra["bn_runvar_maxabs"] = bn_st["bn_runvar_maxabs"]
                    extra["bn_buffers_finite"] = bn_st["bn_buffers_finite"]
                if step % 200 == 0 and act_logger.stats:
                    # Flatten activation stats for JSON (take max act_max_abs across blocks)
                    act_max = max(s.get("act_max_abs", 0.0) for s in act_logger.stats.values())
                    extra["act_max_abs"] = act_max
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
                    **extra,
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
    finally:
        for h in act_hooks:
            h.remove()

    write_samples_metrics(
        run_dir,
        steps_saved,
        grad_evals_saved,
        f_values,
        grad_norm_sq,
    )
    return None
