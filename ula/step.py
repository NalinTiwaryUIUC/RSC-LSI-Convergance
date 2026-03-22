"""
One ULA step: theta = theta - h * grad_U + sqrt(2*h) * noise.
Updates model in place. Step size h is kept small (e.g. 1e-5) for discretization.
Supports gradient accumulation over microbatches when num_microbatches > 1.
"""
from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn

from models.params import flatten_params, unflatten_like
from .grad_flat import compute_grad_U_flat


def overdamped_step(
    model: nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    h: float,
    device: torch.device,
    noise_scale: float = 1.0,
    drift_scale: float = 1.0,
    beta: float = 1.0,
    return_U: bool = False,
    generator: torch.Generator | None = None,
    ce_reduction: str = "mean",
    clip_grad_norm: float | None = None,
    num_microbatches: int = 1,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """Perform one overdamped Langevin (ULA) step. Modifies model parameters in place."""
    theta_prev = flatten_params(model).clone()
    U, grads, grad_norm_pre_clip, grad_norm_post_clip = compute_grad_U_flat(
        model,
        train_data,
        alpha,
        device,
        ce_reduction,
        beta,
        clip_grad_norm,
        num_microbatches,
        microbatch_size,
    )

    noise_std = (2.0 * h) ** 0.5 * noise_scale
    drift = drift_scale * (-h * grads)
    noise = noise_std * torch.randn(
        theta_prev.shape, device=device, dtype=theta_prev.dtype, generator=generator
    )
    delta = drift + noise
    theta_new = theta_prev + delta
    unflatten_like(theta_new, model)

    out: dict[str, Any] = {}
    if return_U:
        out["U"] = float(U) if isinstance(U, float) else U
        grad_norm = grads.norm().item()
        theta_norm = theta_new.norm().item()
        out["grad_norm"] = grad_norm
        out["theta_norm"] = theta_norm
        out["drift_step_norm"] = drift.norm().item()
        out["noise_step_norm"] = noise.norm().item()
        out["delta_theta_norm"] = delta.norm().item()
    if clip_grad_norm is not None and grad_norm_pre_clip is not None and grad_norm_post_clip is not None:
        out["grad_norm_pre_clip"] = grad_norm_pre_clip
        out["grad_norm_post_clip"] = grad_norm_post_clip
    return out


# Backward-compatible name
ula_step = overdamped_step
