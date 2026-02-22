"""
One ULA step: theta = theta - h * grad_U + sqrt(2*h) * noise.
Updates model in place. Step size h is kept small (e.g. 1e-5) for discretization.
Supports gradient accumulation over microbatches when num_microbatches > 1.
"""
from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.params import flatten_params, unflatten_like
from .potential import compute_U


def _compute_U_and_grad_microbatch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
    ce_reduction: str,
    microbatch_size: int,
    num_microbatches: int,
) -> float:
    """Accumulate grad over microbatches; return full-batch U (scalar)."""
    n = x.shape[0]
    model.zero_grad(set_to_none=True)
    U_data_sum = 0.0
    for k in range(num_microbatches):
        start = k * microbatch_size
        end = start + microbatch_size
        x_mb = x[start:end]
        y_mb = y[start:end]
        logits = model(x_mb)
        ce = F.cross_entropy(logits, y_mb, reduction=ce_reduction)
        if ce_reduction == "mean":
            # Full-batch mean = (1/n)*sum CE_i; grad(mean) = (1/n)*sum grad(CE_i). Backward (ce/n) so we add (1/n)*grad(ce_chunk).
            (ce / n).backward()
            U_data_sum += ce.item() * (y_mb.shape[0] / n)
        else:
            ce.backward()
            U_data_sum += ce.item()
    # Add prior gradient: d/dθ (alpha/2 ||θ||^2) = alpha*θ (use model.parameters() so backward accumulates)
    reg = (alpha / 2.0) * sum((p * p).sum() for p in model.parameters())
    reg.backward()
    U = U_data_sum + reg.item()
    return U


def ula_step(
    model: nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    h: float,
    device: torch.device,
    noise_scale: float = 1.0,
    return_U: bool = False,
    generator: torch.Generator | None = None,
    ce_reduction: str = "mean",
    clip_grad_norm: float | None = None,
    num_microbatches: int = 1,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """Perform one ULA step. Modifies model parameters in place. Returns dict with optional U.
    When num_microbatches > 1, accumulates gradient over microbatches (partition-invariant with bn_mode=eval).
    S3: If clip_grad_norm is set, clips grads in place and returns grad_norm_pre_clip, grad_norm_post_clip.
    """
    if isinstance(train_data, tuple):
        x, y = train_data
    else:
        x, y = next(iter(train_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    n_train = x.shape[0]
    if microbatch_size is None:
        microbatch_size = n_train

    theta_prev = flatten_params(model).clone()
    if num_microbatches > 1:
        U = _compute_U_and_grad_microbatch(
            model, x, y, alpha, device, ce_reduction,
            microbatch_size, num_microbatches,
        )
    else:
        model.zero_grad(set_to_none=True)
        U_tensor = compute_U(model, train_data, alpha, device, ce_reduction=ce_reduction)
        U_tensor.backward()
        U = U_tensor.item()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_norm_pre_clip: float | None = grads.norm().item() if clip_grad_norm is not None else None
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_norm_post_clip: float | None = grads.norm().item() if clip_grad_norm is not None else None

    noise_std = (2.0 * h) ** 0.5 * noise_scale
    drift = -h * grads
    noise = noise_std * torch.randn(
        theta_prev.shape, device=device, dtype=theta_prev.dtype, generator=generator
    )
    delta = drift + noise
    theta_new = theta_prev + delta
    unflatten_like(theta_new, model)

    out: dict[str, Any] = {}
    if return_U:
        out["U"] = float(U) if isinstance(U, float) else U.detach().item()
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
