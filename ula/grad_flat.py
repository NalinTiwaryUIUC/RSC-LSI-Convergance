"""
Shared full-batch (or microbatch-accumulated) gradient of beta * U(theta), flat.
Used by overdamped ULA and BAOAB underdamped steps.
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    beta: float = 1.0,
) -> float:
    """Accumulate grad over microbatches; return scalar U after backward (beta * U in loss sense)."""
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
            (beta * ce / n).backward()
            U_data_sum += ce.item() * (y_mb.shape[0] / n)
        else:
            (beta * ce).backward()
            U_data_sum += ce.item()
    reg = (alpha / 2.0) * sum((p * p).sum() for p in model.parameters())
    (beta * reg).backward()
    U = beta * (U_data_sum + reg.item())
    return U


def compute_grad_U_flat(
    model: nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    device: torch.device,
    ce_reduction: str,
    beta: float,
    clip_grad_norm: float | None,
    num_microbatches: int,
    microbatch_size: int | None,
) -> tuple[float, torch.Tensor, float | None, float | None]:
    """
    Compute grad of (beta * U) w.r.t. parameters, return flat concatenated gradient and U value.
    Clips gradients in-place when clip_grad_norm is set.
    """
    if isinstance(train_data, tuple):
        x, y = train_data
    else:
        x, y = next(iter(train_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    n_train = x.shape[0]
    if microbatch_size is None:
        microbatch_size = n_train

    if num_microbatches > 1:
        U = _compute_U_and_grad_microbatch(
            model, x, y, alpha, device, ce_reduction,
            microbatch_size, num_microbatches, beta=beta,
        )
    else:
        model.zero_grad(set_to_none=True)
        U_tensor = compute_U(model, train_data, alpha, device, ce_reduction=ce_reduction)
        (beta * U_tensor).backward()
        U = (beta * U_tensor).item()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_norm_pre_clip: float | None = grads.norm().item() if clip_grad_norm is not None else None
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    grad_norm_post_clip: float | None = grads.norm().item() if clip_grad_norm is not None else None
    return U, grads, grad_norm_pre_clip, grad_norm_post_clip
