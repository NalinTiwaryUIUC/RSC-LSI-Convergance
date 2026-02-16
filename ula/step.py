"""
One ULA step: theta = theta - h * grad_U + sqrt(2*h) * noise.
Updates model in place; returns U (optional), theta_dist, bt_margin, inside_bt.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.params import flatten_params, unflatten_like
from .domain_bt import compute_bt_metrics
from .potential import compute_U


def ula_step(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    alpha: float,
    h: float,
    theta0_flat: torch.Tensor,
    rho2: float,
    device: torch.device,
    return_U: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """
    Perform one ULA step. Modifies model parameters in place.
    Returns dict with theta_dist, bt_margin, inside_bt, and optionally U.
    """
    model.zero_grad(set_to_none=True)
    U = compute_U(model, train_loader, alpha, device)
    U.backward()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    theta = flatten_params(model).clone()
    noise = torch.randn(theta.shape, device=device, dtype=theta.dtype, generator=generator)
    theta = theta - h * grads + (2.0 * h) ** 0.5 * noise
    unflatten_like(theta, model)

    theta_dist, bt_margin, inside_bt = compute_bt_metrics(theta, theta0_flat, rho2)
    out = {
        "theta_dist": theta_dist,
        "bt_margin": bt_margin,
        "inside_bt": inside_bt,
    }
    if return_U:
        out["U"] = U.detach().item()
    return out
