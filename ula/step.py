"""
One ULA step: theta = theta - h * grad_U + sqrt(2*h) * noise.
Updates model in place. Step size h is kept small (e.g. 1e-5) for discretization.
"""
from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn

from models.params import flatten_params, unflatten_like
from .potential import compute_U


def ula_step(
    model: nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    h: float,
    device: torch.device,
    return_U: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Perform one ULA step. Modifies model parameters in place. Returns dict with optional U."""
    theta_prev = flatten_params(model).clone()
    model.zero_grad(set_to_none=True)
    U = compute_U(model, train_data, alpha, device)
    U.backward()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    theta_new = theta_prev - h * grads + (2.0 * h) ** 0.5 * torch.randn(
        theta_prev.shape, device=device, dtype=theta_prev.dtype, generator=generator
    )
    unflatten_like(theta_new, model)

    out: dict[str, Any] = {}
    if return_U:
        out["U"] = U.detach().item()
    return out
