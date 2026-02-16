"""
Gradient norm ||∇_θ f(θ)||^2 for proxy LSI.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def compute_grad_norm_sq(
    f_scalar: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> float:
    """
    Compute sum of squared gradients of f w.r.t. parameters.
    f_scalar must be a scalar tensor with grad_fn.
    """
    grads = torch.autograd.grad(
        f_scalar, parameters, retain_graph=retain_graph, create_graph=create_graph
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total = total + g.data.pow(2).sum().item()
    return total
