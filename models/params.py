"""
Flatten / unflatten model parameters for ULA and probes.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Return a 1D tensor of all model parameters (view of concatenated params)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_like(vector: torch.Tensor, model: nn.Module) -> None:
    """
    Write vector into model parameters in place.
    vector must have same total numel as sum of param numels.
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vector[offset : offset + numel].view_as(p.data))
        offset += numel
    if offset != vector.numel():
        raise ValueError(
            f"vector numel {vector.numel()} != total param numel {offset}"
        )


def param_count(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())
