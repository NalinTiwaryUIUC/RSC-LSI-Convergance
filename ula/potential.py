"""
Potential U(theta) = sum CE(theta; x_i, y_i) + (alpha/2) * ||theta||^2
Full-batch on training subset.
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.params import flatten_params


def compute_U(
    model: nn.Module,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Full-batch negative log posterior (up to constant).
    train_data: either DataLoader (yields single batch) or (x, y) tensors already on device.
    Pass (x, y) pre-loaded on GPU to avoid host-to-device transfer every step.
    """
    model.train()
    if isinstance(train_data, tuple):
        x, y = train_data
    else:
        x, y = next(iter(train_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    logits = model(x)
    total_ce = F.cross_entropy(logits, y, reduction="sum")
    flat = flatten_params(model)
    reg = (alpha / 2.0) * (flat @ flat)
    return total_ce + reg
