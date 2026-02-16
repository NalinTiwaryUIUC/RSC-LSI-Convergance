"""
Potential U(theta) = sum CE(theta; x_i, y_i) + (alpha/2) * ||theta||^2
Full-batch on training subset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.params import flatten_params


def compute_U(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Full-batch negative log posterior (up to constant).
    Assumes train_loader yields a single batch (full subset).
    Model in eval mode for consistency; gradients needed for ULA.
    """
    model.train()  # need gradients
    total_ce = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_ce = total_ce + F.cross_entropy(logits, y, reduction="sum")
    # (alpha/2) * ||theta||^2
    flat = flatten_params(model)
    reg = (alpha / 2.0) * (flat @ flat)
    return total_ce + reg
