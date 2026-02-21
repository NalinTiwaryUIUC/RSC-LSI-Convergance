"""
Probe functions f(θ): NLL, margin, logit projections, param projections, dist.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.params import flatten_params


def _mean_ce_on_probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Mode set by caller
    with torch.no_grad():
        logits = model(x)
    return F.cross_entropy(logits, y, reduction="mean")


def _mean_margin_on_probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Mode set by caller
    with torch.no_grad():
        logits = model(x)
    # logit[true] - max over other classes
    n = logits.size(0)
    logit_true = logits[torch.arange(n, device=logits.device), y]
    logits_masked = logits.clone()
    logits_masked[torch.arange(n, device=logits.device), y] = -float("inf")
    max_other = logits_masked.max(dim=1).values
    margin = (logit_true - max_other).mean()
    return margin


def _logits_flatten(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Mode set by caller
    with torch.no_grad():
        logits = model(x)
    return logits.reshape(-1)


def evaluate_probes(
    model: nn.Module,
    probe_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    theta0_flat: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    logit_proj: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate all probes. Returns dict probe_name -> scalar value.
    probe_data: DataLoader or (x, y) tensors already on device (pre-load for speed).
    """
    if isinstance(probe_data, tuple):
        x, y = probe_data
    else:
        (x, y) = next(iter(probe_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    theta0_flat = theta0_flat.to(device)
    v1, v2 = v1.to(device), v2.to(device)
    logit_proj = logit_proj.to(device)

    theta_flat = flatten_params(model).to(device)
    diff = theta_flat - theta0_flat

    out = {}

    # f_nll: mean CE (need grad for LSI later; here just value)
    # Mode set by caller
    with torch.no_grad():
        logits = model(x)
        out["f_nll"] = F.cross_entropy(logits, y, reduction="mean").item()

    # f_margin
    with torch.no_grad():
        n = logits.size(0)
        logit_true = logits[torch.arange(n, device=logits.device), y]
        logits_masked = logits.clone()
        logits_masked[torch.arange(n, device=logits.device), y] = -float("inf")
        max_other = logits_masked.max(dim=1).values
        out["f_margin"] = (logit_true - max_other).mean().item()

    # f_pc1, f_pc2: random projection of flattened logits
    logit_vec = logits.reshape(-1)
    out["f_pc1"] = (logit_proj[0] @ logit_vec).item()
    out["f_pc2"] = (logit_proj[1] @ logit_vec).item()

    # f_proj1, f_proj2
    out["f_proj1"] = (v1 @ diff).item()
    out["f_proj2"] = (v2 @ diff).item()

    # f_dist
    out["f_dist"] = (diff @ diff).item()

    return out


def _theta_flat_with_grad(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Concatenate params so result has grad (for autograd)."""
    return torch.cat([p.reshape(-1) for p in model.parameters()]).to(device)


def get_probe_value_for_grad(
    model: nn.Module,
    probe_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    theta0_flat: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    logit_proj: torch.Tensor,
    probe_name: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute a single probe value *with* grad (for computing grad_norm_sq).
    probe_data: DataLoader or (x, y) tensors already on device.
    """
    if isinstance(probe_data, tuple):
        x, y = probe_data
    else:
        (x, y) = next(iter(probe_data))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    theta0_flat = theta0_flat.to(device)
    v1, v2 = v1.to(device), v2.to(device)
    logit_proj = logit_proj.to(device)

    # Mode set by caller
    theta_flat = _theta_flat_with_grad(model, device)
    diff = theta_flat - theta0_flat

    if probe_name == "f_nll":
        logits = model(x)
        return F.cross_entropy(logits, y, reduction="mean")
    if probe_name == "f_margin":
        logits = model(x)
        n = logits.size(0)
        logit_true = logits[torch.arange(n, device=logits.device), y]
        logits_masked = logits.clone()
        logits_masked[torch.arange(n, device=logits.device), y] = -float("inf")
        max_other = logits_masked.max(dim=1).values
        return (logit_true - max_other).mean()
    if probe_name == "f_pc1":
        logits = model(x)
        return (logit_proj[0] @ logits.reshape(-1)).squeeze()
    if probe_name == "f_pc2":
        logits = model(x)
        return (logit_proj[1] @ logits.reshape(-1)).squeeze()
    if probe_name == "f_proj1":
        return (v1 @ diff).squeeze()
    if probe_name == "f_proj2":
        return (v2 @ diff).squeeze()
    if probe_name == "f_dist":
        return (diff @ diff).squeeze()
    raise ValueError(f"Unknown probe: {probe_name}")
