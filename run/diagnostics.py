"""
Diagnostic utilities for ULA chain: param/grad stats, probe stability, BN buffers, activation hooks.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def param_vector_stats(params: List[torch.Tensor]) -> Tuple[float, float, bool, int]:
    """theta_norm, theta_max_abs, finite, nan_count_params."""
    theta_norm_sq = 0.0
    theta_max_abs = 0.0
    finite = True
    nan_count = 0
    for p in params:
        x = p.detach()
        theta_norm_sq += x.float().pow(2).sum().item()
        theta_max_abs = max(theta_max_abs, x.abs().max().item())
        isfin = torch.isfinite(x)
        if not bool(isfin.all()):
            finite = False
            nan_count += int((~isfin).sum().item())
    return (theta_norm_sq ** 0.5), theta_max_abs, finite, nan_count


@torch.no_grad()
def grad_vector_stats(params: List[torch.Tensor]) -> Tuple[float, float, bool, int]:
    """gradU_norm, gradU_max_abs, finite_grad, nan_count_grads."""
    g_norm_sq = 0.0
    g_max_abs = 0.0
    finite = True
    nan_count = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        g_norm_sq += g.float().pow(2).sum().item()
        g_max_abs = max(g_max_abs, g.abs().max().item())
        isfin = torch.isfinite(g)
        if not bool(isfin.all()):
            finite = False
            nan_count += int((~isfin).sum().item())
    return (g_norm_sq ** 0.5), g_max_abs, finite, nan_count


@torch.no_grad()
def probe_metrics(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Dict[str, float | bool]:
    """logit_max_abs, logsumexp_max, pmax_mean, nll_probe, margin_probe, logits_finite."""
    logits = model(xb)
    logit_max_abs = logits.abs().max().item()
    lse = torch.logsumexp(logits, dim=1)
    logsumexp_max = lse.max().item()
    probs = F.softmax(logits, dim=1)
    pmax_mean = probs.max(dim=1).values.mean().item()
    nll = F.cross_entropy(logits, yb, reduction="sum").item()
    y_logit = logits.gather(1, yb.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, yb.view(-1, 1), float("-inf"))
    max_other = tmp.max(dim=1).values
    margin = (y_logit - max_other).mean().item()
    finite = bool(torch.isfinite(logits).all())
    return {
        "logit_max_abs": logit_max_abs,
        "logsumexp_max": logsumexp_max,
        "pmax_mean": pmax_mean,
        "nll_probe": nll,
        "margin_probe": margin,
        "logits_finite": finite,
    }


@torch.no_grad()
def bn_buffer_stats(model: nn.Module) -> Dict[str, float | bool]:
    """bn_runmean_maxabs, bn_runvar_maxabs, bn_buffers_finite."""
    mx_rm = 0.0
    mx_rv = 0.0
    finite = True
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            rm = m.running_mean
            rv = m.running_var
            mx_rm = max(mx_rm, rm.abs().max().item())
            mx_rv = max(mx_rv, rv.abs().max().item())
            if not bool(torch.isfinite(rm).all() and torch.isfinite(rv).all()):
                finite = False
    return {"bn_runmean_maxabs": mx_rm, "bn_runvar_maxabs": mx_rv, "bn_buffers_finite": finite}


class ActivationLogger:
    """Collects activation stats from forward hooks."""

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float | bool]] = {}

    def hook(self, name: str) -> Callable:
        def _hook(module: nn.Module, inp: object, out: torch.Tensor) -> None:
            x = out.detach()
            self.stats[name] = {
                "act_max_abs": x.abs().max().item(),
                "act_mean_abs": x.abs().mean().item(),
                "act_std": x.float().std().item(),
                "act_finite": bool(torch.isfinite(x).all()),
            }

        return _hook


def register_activation_hooks(
    model: nn.Module,
    predicate: Callable[[str, nn.Module], bool],
) -> Tuple[ActivationLogger, List[torch.utils.hooks.RemovableHandle]]:
    """Register forward hooks on modules matching predicate. Returns logger and handles."""
    alog = ActivationLogger()
    hooks = []
    for name, m in model.named_modules():
        if predicate(name, m):
            hooks.append(m.register_forward_hook(alog.hook(name)))
    return alog, hooks


def basic_block_predicate(name: str, m: nn.Module) -> bool:
    """Predicate for BasicBlock/Bottleneck in ResNet."""
    return m.__class__.__name__ in {"BasicBlock", "Bottleneck"}
