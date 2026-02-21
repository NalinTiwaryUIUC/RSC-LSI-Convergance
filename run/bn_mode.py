"""
BatchNorm mode utilities for ULA sampling.
Mode B: BN uses batch statistics (train behavior) but running buffers are frozen.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def set_bn_batchstats_freeze_buffers(model: nn.Module) -> None:
    """
    BN uses batch statistics (train behavior) but running_mean/var are frozen.
    Dropout is disabled.
    """
    model.train()  # enable BN batch stats

    for m in model.modules():
        # Disable dropout everywhere
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.eval()

        # Force BN layers into train mode (batch stats) but freeze buffers
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            m.momentum = 0.0  # freezes running_mean/var updates (buffers stop changing)


@torch.no_grad()
def snapshot_bn_running_stats(model: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot BN running_mean and running_var for drift checks."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            stats[name] = (m.running_mean.clone(), m.running_var.clone())
    return stats


@torch.no_grad()
def max_bn_running_delta(
    stats_before: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
) -> float:
    """Max absolute delta in BN running stats since snapshot."""
    mx = 0.0
    for name, m in model.named_modules():
        if name in stats_before:
            rm0, rv0 = stats_before[name]
            mx = max(mx, (m.running_mean - rm0).abs().max().item())
            mx = max(mx, (m.running_var - rv0).abs().max().item())
    return mx
