"""
Domain B_t monitoring: ball around theta0.
"""
from __future__ import annotations

import torch


def compute_bt_metrics(
    theta_flat: torch.Tensor,
    theta0_flat: torch.Tensor,
    rho2: float,
) -> tuple[float, float, bool]:
    """
    Returns:
        theta_dist: ||theta - theta0||_2
        bt_margin: theta_dist - rho2
        inside_bt: (bt_margin <= 0)
    """
    diff = theta_flat - theta0_flat
    theta_dist = diff.norm(2).item()
    bt_margin = theta_dist - rho2
    inside_bt = bt_margin <= 0
    return theta_dist, bt_margin, inside_bt


def get_rho2(theta0_flat: torch.Tensor, factor: float = 0.05) -> float:
    """rho2 = factor * ||theta0||_2."""
    r0 = theta0_flat.norm(2).item()
    return factor * r0
