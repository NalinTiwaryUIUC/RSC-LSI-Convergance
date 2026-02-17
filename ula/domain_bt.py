"""
Domain B_t monitoring: shifting ball (centered at previous iterate).
B_t = ball of radius rho2 around theta_{t-1}. Check step length ||theta_t - theta_{t-1}||_2.
If iterate-to-iterate movement is small, guarantees hold along the optimization path.
"""
from __future__ import annotations

import torch


def compute_bt_metrics(
    theta_new_flat: torch.Tensor,
    theta_prev_flat: torch.Tensor,
    rho2: float,
) -> tuple[float, float, bool]:
    """
    Step-length check (shifting ball): B_t centered at theta_{t-1}.
    Returns:
        theta_dist: ||theta_new - theta_prev||_2 (step length)
        bt_margin: theta_dist - rho2
        inside_bt: (bt_margin <= 0) — step stayed within ball
    """
    diff = theta_new_flat - theta_prev_flat
    theta_dist = diff.norm(2).item()
    bt_margin = theta_dist - rho2
    inside_bt = bt_margin <= 0
    return theta_dist, bt_margin, inside_bt


def get_rho2(theta0_flat: torch.Tensor, factor: float = 0.05) -> float:
    """rho2 = factor * ||theta0||_2 (max allowed step length for shifting B_t)."""
    r0 = theta0_flat.norm(2).item()
    return factor * r0
