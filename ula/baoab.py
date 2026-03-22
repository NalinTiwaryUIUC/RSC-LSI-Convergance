"""
BAOAB integrator for underdamped Langevin: momentum v, friction gamma, step h.

Half kick -> half drift -> OU on v -> half drift -> half kick (two gradient evals per step).
"""
from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn

from models.params import flatten_params, unflatten_like
from .grad_flat import compute_grad_U_flat


def underdamped_baoab_step(
    model: nn.Module,
    v: torch.Tensor,
    train_data: Union[torch.utils.data.DataLoader, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    h: float,
    gamma: float,
    device: torch.device,
    noise_scale: float = 1.0,
    drift_scale: float = 1.0,
    beta: float = 1.0,
    return_U: bool = False,
    generator: torch.Generator | None = None,
    ce_reduction: str = "mean",
    clip_grad_norm: float | None = None,
    num_microbatches: int = 1,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """
    One BAOAB step. Updates model parameters and v in place.
    v must be 1D, same length as flatten_params(model), same device/dtype as parameters.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    dtype = v.dtype
    theta0 = flatten_params(model).clone()

    # --- First half kick ---
    U1, grads1, gpre1, gpost1 = compute_grad_U_flat(
        model,
        train_data,
        alpha,
        device,
        ce_reduction,
        beta,
        clip_grad_norm,
        num_microbatches,
        microbatch_size,
    )
    v.add_(-(0.5 * h) * drift_scale * grads1)

    # --- First half drift ---
    theta = flatten_params(model)
    theta.add_(v, alpha=0.5 * h)
    unflatten_like(theta, model)

    # --- OU (Ornstein-Uhlenbeck) on v ---
    exp_m = torch.exp(torch.tensor(-gamma * h, device=device, dtype=dtype))
    inner = torch.exp(torch.tensor(-2.0 * gamma * h, device=device, dtype=dtype))
    ou_scale = torch.sqrt(torch.clamp(1.0 - inner, min=0.0))
    noise_std = noise_scale * ou_scale
    xi = torch.randn(v.shape, device=device, dtype=dtype, generator=generator)
    ou_noise = noise_std * xi
    v.mul_(exp_m).add_(ou_noise)

    # --- Second half drift ---
    theta = flatten_params(model)
    theta.add_(v, alpha=0.5 * h)
    unflatten_like(theta, model)

    # --- Second half kick ---
    U2, grads2, gpre2, gpost2 = compute_grad_U_flat(
        model,
        train_data,
        alpha,
        device,
        ce_reduction,
        beta,
        clip_grad_norm,
        num_microbatches,
        microbatch_size,
    )
    v.add_(-(0.5 * h) * drift_scale * grads2)

    theta_final = flatten_params(model)
    out: dict[str, Any] = {}
    if return_U:
        # Log U from first gradient eval (common convention)
        U = U1
        out["U"] = float(U) if isinstance(U, float) else U
        out["U_end"] = float(U2) if isinstance(U2, float) else U2
        gn = grads2.norm().item()
        out["grad_norm"] = gn
        out["theta_norm"] = theta_final.norm().item()
        vn = v.norm().item()
        out["v_norm"] = vn
        out["kinetic_energy"] = 0.5 * (vn**2)
        tnorm = theta_final.norm().item()
        if tnorm > 0 and vn > 0:
            cos_tv = (theta_final @ v).item() / (tnorm * vn)
        else:
            cos_tv = float("nan")
        out["theta_v_cosine"] = cos_tv
        out["noise_step_norm"] = ou_noise.norm().item()
        out["delta_theta_norm"] = (theta_final - theta0).norm().item()
    if clip_grad_norm is not None and gpre1 is not None and gpost1 is not None:
        out["grad_norm_pre_clip_first"] = gpre1
        out["grad_norm_post_clip_first"] = gpost1
    if clip_grad_norm is not None and gpre2 is not None and gpost2 is not None:
        out["grad_norm_pre_clip"] = gpre2
        out["grad_norm_post_clip"] = gpost2
    return out
