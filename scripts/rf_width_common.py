#!/usr/bin/env python3
"""
Shared utilities for random-feature width-convergence experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class SyntheticBinaryData:
    x: np.ndarray
    y: np.ndarray
    w_teacher: np.ndarray
    teacher_scale: float


def make_synthetic_binary_data(
    n: int,
    p: int,
    teacher_scale: float,
    seed: int,
) -> SyntheticBinaryData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    w_teacher = rng.normal(scale=1.0 / np.sqrt(p), size=p)
    probs = sigmoid(teacher_scale * (x @ w_teacher))
    y = rng.binomial(n=1, p=probs).astype(np.float64)
    return SyntheticBinaryData(x=x.astype(np.float64), y=y, w_teacher=w_teacher, teacher_scale=teacher_scale)


@dataclass
class RandomFeatureBank:
    w_max: np.ndarray
    b_max: np.ndarray

    @property
    def m_max(self) -> int:
        return int(self.w_max.shape[0])

    @property
    def p(self) -> int:
        return int(self.w_max.shape[1])


def make_random_feature_bank(
    p: int,
    m_max: int,
    seed: int,
) -> RandomFeatureBank:
    rng = np.random.default_rng(seed)
    w_max = rng.normal(scale=1.0 / np.sqrt(p), size=(m_max, p))
    b_max = rng.normal(size=(m_max,))
    return RandomFeatureBank(w_max=w_max.astype(np.float64), b_max=b_max.astype(np.float64))


def design_matrix_from_bank(
    x: np.ndarray,
    bank: RandomFeatureBank,
    width: int,
) -> np.ndarray:
    if width <= 0 or width > bank.m_max:
        raise ValueError(f"width must be in [1, {bank.m_max}]")
    z = x @ bank.w_max[:width, :].T + bank.b_max[:width]
    return (relu(z) / np.sqrt(width)).astype(np.float64)


def logistic_posterior_value_grad_hess(
    theta: np.ndarray,
    phi: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits = phi @ theta
    p = sigmoid(logits)
    # Stable logistic NLL: log(1+exp(z)) - y z.
    nll = np.logaddexp(0.0, logits) - y * logits
    u = float(nll.sum() + 0.5 * alpha * np.dot(theta, theta))
    grad = phi.T @ (p - y) + alpha * theta
    d = p * (1.0 - p)
    # H = Phi^T D Phi + alpha I.
    hess = (phi.T * d) @ phi
    hess.flat[:: hess.shape[0] + 1] += alpha
    return u, grad, hess


def logistic_grad(theta: np.ndarray, phi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    logits = phi @ theta
    p = sigmoid(logits)
    return phi.T @ (p - y) + alpha * theta


def logistic_hessian(theta: np.ndarray, phi: np.ndarray, alpha: float) -> np.ndarray:
    logits = phi @ theta
    p = sigmoid(logits)
    d = p * (1.0 - p)
    hess = (phi.T * d) @ phi
    hess.flat[:: hess.shape[0] + 1] += alpha
    return hess


@dataclass
class MapResult:
    theta_map: np.ndarray
    n_iter: int
    converged: bool
    grad_norm: float
    objective: float


def solve_map_newton(
    phi: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> MapResult:
    m = phi.shape[1]
    theta = np.zeros(m, dtype=np.float64)
    converged = False
    final_u = float("nan")
    final_grad_norm = float("nan")
    n_iter = 0
    for it in range(1, max_iter + 1):
        u, grad, hess = logistic_posterior_value_grad_hess(theta, phi, y, alpha)
        grad_norm = float(np.linalg.norm(grad))
        final_u = u
        final_grad_norm = grad_norm
        n_iter = it
        if grad_norm <= tol:
            converged = True
            break
        # Newton direction with small damping for numeric robustness.
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(hess + 1e-8 * np.eye(m), grad)
        # Backtracking line search.
        t = 1.0
        c = 1e-4
        while t >= 1e-8:
            theta_next = theta - t * step
            u_next, _, _ = logistic_posterior_value_grad_hess(theta_next, phi, y, alpha)
            if u_next <= u - c * t * float(np.dot(grad, step)):
                theta = theta_next
                break
            t *= 0.5
        else:
            theta = theta - 1e-3 * grad
    return MapResult(
        theta_map=theta,
        n_iter=n_iter,
        converged=converged,
        grad_norm=final_grad_norm,
        objective=final_u,
    )


def smoothness_bound(phi: np.ndarray, alpha: float) -> float:
    gram = phi.T @ phi
    lmax = float(np.linalg.eigvalsh(gram)[-1])
    return float(alpha + 0.25 * lmax)


@dataclass
class HessianSpectrum:
    lambda_min: float
    lambda_med: float
    lambda_max: float
    condition: float


def hessian_spectrum(h: np.ndarray) -> HessianSpectrum:
    evals = np.linalg.eigvalsh(h)
    lam_min = float(evals[0])
    lam_med = float(evals[len(evals) // 2])
    lam_max = float(evals[-1])
    condition = float(lam_max / lam_min) if lam_min > 0.0 else float("inf")
    return HessianSpectrum(lam_min, lam_med, lam_max, condition)


def apply_hinv_sqrt(h: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h)
    safe = np.maximum(evals, eps)
    return evecs @ ((1.0 / np.sqrt(safe)) * (evecs.T @ z))


def stable_row(values: dict[str, object], fieldnames: Iterable[str]) -> dict[str, object]:
    return {k: values.get(k, "") for k in fieldnames}

