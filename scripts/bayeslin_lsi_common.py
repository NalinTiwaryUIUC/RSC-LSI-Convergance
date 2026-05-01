#!/usr/bin/env python3
"""
Pure math and helpers for Bayesian linear regression LSI width experiments.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np


class StdlibRNG:
    """
    Numpy-Generator-like RNG using only the stdlib (for broken ``numpy.random`` installs).
    """

    __slots__ = ("_r",)

    def __init__(self, seed: int) -> None:
        self._r = random.Random(seed)

    def normal(
        self, loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, tuple[int, ...]]] = None
    ) -> Any:
        if size is None:
            return float(loc + scale * self._r.gauss(0.0, 1.0))
        n = int(np.prod(size))
        arr = np.fromiter(
            (loc + scale * self._r.gauss(0.0, 1.0) for _ in range(n)),
            dtype=np.float64,
            count=n,
        )
        return arr.reshape(size)


def make_rng(seed: int) -> Any:
    """
    Return an RNG with a ``.normal(...)`` API compatible with ``generate_linear_regression_data``.

    Order: ``numpy.random.default_rng`` → ``RandomState`` → stdlib :class:`StdlibRNG`
    (last resort when ``numpy.random`` is a broken stub, e.g. ``ImportError: unknown location``).
    """
    try:
        from numpy.random import default_rng

        return default_rng(seed)
    except (ImportError, AttributeError, TypeError, OSError):
        pass
    try:
        from numpy.random import RandomState

        return RandomState(seed)
    except (ImportError, AttributeError, TypeError, OSError):
        pass
    return StdlibRNG(seed)


def potential_U(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float, sigma: float) -> float:
    r = X @ theta - y
    return float(0.5 / (sigma**2) * (r @ r) + 0.5 * alpha * float(theta @ theta))


def build_posterior_precision(X: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    m = X.shape[1]
    gram = X.T @ X
    return (1.0 / (sigma**2)) * gram + alpha * np.eye(m, dtype=np.float64)


def build_rhs_b(X: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    return (1.0 / (sigma**2)) * (X.T @ y)


def theta_star_from_normal_eqs(H: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(H, b)


def grad_U(theta: np.ndarray, H: np.ndarray, b: np.ndarray) -> np.ndarray:
    return H @ theta - b


def grad_U_via_star(theta: np.ndarray, theta_star: np.ndarray, H: np.ndarray) -> np.ndarray:
    return H @ (theta - theta_star)


def generate_linear_regression_data(
    m: int,
    c: int,
    alpha: float,
    sigma: float,
    teacher_scale: float,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = c * m
    X = rng.normal(size=(n, m)).astype(np.float64)
    theta_true = rng.normal(scale=teacher_scale / np.sqrt(m), size=m).astype(np.float64)
    y = (X @ theta_true + sigma * rng.normal(size=n)).astype(np.float64)
    return X, y, theta_true


def spectrum_summary(H: np.ndarray) -> dict[str, float]:
    evals = np.linalg.eigvalsh(H)
    lam_min = float(evals[0])
    lam_med = float(evals[len(evals) // 2])
    lam_max = float(evals[-1])
    cond = float(lam_max / max(lam_min, 1e-300))
    c_lsi = float(1.0 / max(lam_min, 1e-300))
    return {
        "lambda_min": lam_min,
        "lambda_med": lam_med,
        "lambda_max": lam_max,
        "condition": cond,
        "C_LSI": c_lsi,
        "C_PI": c_lsi,
    }


def convergence_metrics_row(
    theta: np.ndarray,
    theta_star: np.ndarray,
    X: np.ndarray,
    H: np.ndarray,
    alpha: float,
    sigma: float,
    y: np.ndarray,
    delta0_euc: float,
    delta0_H: float,
    delta0_pred: float,
    u0_gap: float,
) -> dict[str, float]:
    d = theta - theta_star
    d_euc = float(np.linalg.norm(d))
    d_h = float(np.sqrt(max(d @ H @ d, 0.0)))
    r = X @ d
    n = X.shape[0]
    d_pred = float(np.linalg.norm(r) / np.sqrt(n))
    e_euc = d_euc / max(delta0_euc, 1e-300)
    e_h = d_h / max(delta0_H, 1e-300)
    e_pred = d_pred / max(delta0_pred, 1e-300)
    u_now = potential_U(theta, X, y, alpha, sigma)
    u_star = potential_U(theta_star, X, y, alpha, sigma)
    u_gap = float(u_now - u_star)
    e_u = u_gap / max(u0_gap, 1e-300)
    g = grad_U(theta, H, build_rhs_b(X, y, sigma))
    grad_norm = float(np.linalg.norm(g))
    theta_norm = float(np.linalg.norm(theta))
    max_abs = float(np.max(np.abs(theta)))
    nan_or_inf = int(
        (not np.isfinite(theta).all())
        or (not np.isfinite(g).all())
        or (not np.isfinite(d_euc))
        or (not np.isfinite(d_h))
    )
    return {
        "D_euc": d_euc,
        "e_euc": e_euc,
        "D_H": d_h,
        "e_H": e_h,
        "U_gap": u_gap,
        "e_U": e_u,
        "D_pred": d_pred,
        "e_pred": e_pred,
        "grad_norm": grad_norm,
        "theta_norm": theta_norm,
        "max_abs_theta": max_abs,
        "nan_or_inf": float(nan_or_inf),
    }


def global_step_size(width_to_lambda_max: dict[int, float], h_factor: float) -> float:
    lmx = max(width_to_lambda_max.values())
    if lmx <= 0.0:
        raise ValueError("non-positive lambda_max")
    return float(h_factor / lmx)


def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-300)))


def fit_rate(times: np.ndarray, log_e: np.ndarray, t0: float, t1: float) -> float:
    mask = (times >= t0) & (times <= t1) & np.isfinite(log_e)
    if mask.sum() < 2:
        return float("nan")
    x = times[mask]
    y = log_e[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0.0:
        return float("nan")
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(-slope)


def first_time_leq(times: np.ndarray, vals: np.ndarray, thr: float) -> float:
    idx = np.where(vals <= thr)[0]
    if idx.size == 0:
        return float("nan")
    return float(times[idx[0]])


@dataclass
class RateWindow:
    name: str
    t0: float
    t1: float


DEFAULT_RATE_WINDOWS: tuple[RateWindow, ...] = (
    RateWindow("early", 0.0, 2.0),
    RateWindow("mid", 2.0, 10.0),
    RateWindow("full", 0.0, 10.0),
    RateWindow("long", 5.0, 20.0),
)

THRESHOLDS: tuple[float, ...] = (0.5, 0.1, 0.01)


def gd_step(theta: np.ndarray, H: np.ndarray, b: np.ndarray, h: float) -> np.ndarray:
    return theta - h * grad_U(theta, H, b)


def iterate_gd_closed_form(
    theta0: np.ndarray,
    theta_star: np.ndarray,
    H: np.ndarray,
    h: float,
    n_steps: int,
) -> np.ndarray:
    """theta_k - theta_star = (I - hH)^k (theta0 - theta_star)."""
    m = H.shape[0]
    delta = theta0 - theta_star
    I = np.eye(m, dtype=np.float64)
    M = I - h * H
    Mk = np.linalg.matrix_power(M, n_steps)
    return theta_star + Mk @ delta
