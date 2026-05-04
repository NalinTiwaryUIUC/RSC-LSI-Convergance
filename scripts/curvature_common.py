"""
Curvature primitives for the negative-curvature / NME experiment.

This module provides:
    * `flat_grad`               -- mean-reduction loss gradient flattened to R^p.
    * `hvp_full`                -- Hessian-vector product for an arbitrary loss
                                   (default: mean cross-entropy).
    * `ggn_ce_vp`               -- Generalized Gauss-Newton vector product for
                                   cross-entropy loss with mean reduction
                                   (softmax-covariance form).
    * `nme_vp`                  -- model-Hessian / NME vector product
                                   (Hv - Gv); only source of negative curvature.
    * `make_torch_linop`        -- numpy <-> torch bridge that wraps a torch
                                   matvec into `scipy.sparse.linalg.LinearOperator`.
    * `top_k_smallest_eigs`     -- `eigsh(which="SA", ...)` returning the smallest
                                   algebraic eigenvalues of a matvec callable.
    * `slq_trace_neg`           -- stochastic Lanczos quadrature estimate of
                                   `tr((-N)_+) = sum_j max(0, -lambda_j(N))`.
    * `cumulative_eta_metrics`  -- gamma_emp, T_neg, k50/k80/k90 from a
                                   sequence of (negative) eigenvalues.
    * `r_eff_neg`, `e_iso`,     -- summary scalars from the writeup.
      `e_aniso`

All numerical helpers are pure / GPU-agnostic; the torch ones run on whatever
device the model lives on.

References:
    minimal_negative_curvature_experiment.md (sections "Operators to compute",
    "Lanczos diagnostics", "SLQ estimate", "Main metrics").
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse.linalg as sla
from scipy.sparse.linalg import ArpackNoConvergence


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def ce_mean_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy. Default loss for the curvature experiment."""
    return F.cross_entropy(logits, y, reduction="mean")


# ---------------------------------------------------------------------------
# Gradient and Hessian-vector product on flat parameters
# ---------------------------------------------------------------------------

def _params_list(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def flat_grad(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ce_mean_loss,
    create_graph: bool = False,
) -> torch.Tensor:
    """Return ``\u2207_\u03b8 loss_fn(model(x), y)`` flattened to a single 1D tensor."""
    params = _params_list(model)
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return _flatten(grads)


def hvp_full(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v_flat: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ce_mean_loss,
) -> torch.Tensor:
    """Hessian-vector product H_L v for the (arbitrary) scalar loss ``loss_fn``.

    Implementation: Pearlmutter-style double backward.
    First grad with ``create_graph=True`` builds a graph for the gradient
    function, the second grad takes vector-Jacobian product against ``v_flat``.
    Symmetry of H makes ``vec(\u2207\u00b2L v) = vec(v\u1d40 \u2207\u00b2L)``.
    """
    params = _params_list(model)
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = _flatten(grads)
    hv = torch.autograd.grad(g_flat, params, grad_outputs=v_flat, retain_graph=False)
    return _flatten(hv)


# ---------------------------------------------------------------------------
# Generalized Gauss-Newton VP for cross-entropy (softmax covariance form)
# ---------------------------------------------------------------------------

def ggn_ce_vp(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v_flat: torch.Tensor,
) -> torch.Tensor:
    """Generalized Gauss-Newton VP for ``loss = mean cross-entropy``.

    For per-sample logits ``z_i = f(\u03b8; x_i)`` and labels ``y_i``, with
    ``s_i = softmax(z_i)`` and ``S_i = diag(s_i) - s_i s_i\u1d40``,

        G v = (1/B) \u03a3_i J_i\u1d40 S_i J_i v.

    Computed via:
        1.  ``J\u1d40 u`` as a function of an aux tensor ``u`` (zero-init,
            requires_grad), via VJP through a single backward;
        2.  ``J v`` via grad of ``(J\u1d40 u) \u00b7 v`` w.r.t. ``u``
            (Pearlmutter forward-mode emulation);
        3.  Apply S row-wise: ``(S J v)_i = (J v)_i - s_i (s_i \u00b7 (J v)_i)``;
        5.  Outer VJP with the weighted residual gives ``G v``.

    The (1/B) factor is folded into the cotangent (matches CE mean reduction).
    """
    params = _params_list(model)
    logits = model(x)
    batch = logits.shape[0]
    if batch == 0:
        raise ValueError("ggn_ce_vp requires non-empty batch")
    u = torch.zeros_like(logits, requires_grad=True)
    g_uT = torch.autograd.grad(
        (logits * u).sum(),
        params,
        create_graph=True,
    )
    g_uT_flat = _flatten(g_uT)
    inner = (g_uT_flat * v_flat).sum()
    Jv = torch.autograd.grad(inner, u, retain_graph=True)[0]
    with torch.no_grad():
        s = torch.softmax(logits.detach(), dim=-1)
    # Softmax covariance applied per row:
    #   (S_i Jv_i)_c = s_{i,c} (Jv_i)_c - s_{i,c} (s_i \cdot Jv_i)
    # i.e. S Jv = s \odot Jv - s (s \cdot Jv).
    sJv = s * Jv
    s_dot_Jv = sJv.sum(dim=-1, keepdim=True)
    weighted = (sJv - s * s_dot_Jv) / float(batch)
    g_out = torch.autograd.grad(
        (logits * weighted).sum(),
        params,
    )
    return _flatten(g_out)


# ---------------------------------------------------------------------------
# NME / model-Hessian VP
# ---------------------------------------------------------------------------

def nme_vp(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v_flat: torch.Tensor,
) -> torch.Tensor:
    """Model-Hessian VP: N v = H_L v - G v (CE, mean reduction)."""
    return hvp_full(model, x, y, v_flat) - ggn_ce_vp(model, x, y, v_flat)


# ---------------------------------------------------------------------------
# Numpy <-> torch bridge for scipy LinearOperator
# ---------------------------------------------------------------------------

def make_torch_linop(
    matvec_torch: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    np_dtype: np.dtype = np.float64,
) -> sla.LinearOperator:
    """Wrap a torch matvec into a scipy `LinearOperator`.

    The numpy interface uses ``np_dtype`` (default float64). The torch matvec
    receives / returns ``torch_dtype`` tensors on ``device``. Symmetric.
    """

    def _matvec(v_np):
        v = torch.as_tensor(np.asarray(v_np).reshape(-1), dtype=torch_dtype, device=device)
        out = matvec_torch(v)
        return np.asarray(out.detach().cpu().numpy(), dtype=np_dtype).reshape(-1)

    return sla.LinearOperator((p, p), matvec=_matvec, rmatvec=_matvec, dtype=np_dtype)


def top_k_smallest_eigs(
    linop: sla.LinearOperator,
    *,
    k: int = 20,
    ncv: int | None = None,
    maxiter: int = 80,
    tol: float = 0.0,
) -> np.ndarray:
    """Return the ``k`` smallest algebraic eigenvalues of a symmetric LinearOperator.

    Uses ``scipy.sparse.linalg.eigsh(which="SA")``. Returns a 1D ``np.float64``
    array sorted ascending of length ``k``.
    """
    p = linop.shape[0]
    if k <= 0:
        return np.array([], dtype=np.float64)
    if k >= p:
        raise ValueError(f"k={k} must be < p={p} for eigsh")
    if ncv is None or ncv <= 0:
        ncv = min(max(2 * k + 1, 4 * k + 1), p - 1)

    def _eigsh(mi: int, nv: int) -> np.ndarray:
        return sla.eigsh(
            linop,
            k=k,
            which="SA",
            ncv=min(max(nv, k + 1), p - 1),
            maxiter=mi,
            tol=tol,
            return_eigenvectors=False,
        )

    try:
        eigs = _eigsh(maxiter, ncv)
    except ArpackNoConvergence as err:
        # Partial spectrum (if any) is not reliable for "smallest k"; retry harder.
        ev = getattr(err, "eigenvalues", None)
        if ev is not None and len(ev) >= k:
            eigs = np.sort(np.asarray(ev, dtype=np.float64))[:k]
        else:
            ncv2 = min(p - 1, max(ncv * 2, 4 * k + 1, 2 * k + 1))
            maxiter2 = max(200, maxiter * 5, 10 * k)
            try:
                eigs = _eigsh(maxiter2, ncv2)
            except ArpackNoConvergence as err2:
                ev2 = getattr(err2, "eigenvalues", None)
                if ev2 is not None and len(ev2) >= k:
                    eigs = np.sort(np.asarray(ev2, dtype=np.float64))[:k]
                else:
                    raise err2 from err
    eigs = np.asarray(eigs, dtype=np.float64)
    return np.sort(eigs)


# ---------------------------------------------------------------------------
# Stochastic Lanczos Quadrature for tr((-N)_+)
# ---------------------------------------------------------------------------

def _lanczos_tridiag(
    matvec: Callable[[np.ndarray], np.ndarray],
    v0: np.ndarray,
    *,
    steps: int,
    reorth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-step Lanczos starting from a unit-norm vector ``v0``.

    Returns ``(alpha, beta)`` with ``alpha`` of length ``m`` (actual step count
    after early termination) and ``beta`` of length ``m-1`` (subdiagonal).
    """
    p = v0.shape[0]
    alpha: list[float] = []
    beta: list[float] = []
    V: list[np.ndarray] = []
    v_prev = np.zeros(p, dtype=np.float64)
    v = np.asarray(v0, dtype=np.float64).copy()
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    v = v / nrm
    beta_prev = 0.0
    for step in range(steps):
        V.append(v)
        w = matvec(v)
        a = float(v.dot(w))
        alpha.append(a)
        w = w - a * v - beta_prev * v_prev
        if reorth:
            for vk in V:
                w = w - float(vk.dot(w)) * vk
        b = float(np.linalg.norm(w))
        if b < 1e-12:
            break
        beta.append(b)
        v_prev = v
        v = w / b
        beta_prev = b
    return np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)


def slq_trace_neg(
    matvec: Callable[[np.ndarray], np.ndarray],
    p: int,
    *,
    num_probes: int = 8,
    lanczos_steps: int = 30,
    rng: np.random.Generator | None = None,
    reorth: bool = True,
) -> float:
    """Stochastic Lanczos Quadrature for ``tr(f(N))`` with ``f(\u03bb)=max(0,-\u03bb)``.

    Probe distribution: Rademacher (\u00b11). For each probe ``z`` we run
    ``lanczos_steps`` Lanczos iterations on ``N`` starting from ``z/||z||``,
    eigendecompose the resulting tridiagonal ``T_k`` to ``\u03b8_j``, and use the
    Gauss-Lanczos quadrature weights ``w_j = (V_{0,j})\u00b2`` to estimate

        z\u1d40 f(N) z \u2248 ||z||\u00b2 \u03a3_j w_j f(\u03b8_j).

    Average over probes.
    """
    if num_probes <= 0 or lanczos_steps <= 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()

    total = 0.0
    for _ in range(num_probes):
        z = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=p)
        z_norm_sq = float(z.dot(z))
        alpha, beta = _lanczos_tridiag(matvec, z, steps=lanczos_steps, reorth=reorth)
        m = alpha.shape[0]
        if m == 0:
            continue
        T = np.diag(alpha)
        if m > 1:
            off = beta[: m - 1]
            T = T + np.diag(off, 1) + np.diag(off, -1)
        theta, V = np.linalg.eigh(T)
        weights = V[0, :] ** 2
        f_theta = np.maximum(0.0, -theta)
        total += z_norm_sq * float(np.sum(weights * f_theta))
    return total / num_probes


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def cumulative_eta_metrics(eigs: np.ndarray | Iterable[float]) -> dict:
    """Convert the k smallest algebraic eigenvalues of ``N`` to summary stats.

    Returns
    -------
    dict with keys:
        gamma_emp     : max(0, -lambda_min(N))
        T_neg_top     : sum of (-lambda_j)_+ over the supplied eigenvalues
        eta_sorted    : (-lambda_j)_+ sorted descending
        cum_eta       : cumulative sum of eta_sorted
        cum_frac      : cum_eta / T_neg_top (NaN if T_neg_top == 0)
        k50, k80, k90 : smallest 1-indexed k with cum_frac >= {0.5, 0.8, 0.9}
                        (0 if T_neg_top == 0)
    """
    arr = np.asarray(list(eigs) if not isinstance(eigs, np.ndarray) else eigs, dtype=np.float64)
    eta = np.maximum(0.0, -arr)
    eta_sorted = np.sort(eta)[::-1]
    T_neg = float(eta_sorted.sum())
    gamma_emp = float(eta_sorted[0]) if eta_sorted.size > 0 else 0.0
    cum_eta = np.cumsum(eta_sorted) if eta_sorted.size > 0 else np.zeros(0, dtype=np.float64)
    if T_neg <= 0.0 or eta_sorted.size == 0:
        cum_frac = np.zeros(0, dtype=np.float64)
        k50 = k80 = k90 = 0
    else:
        cum_frac = cum_eta / T_neg
        k50 = int(np.searchsorted(cum_frac, 0.50) + 1)
        k80 = int(np.searchsorted(cum_frac, 0.80) + 1)
        k90 = int(np.searchsorted(cum_frac, 0.90) + 1)
        k50 = min(k50, cum_frac.size)
        k80 = min(k80, cum_frac.size)
        k90 = min(k90, cum_frac.size)
    return {
        "gamma_emp": gamma_emp,
        "T_neg_top": T_neg,
        "eta_sorted": eta_sorted,
        "cum_eta": cum_eta,
        "cum_frac": cum_frac,
        "k50": k50,
        "k80": k80,
        "k90": k90,
    }


def r_eff_neg(T_neg: float, gamma_emp: float) -> float:
    """``T_neg / gamma_emp`` (0 when ``gamma_emp <= 0``)."""
    if gamma_emp <= 0.0:
        return 0.0
    return float(T_neg) / float(gamma_emp)


def e_iso(gamma_emp: float, p: int) -> float:
    """Pessimistic isotropic exponent proxy: ``gamma_emp * p``."""
    return float(gamma_emp) * float(p)


def e_aniso(T_neg: float) -> float:
    """Anisotropic exponent proxy: ``T_neg`` itself."""
    return float(T_neg)
