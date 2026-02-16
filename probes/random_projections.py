"""
Fixed random projections for probes: v1, v2 (param space) and logit projection matrix.
Persisted so identical across widths/chains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def _data_dir(data_dir: str | Path = "experiments/data") -> Path:
    return Path(data_dir)


def get_or_create_param_projections(
    d_param: int,
    seed: int = 12345,
    data_dir: str | Path = "experiments/data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or create v1, v2 (each shape (d_param,)). Saved as v1.pt, v2.pt."""
    d = _data_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    v1_path = d / "v1.pt"
    v2_path = d / "v2.pt"
    if v1_path.exists() and v2_path.exists():
        v1 = torch.load(v1_path, weights_only=True)
        v2 = torch.load(v2_path, weights_only=True)
        if v1.shape[0] == d_param and v2.shape[0] == d_param:
            return v1, v2
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(d_param, generator=g)
    v2 = torch.randn(d_param, generator=g)
    torch.save(v1, v1_path)
    torch.save(v2, v2_path)
    return v1, v2


def get_or_create_logit_projection(
    logit_dim: int,
    n_components: int = 2,
    seed: int = 54321,
    data_dir: str | Path = "experiments/data",
) -> torch.Tensor:
    """Get or create logit projection matrix shape (n_components, logit_dim). Saved as logit_proj.pt."""
    d = _data_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "logit_proj.pt"
    if path.exists():
        proj = torch.load(path, weights_only=True)
        if proj.shape == (n_components, logit_dim):
            return proj
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(n_components, logit_dim, generator=g)
    torch.save(proj, path)
    return proj
