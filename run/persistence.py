"""
Persistence: run_config.yaml, iter_metrics.jsonl, samples_metrics.npz.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from config import RunConfig


def write_run_config(config: RunConfig, run_dir: str | Path) -> None:
    """Write run_config.yaml to run_dir."""
    path = Path(run_dir) / "run_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(path)


def load_run_config(run_dir: str | Path) -> RunConfig:
    """Load run_config.yaml from run_dir."""
    path = Path(run_dir) / "run_config.yaml"
    return RunConfig.from_yaml(path)


def write_iter_metrics(
    step: int,
    grad_evals: int,
    run_dir: str | Path,
    U_train: float | None = None,
    grad_norm: float | None = None,
    theta_norm: float | None = None,
    f_nll: float | None = None,
) -> None:
    """Append one line to iter_metrics.jsonl."""
    path = Path(run_dir) / "iter_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, "grad_evals": grad_evals}
    if U_train is not None:
        record["U_train"] = U_train
    if grad_norm is not None:
        record["grad_norm"] = grad_norm
    if theta_norm is not None:
        record["theta_norm"] = theta_norm
    if f_nll is not None:
        record["f_nll"] = f_nll
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_samples_metrics(
    run_dir: str | Path,
    steps: List[int],
    grad_evals_list: List[int],
    f_values: Dict[str, List[float]],
    grad_norm_sq: Dict[str, List[float]] | None = None,
) -> None:
    """Write samples_metrics.npz (arrays for saved samples)."""
    path = Path(run_dir) / "samples_metrics.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "step": np.array(steps, dtype=np.int64),
        "grad_evals": np.array(grad_evals_list, dtype=np.int64),
    }
    for name, vals in f_values.items():
        out[name] = np.array(vals, dtype=np.float64)
    if grad_norm_sq:
        for name, vals in grad_norm_sq.items():
            if vals:
                out[f"grad_norm_sq__{name}"] = np.array(vals, dtype=np.float64)
    np.savez_compressed(path, **out)
