"""
Fixed train subset and probe indices for reproducible runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


def _indices_dir(data_dir: str | Path = "experiments/data") -> Path:
    return Path(data_dir)


def get_train_subset_indices(
    n: int,
    dataset_seed: int = 42,
    data_dir: str | Path = "experiments/data",
    num_train: int = 50_000,
) -> List[int]:
    """
    Return first n indices from a fixed permutation of [0 .. num_train-1].
    Persists to train_subset_indices.json keyed by n so the subset is
    identical across widths/chains.
    """
    d = _indices_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "train_subset_indices.json"

    if path.exists():
        with open(path) as f:
            stored = json.load(f)
    else:
        stored = {}

    key = str(n)
    if key in stored:
        return stored[key]

    rng = np.random.default_rng(dataset_seed)
    perm = rng.permutation(num_train)
    indices = perm[:n].tolist()
    stored[key] = indices
    with open(path, "w") as f:
        json.dump(stored, f, indent=0)
    return indices


def get_probe_indices(
    probe_size: int = 512,
    dataset_seed: int = 43,
    data_dir: str | Path = "experiments/data",
    num_test: int = 10_000,
) -> List[int]:
    """
    Return fixed probe_size test indices (fixed permutation).
    Persists to probe_indices.json.
    """
    d = _indices_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "probe_indices.json"

    if path.exists():
        with open(path) as f:
            stored = json.load(f)
    else:
        stored = {}

    key = str(probe_size)
    if key in stored:
        return stored[key]

    rng = np.random.default_rng(dataset_seed)
    perm = rng.permutation(num_test)
    indices = perm[:probe_size].tolist()
    stored[key] = indices
    with open(path, "w") as f:
        json.dump(stored, f, indent=0)
    return indices
