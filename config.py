"""
Run config schema for LSI/ULA experiments.
Config-driven so runs are reproducible and components replaceable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import torch


def get_device(prefer_cuda: bool = True, device_id: int | None = None) -> "torch.device":
    """Return the best available device (cuda if available and prefer_cuda, else cpu)."""
    import torch
    if prefer_cuda and torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class RunConfig:
    """Schema for a single ULA chain run."""

    # Data
    n_train: int = 1024
    probe_size: int = 512
    subset_indices_file: str = "train_subset_indices.json"
    probe_indices_file: str = "probe_indices.json"
    data_dir: str = "experiments/data"

    # Model
    width_multiplier: float = 1.0
    num_classes: int = 10

    # ULA
    h: float = 1e-4  # larger steps for more movement
    alpha: float = 0.01  # reduced from 0.05 to lessen ∇NLL/αθ cancellation; improves SNR
    temperature: float = 1.0
    noise_scale: float = 1.0  # standard ULA uses 1; <1 = less noise, >1 = more diffusion

    # Chain
    log_every: int = 1000  # write iter_metrics every N steps (1 = every step)
    progress_print_every: int = 10_000  # print progress to stdout every N steps (0 = disable)
    pretrain_steps: int = 2000  # number of full-batch SGD steps before ULA (more = start nearer a mode)
    pretrain_lr: float = 0.02  # learning rate for pretraining
    T: int = 200_000
    B: int = 50_000
    S: int = 200
    K: int = 4
    sigma_init_scale: float = 1e-4

    # Probes / LSI
    grad_norm_stride: int = 5  # compute grad norms every G saved samples

    # BN mode for ULA sampling: "eval" (frozen running stats) | "batchstat_frozen" (batch stats, frozen buffers)
    bn_mode: str = "batchstat_frozen"

    # Seeds (for reproducibility)
    dataset_seed: int = 42
    chain_seed: int = 0
    probe_projection_seed: int = 12345

    # Run metadata
    chain_id: int = 0
    run_dir: str | None = None
    param_count: int | None = None  # set at run time for diagnostics (d = num params)
    ou_radius_pred: float | None = None  # sqrt(d/alpha), OU stationary std; for "pure prior diffusion" test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        # Ignore unknown keys so we can load configs with extra fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


def ensure_directories(config: RunConfig | None = None, base: str | Path = "experiments") -> None:
    """Create experiment directory layout."""
    base = Path(base)
    dirs = [
        base / "runs",
        base / "analysis",
        base / "summaries",
        base / "figures",
        base / "data",
        base / "checkpoints",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
