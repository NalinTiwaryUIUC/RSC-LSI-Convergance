"""
CIFAR-10 loaders with exact transforms from the experiment plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


# Exact transforms from EXPERIMENT_PLAN_LSI_PYTORCH.md §2.3
TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])


def get_train_loader(
    n: int,
    batch_size: Optional[int] = None,
    dataset_seed: int = 42,
    data_dir: str | Path = "experiments/data",
    root: str | Path = "./data",
    indices: Optional[List[int]] = None,
    pin_memory: bool = False,
    eval_transform: bool = False,
) -> DataLoader:
    """
    CIFAR-10 training loader on a subset of n examples.
    If indices is None, uses get_train_subset_indices(n, dataset_seed, data_dir).
    batch_size=None means full-batch (batch_size = n).
    Set pin_memory=True when using GPU for faster host-to-device transfer.
    eval_transform=True: use TEST_TRANSFORM (no augmentation) for deterministic
    data; use for pretrain+sampling so both see identical inputs.
    """
    from .indices import get_train_subset_indices

    if indices is None:
        indices = get_train_subset_indices(n, dataset_seed, data_dir)
    if batch_size is None:
        batch_size = n

    transform = TEST_TRANSFORM if eval_transform else TRAIN_TRANSFORM
    dataset = CIFAR10(root=str(root), train=True, download=True, transform=transform)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )


def get_probe_loader(
    probe_size: int = 512,
    dataset_seed: int = 43,
    data_dir: str | Path = "experiments/data",
    root: str | Path = "./data",
    indices: Optional[List[int]] = None,
    pin_memory: bool = False,
) -> DataLoader:
    """
    CIFAR-10 test loader restricted to fixed probe indices.
    batch_size = probe_size (full probe set in one batch by default).
    Set pin_memory=True when using GPU for faster host-to-device transfer.
    """
    from .indices import get_probe_indices

    if indices is None:
        indices = get_probe_indices(probe_size, dataset_seed, data_dir)

    dataset = CIFAR10(root=str(root), train=False, download=True, transform=TEST_TRANSFORM)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=len(indices), shuffle=False, num_workers=0, pin_memory=pin_memory
    )
