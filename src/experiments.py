"""Shared experiment utilities for cross-validation and hyperparameter tuning.

This module centralizes common routines used by run_cross_validation.py and
run_hyperparameter_tuning.py to keep scripts lean, reusable, and leakage-safe.

Provided helpers
----------------
- set_seeds(seed): Reproducible seeding across torch, numpy, and random.
- compute_mean_std_on_indices(dataset, indices): Stats on a subset (train-only).
- make_fold_loaders(base_dataset_no_norm, train_indices, val_indices, batch_size):
  Build normalized datasets per fold and return DataLoaders without leakage.
- build_model_components(lr, optimizer_name, device, scheduler_cfg=None):
  Construct model, criterion, optimizer, and optional scheduler.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset

from .model import GTSRB_CNN
from .data_loader import get_mean_std, get_gtsrb_datasets, create_fold_dataloaders


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for basic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mean_std_on_indices(dataset, indices: Iterable[int]):
    """Compute mean/std using only the samples referenced by ``indices``.

    Args:
        dataset: A dataset without normalization applied.
        indices (Iterable[int]): Indices belonging to the training split.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (mean, std) with shape [C].
    """
    subset = Subset(dataset, list(indices))
    return get_mean_std(subset)


def make_fold_loaders(
    base_dataset_no_norm,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = False,
):
    """Create train/val loaders for a fold with train-only normalization stats.

    Workflow:
    1) Compute mean/std on train_indices only.
    2) Rebuild datasets with normalization.
    3) Return train/val DataLoaders constructed on the same indices.

    Args:
        base_dataset_no_norm: Dataset without normalization (used only for stats).
        train_indices (Iterable[int]): Training indices for the fold.
        val_indices (Iterable[int]): Validation indices for the fold.
        batch_size (int): Dataloader batch size.

    Returns:
        tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
            (train_loader, val_loader, mean, std)
    """
    mean, std = compute_mean_std_on_indices(base_dataset_no_norm, train_indices)
    train_ds, eval_ds = get_gtsrb_datasets(mean, std)
    train_loader, val_loader = create_fold_dataloaders(
        train_indices,
        val_indices,
        train_ds,
        eval_ds,
        batch_size=batch_size,
        shuffle_train=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, mean, std


def build_model_components(
    lr: float,
    optimizer_name: str,
    device: torch.device,
    scheduler_cfg: Optional[dict] = None,
):
    """Construct model, criterion, optimizer, and optional LR scheduler.

    Args:
        lr (float): Learning rate for the optimizer.
        optimizer_name (str): Name of the optimizer class in torch.optim (e.g., 'Adam', 'SGD').
        device (torch.device): Target device.
        scheduler_cfg (dict | None): Optional scheduler config like
            {"step_size": 5, "gamma": 0.5}. If provided, uses StepLR; otherwise None.

    Returns:
        tuple: (model, criterion, optimizer, scheduler)
    """
    model = GTSRB_CNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    scheduler = None
    if scheduler_cfg and "step_size" in scheduler_cfg and "gamma" in scheduler_cfg:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg["step_size"], gamma=scheduler_cfg["gamma"]
        )

    return model, criterion, optimizer, scheduler
