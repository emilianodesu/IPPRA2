"""Data loading utilities for the GTSRB dataset.

This module centralizes all data preparation concerns for the project:

- Building standard training and evaluation transforms (with optional normalization)
- Computing dataset statistics (mean and standard deviation)
- Creating DataLoaders for common workflows (single split, K-fold, official test)

Conventions
----------
- Images are resized to 32x32 RGB to match the model's expected input.
- Normalization is applied when dataset statistics are provided; otherwise it's skipped
    (e.g., when first computing the statistics).
- Augmentations (rotation + color jitter) are used only for the training set.

Notes
-----
The GTSRB dataset structure and downloading are handled by ``torchvision.datasets.GTSRB``.
This module wraps those APIs in a consistent, reusable way for the rest of the codebase.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def make_transforms(mean=None, std=None, img_size=(32, 32)):
    """Create torchvision transforms for training and evaluation.

    When ``mean`` and ``std`` are provided, a Normalize transform is appended to
    both the training and test/eval pipelines.

    Args:
        mean (torch.Tensor | tuple | list | None): Per-channel mean (RGB) used for normalization.
        std (torch.Tensor | tuple | list | None): Per-channel std (RGB) used for normalization.
        img_size (tuple[int, int]): Target spatial size HxW for resizing (default: (32, 32)).

    Returns:
        tuple[transforms.Compose, transforms.Compose]: A pair ``(train_transform, test_transform)``.
    """
    apply_norm = (mean is not None and std is not None)

    train_tfms = [
        transforms.Resize(img_size),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ]
    test_tfms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ]

    if apply_norm:
        train_tfms.append(transforms.Normalize(mean=mean, std=std))
        test_tfms.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)


def get_mean_std(dataset):
    """Compute per-channel mean and std for an image dataset.

    This function loads the entire dataset in a single batch to compute statistics.
    It is simple and accurate but may be memory-heavy for very large datasets.

    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding images shaped as CxHxW tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(mean, std)`` with shape ``[C]`` (C=3 for RGB).
    """
    # Note: We compute stats on the training split only.
    loader = DataLoader(dataset, batch_size=len(
        dataset), num_workers=2, shuffle=False)
    data = next(iter(loader))
    # For single-channel, use data[0].mean() / .std(); for RGB, use per-channel stats.
    mean = data[0].mean(dim=[0, 2, 3])
    std = data[0].std(dim=[0, 2, 3])
    return mean, std


def get_official_test_loader(batch_size=128, mean=None, std=None):
    """Create a DataLoader for the official GTSRB test split.

    Args:
        batch_size (int): Batch size for iteration.
        mean (torch.Tensor | tuple | list | None): Optional RGB mean for normalization.
        std (torch.Tensor | tuple | list | None): Optional RGB std for normalization.

    Returns:
        torch.utils.data.DataLoader: Non-shuffled loader over the official test set.
    """
    _, test_transform = make_transforms(mean=mean, std=std)

    official_test_dataset = datasets.GTSRB(
        root='./data',
        split='test',
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        official_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def get_gtsrb_datasets(mean=None, std=None):
    """Get full training datasets with and without augmentations.

    The "augmented" dataset applies basic data augmentation and is intended for training.
    The "no_aug" dataset applies only resizing, tensor conversion, and optional normalization,
    and is intended for validation/test splits.

    Args:
        mean (torch.Tensor | tuple | list | None): Optional RGB mean for normalization.
        std (torch.Tensor | tuple | list | None): Optional RGB std for normalization.

    Returns:
        tuple[datasets.GTSRB, datasets.GTSRB]: ``(full_dataset_aug, full_dataset_no_aug)``
    """
    train_transform, test_transform = make_transforms(mean=mean, std=std)

    # Dataset with augmentations for training
    full_dataset_aug = datasets.GTSRB(
        root='./data', split='train', download=True, transform=train_transform)

    # Dataset without augmentations for validation/testing
    full_dataset_no_aug = datasets.GTSRB(
        root='./data', split='train', download=True, transform=test_transform)

    return full_dataset_aug, full_dataset_no_aug


def create_fold_dataloaders(
    train_indices,
    val_indices,
    full_dataset_aug,
    full_dataset_no_aug,
    batch_size=128,
    shuffle_train=True,
    num_workers=2,
    pin_memory=False,
):
    """Creates train/val DataLoaders for a specific fold given index splits.

    Args:
        train_indices (Iterable[int]): Indices of samples used for training.
        val_indices (Iterable[int]): Indices of samples used for validation.
        full_dataset_aug (torch.utils.data.Dataset): Augmented dataset for training.
        full_dataset_no_aug (torch.utils.data.Dataset): Non-augmented dataset for evaluation.
        batch_size (int): Batch size for both loaders.
        shuffle_train (bool): Whether to shuffle the training loader.

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, val_loader)`` for the fold.
    """
    train_subset = Subset(full_dataset_aug, train_indices)
    val_subset = Subset(full_dataset_no_aug, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_dataloaders(batch_size=64, random_state=42, num_workers=2, pin_memory=False):
    """Create standard train/val/test loaders for a 70/15/15 split.

    Workflow (no data leakage):
    1) Build datasets without normalization.
    2) Perform a stratified 70/15/15 split of the original training data.
    3) Compute mean/std on the TRAIN SPLIT ONLY.
    4) Rebuild datasets with normalization using the train-split statistics.
    5) Return DataLoaders for train/val/test (split) and the official test set,
       along with the computed normalization statistics.

    Args:
        batch_size (int): Number of samples per batch.
        random_state (int): Seed for the stratified split reproducibility.

    Returns:
        tuple:
            (train_loader, val_loader, test_loader_split, test_loader_official, mean, std)
    """
    # 1) Load datasets WITHOUT normalization so we can split and compute stats safely
    _, train_dataset_no_aug_no_norm = get_gtsrb_datasets()

    # 2) Perform a stratified 70/15/15 split using labels from the unnormalized dataset
    targets = [sample[1] for sample in train_dataset_no_aug_no_norm]
    train_indices, temp_indices = train_test_split(
        range(len(targets)), test_size=0.30, stratify=targets, random_state=random_state
    )
    temp_targets = [targets[i] for i in temp_indices]
    val_indices, test_indices_split = train_test_split(
        temp_indices, test_size=0.50, stratify=temp_targets, random_state=random_state
    )

    # 3) Compute mean/std on TRAIN SPLIT ONLY to avoid leakage
    train_no_norm_subset = Subset(train_dataset_no_aug_no_norm, train_indices)
    mean, std = get_mean_std(train_no_norm_subset)
    print(f"Calculated Mean (train only): {mean}")
    print(f"Calculated Std (train only): {std}")

    # 4) Re-create datasets WITH normalization applied (based on train stats)
    train_dataset_aug, train_dataset_no_aug = get_gtsrb_datasets(mean, std)

    # 5) Create subsets corresponding to the precomputed indices
    train_subset = Subset(train_dataset_aug, train_indices)
    val_subset = Subset(train_dataset_no_aug, val_indices)
    test_subset_split = Subset(train_dataset_no_aug, test_indices_split)

    # 6) Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader_split = DataLoader(
        test_subset_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Official test loader uses the same normalization stats
    test_loader_official = get_official_test_loader(
        batch_size=batch_size, mean=mean, std=std
    )

    return train_loader, val_loader, test_loader_split, test_loader_official, mean, std
