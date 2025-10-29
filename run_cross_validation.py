"""K-fold cross-validation entry point for GTSRB classification.

This script orchestrates a stratified K-fold evaluation using the shared
data-loading utilities, model definition, and Trainer abstraction. It reports
per-fold metrics and a summary (mean/std accuracy and average training time).
Results are saved under ``results_cv/cv_summary.json``.
"""

import time
import json
import os
import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Import our custom modules
from src.data_loader import get_gtsrb_datasets
from src.model import GTSRB_CNN
from src.trainer import Trainer
from src.experiments import set_seeds, make_fold_loaders, build_model_components


def build_default_components(device, lr):
    """Wrapper around shared builder with default Adam + StepLR config."""
    return build_model_components(
        lr=lr,
        optimizer_name="Adam",
        device=device,
        scheduler_cfg={"step_size": 5, "gamma": 0.5},
    )


def run_fold(fold_num, train_indices, val_indices, base_dataset_no_norm, args, device, best_lr):
    """Run training and validation for a single fold.

    Args:
        fold_num (int): Index of the current fold (0-based).
        train_indices (Iterable[int]): Indices assigned to training in this fold.
        val_indices (Iterable[int]): Indices assigned to validation in this fold.
        base_dataset_no_norm: Base dataset without normalization (used to compute per-fold stats).
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run training on.
        best_lr (float): Learning rate chosen from prior tuning.

    Returns:
        tuple[float, float]: ``(best_val_acc_percent, fold_duration_seconds)``
    """
    print(f"\n----- Starting Fold {fold_num+1}/{args.k_folds} -----")

    # Build fold loaders using train-only statistics (leakage-safe)
    train_loader, val_loader, mean, std = make_fold_loaders(
        base_dataset_no_norm, train_indices, val_indices, batch_size=args.batch_size
    )

    # Initialize a fresh model and optimizer for this fold
    model, criterion, optimizer, scheduler = build_default_components(
        device=device, lr=best_lr
    )

    # Use the Trainer
    trainer = Trainer(model, train_loader, val_loader,
                      criterion, optimizer, device, scheduler)

    fold_start_time = time.time()
    # We don't need the model state here
    best_val_acc, _ = trainer.run(args.epochs)
    fold_duration = time.time() - fold_start_time

    print(
        f"Fold {fold_num+1} finished in {fold_duration:.2f}s. Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, fold_duration


def main(args):
    """Entry point that orchestrates the full K-fold cross-validation run."""
    BEST_LR = 0.0002979295061164981  # Best learning rate found during hyperparameter tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.random_state)

    # --- 1. Load base dataset once (without normalization) ---
    _, base_dataset_no_norm = get_gtsrb_datasets()

    targets = [sample[1] for sample in base_dataset_no_norm]

    # --- 2. Setup K-Fold ---
    skf = StratifiedKFold(n_splits=args.k_folds,
                          shuffle=True, random_state=args.random_state)

    fold_results = []
    fold_times = []

    # --- 3. Loop through folds ---
    for fold_num, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        val_acc, fold_time = run_fold(
            fold_num,
            train_indices,
            val_indices,
            base_dataset_no_norm,
            args,
            device,
            BEST_LR,
        )
        fold_results.append(val_acc)
        fold_times.append(fold_time)

    # --- 4. Report Final Results ---
    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    mean_time = np.mean(fold_times)

    print("\n--- Cross-Validation Summary ---")
    print(f"Fold Accuracies: {[f'{acc:.2f}%' for acc in fold_results]}")
    print(f"Mean Validation Accuracy: {mean_accuracy:.2f}%")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.2f}")
    print(f"Average Fold Training Time: {mean_time:.2f}s")

    # Save results to a file
    results = {
        'fold_accuracies': fold_results,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'avg_fold_time': mean_time,
        'args': vars(args)
    }
    os.makedirs('results_cv', exist_ok=True)
    with open(os.path.join('results_cv', 'cv_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print("\nCross-validation complete. Summary saved to 'results_cv/cv_summary.json'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run K-Fold Cross-Validation for GTSRB classification.")
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train per fold')
    parser.add_argument('--random_state', type=int,
                        default=42, help='Seed for reproducibility')

    the_args = parser.parse_args()
    main(the_args)
