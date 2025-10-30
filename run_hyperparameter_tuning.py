"""Hyperparameter tuning with Optuna for GTSRB classification.

This script runs an Optuna study to optimize hyperparameters (currently the
learning rate) for the :class:`src.model.GTSRB_CNN`. Each trial performs
K-fold cross-validation where normalization statistics are computed on the
TRAIN SPLIT ONLY per fold to avoid leakage, then datasets are rebuilt for
training/validation of that fold. The objective returns mean validation
accuracy across folds.

Workflow
--------
1) Suggest hyperparameters for the trial (e.g., learning rate).
2) Load the base dataset without normalization and build stratified folds.
3) For each fold, compute train-only mean/std, rebuild datasets and loaders,
     train the model, and record best validation accuracy.
4) Return the mean accuracy across folds for Optuna to maximize.

Command-line arguments
----------------------
--n_trials: Number of optimization trials to run (default: 25).

Notes
-----
- Reproducibility: ``set_seeds`` is called for fold-level reproducibility of
    augmentations and initializations.
- Leakage-safety: Per-fold normalization is used (no peeking at validation).
- Efficiency: Fewer folds and fewer epochs are used in tuning than in the
    final cross-validation to reduce runtime while preserving ranking fidelity.
"""

import argparse
import torch
from sklearn.model_selection import StratifiedKFold
import optuna
import numpy as np

from src.trainer import Trainer
from src.data_loader import get_gtsrb_datasets
from src.experiments import set_seeds, make_fold_loaders, build_model_components

def objective(trial):
    """Objective function for Optuna.

    Each trial runs K-fold cross-validation with the suggested hyperparameters.
    For strict evaluation and to prevent data leakage, we compute normalization
    statistics on the TRAIN SPLIT ONLY per fold, then rebuild datasets with
    those stats before training/validation.
    """
    # --- 1. Suggest Hyperparameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = "Adam"
    epochs = 10 # Keep epochs fixed for a fair comparison between trials
    batch_size = 64

    # --- 2. Load datasets WITHOUT normalization to derive splits and per-fold stats ---
    # At this stage, transforms are Resize/ToTensor only; no normalization applied.
    _, full_dataset_no_aug_no_norm = get_gtsrb_datasets()
    targets = [sample[1] for sample in full_dataset_no_aug_no_norm]

    # --- 3. Run Cross-Validation for this trial ---
    k_folds = 3 # Use fewer folds (e.g., 3) for faster tuning
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    for _, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        # Optional: seed for reproducibility of augmentations, etc.
        set_seeds(42)

        # Create DataLoaders for this fold using train-only normalization stats
        train_loader, val_loader, _, _ = make_fold_loaders(
            full_dataset_no_aug_no_norm,
            train_indices,
            val_indices,
            batch_size=batch_size,
        )

        # Initialize model and optimizer
        model, criterion, optimizer, _ = build_model_components(
            lr=lr, optimizer_name=optimizer_name, device=device
        )

        # Run training using our Trainer class
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
        best_val_acc, _ = trainer.run(epochs)
        fold_accuracies.append(best_val_acc)

    # --- 4. Return the mean accuracy for Optuna to maximize ---
    mean_accuracy = np.mean(fold_accuracies)
    return float(mean_accuracy)


def main(parsed_args):
    """ Main function to start the hyperparameter search. """
    print(f"--- Starting Hyperparameter Tuning for {parsed_args.n_trials} Trials ---")

    # Create a study object and specify the direction is to maximize accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=parsed_args.n_trials)

    print("\n--- Tuning Complete ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (Mean Accuracy): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GTSRB classification.")
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of optimization trials to run.')
    args = parser.parse_args()
    main(args)
