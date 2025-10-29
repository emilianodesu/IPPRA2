# run_hyperparameter_tuning.py

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


def main(args):
    """ Main function to start the hyperparameter search. """
    print(f"--- Starting Hyperparameter Tuning for {args.n_trials} Trials ---")

    # Create a study object and specify the direction is to maximize accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

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
