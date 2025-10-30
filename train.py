"""Training entry point for GTSRB traffic sign classification.

This script trains a :class:`src.model.GTSRB_CNN` on the GTSRB dataset using
PyTorch. It sets seeds for reproducibility, configures performant data loading
(num_workers, pin_memory on CUDA), runs a standard training loop via
``src.trainer.Trainer``, and saves the best-performing model checkpoint.

Outputs
-------
- ``<save_dir>/best_model.pth``: Best validation-accuracy model state dict.
- Console logs with per-epoch metrics and the best validation accuracy.

Command-line arguments
----------------------
--lr:          Learning rate for the optimizer (default: 1e-3)
--batch_size:  Batch size for training (default: 64)
--epochs:      Number of training epochs (default: 20)
--save_dir:    Directory to write model checkpoints (default: models)
--seed:        Random seed for reproducibility (default: 42)
--num_workers: DataLoader worker processes for I/O (default: 2)

Notes
-----
- Seeding is applied via ``src.experiments.set_seeds`` for reproducible runs.
- ``pin_memory`` is enabled when running on CUDA to speed up host-to-device
    transfers.
- The actual training and validation loop is encapsulated in ``Trainer`` to
    keep this script minimal and focused on orchestration.
"""

import argparse
import os
import torch
from torch import nn
from torch import optim

from src.data_loader import create_dataloaders
from src.experiments import set_seeds
from src.model import GTSRB_CNN
from src.trainer import Trainer

def main(parsed_args):
    """
    Main function to set up and run a single training process.
    """
    # --- 1. Setup ---
    set_seeds(parsed_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(parsed_args.save_dir, exist_ok=True)

    # --- 2. Load Data ---
    # Use a few workers and pin_memory on CUDA for faster host-to-device transfer
    pin_memory = device.type == 'cuda'
    train_loader, val_loader, _, _, _, _ = create_dataloaders(
        batch_size=parsed_args.batch_size,
        random_state=parsed_args.seed,
        num_workers=parsed_args.num_workers,
        pin_memory=pin_memory,
    )

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = GTSRB_CNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=parsed_args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --- 4. Use Trainer to run the training loop ---
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler)
    best_val_acc, best_model_state = trainer.run(parsed_args.epochs)

    # --- 5. Save the best model ---
    if best_model_state:
        save_path = os.path.join(parsed_args.save_dir, 'best_model.pth')
        torch.save(best_model_state, save_path)
        print(f"\nTraining complete. Best model saved to {save_path}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    else:
        print("\nTraining complete, but no model was saved as validation accuracy did not improve.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN for GTSRB classification.")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save the best model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers for I/O')

    args = parser.parse_args()
    main(args)
