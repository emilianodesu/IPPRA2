# train.py

import argparse
import os
import torch
from torch import nn
from torch import optim

from src.data_loader import create_dataloaders
from src.experiments import set_seeds
from src.model import GTSRB_CNN
from src.trainer import Trainer

def main(args):
    """
    Main function to set up and run a single training process.
    """
    # --- 1. Setup ---
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. Load Data ---
    # Use a few workers and pin_memory on CUDA for faster host-to-device transfer
    pin_memory = device.type == 'cuda'
    train_loader, val_loader, _, _, _, _ = create_dataloaders(
        batch_size=args.batch_size,
        random_state=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = GTSRB_CNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --- 4. Use Trainer to run the training loop ---
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler)
    best_val_acc, best_model_state = trainer.run(args.epochs)

    # --- 5. Save the best model ---
    if best_model_state:
        save_path = os.path.join(args.save_dir, 'best_model.pth')
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
