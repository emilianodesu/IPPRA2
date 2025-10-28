"""Training loop utilities for supervised image classification.

This module provides a ``Trainer`` class that encapsulates the boilerplate for
epoch-based training and validation with PyTorch. It handles:

- Switching the model between train/eval modes
- Forward/backward passes, optimizer stepping
- Loss/accuracy aggregation
- Optional LR scheduler stepping per-epoch
- A simple history dictionary for post-hoc analysis/plotting

Progress bars are rendered with ``tqdm.auto`` and should work in notebooks and
terminals alike.
"""

import torch
from tqdm.auto import tqdm

class Trainer:
    """High-level trainer for model fitting and validation.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): Batches for training.
        val_loader (torch.utils.data.DataLoader): Batches for validation.
        criterion (Callable): Loss function mapping (outputs, targets) -> loss tensor.
        optimizer (torch.optim.Optimizer): Optimizer instance for ``model.parameters()``.
        device (torch.device): Device on which to run computations.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Optional LR scheduler
            stepped once per epoch, after validation.

    Attributes:
        history (dict[str, list[float]]): Aggregated metrics across epochs with keys:
            - ``train_loss``
            - ``train_acc`` (percent)
            - ``val_loss``
            - ``val_acc`` (percent)
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        """Initializes the Trainer with model, data loaders, loss function, optimizer, and device."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _train_epoch(self, epoch=None, epochs=None):
        """Run a single training epoch.

        Returns:
            tuple[float, float]: ``(avg_loss, accuracy_percent)`` for the epoch.
        """
        return self._run_epoch(
            loader=self.train_loader,
            train=True,
            epoch=epoch,
            epochs=epochs,
            phase_name="Train",
        )

    def _validate_epoch(self, epoch=None, epochs=None):
        """Run a single validation epoch without gradients.

        Returns:
            tuple[float, float]: ``(avg_loss, accuracy_percent)`` for the epoch.
        """
        return self._run_epoch(
            loader=self.val_loader,
            train=False,
            epoch=epoch,
            epochs=epochs,
            phase_name="Val",
        )

    def _run_epoch(self, loader, train: bool, epoch=None, epochs=None, phase_name: str = "?"):
        """Common epoch runner for training and validation phases.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader to iterate.
            train (bool): If True, enables training mode and backprop; otherwise eval/no_grad.
            epoch (int | None): 0-based epoch index for display purposes only.
            epochs (int | None): Total epochs for display purposes only.
            phase_name (str): Short label used in the progress bar (e.g., ``"Train"`` or ``"Val"``).

        Returns:
            tuple[float, float]: ``(epoch_loss, epoch_accuracy_percent)``
        """
        desc = f"{phase_name:<5}[{(epoch+1) if epoch is not None else '?'} / {epochs if epochs is not None else '?'}]"

        if train:
            self.model.train()
            context = torch.enable_grad()
        else:
            self.model.eval()
            context = torch.no_grad()

        running_loss = 0.0
        correct = 0
        total = 0

        with context:
            for inputs, labels in tqdm(
                loader,
                desc=desc,
                leave=False,
                total=len(loader),
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / max(total, 1)
        return epoch_loss, epoch_acc

    def run(self, epochs):
        """Execute the training/validation loop for ``epochs`` iterations.

        LR scheduler (if provided) is stepped once per epoch, after validation.

        Args:
            epochs (int): Number of epochs to run.

        Returns:
            tuple[float, dict]: ``(best_val_accuracy_percent, best_model_state_dict)``
        """
        best_val_accuracy = 0.0
        best_model_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(epoch, epochs)
            val_loss, val_acc = self._validate_epoch(epoch, epochs)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict()
                print(f"-> New best model found with accuracy: {best_val_accuracy:.2f}%")

            if self.scheduler:
                self.scheduler.step()

        return best_val_accuracy, best_model_state
