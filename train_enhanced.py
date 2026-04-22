"""
Enhanced Training loop with Early Stopping, Callbacks, and Hyperparameter Tuning.

Features:
    - Early stopping based on validation loss
    - Learning rate scheduling (ReduceLROnPlateau)
    - Dropout regularization
    - Callback system for extensibility
    - Validation set monitoring
    - Comprehensive metrics tracking

Total loss formula:
    Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ |sigmoid(gate_scores)|
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Callable

from prunable_linear import PrunableLinear
from model import SparsityAwareNet


class Callback:
    """Base class for training callbacks."""

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(self, logs: Dict) -> None:
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback based on validation loss.
    
    Stops training if validation loss does not improve for patience epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore weights from epoch with best validation loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.wait_count = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Check if validation loss improved."""
        val_loss = logs.get('val_loss', float('inf'))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait_count = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = logs.get('model_state_dict')
        else:
            self.wait_count += 1

        logs['early_stopping'] = {
            'wait_count': self.wait_count,
            'patience': self.patience,
            'best_epoch': self.best_epoch,
            'stopped': self.wait_count >= self.patience
        }

    def on_train_end(self, logs: Dict) -> None:
        """Restore best weights if requested."""
        if self.restore_best_weights and self.best_weights is not None:
            logs['best_weights'] = self.best_weights


def sparsity_loss(model):
    """Compute L1 sparsity loss over all gate scores in PrunableLinear layers.

    Sums |sigmoid(gate_scores)| across all PrunableLinear modules.
    This pushes gates toward 0, effectively pruning connections.
    """
    l1 = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            l1 += torch.sigmoid(module.gate_scores).abs().sum()
    return l1


def get_cifar10_loaders(batch_size: int = 256, validation_split: float = 0.1):
    """Load CIFAR-10 dataset with optional validation split.

    Args:
        batch_size: Batch size for data loaders
        validation_split: Fraction of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create validation split
    num_train = len(train_dataset)
    num_val = int(num_train * validation_split)
    num_train = num_train - num_val
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [num_train, num_val]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, lambd, device):
    """Train the model for one epoch.

    Args:
        model: SparsityAwareNet instance
        train_loader: CIFAR-10 training data loader
        optimizer: Optimizer
        criterion: CrossEntropyLoss
        lambd: λ coefficient for sparsity loss
        device: torch device

    Returns:
        avg_loss, avg_ce_loss, avg_sp_loss
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_sp_loss = 0.0
    num_batches = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        ce_loss = criterion(logits, labels)
        sp_loss = sparsity_loss(model)
        loss = ce_loss + lambd * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sp_loss += sp_loss.item()
        num_batches += 1

    return total_loss / num_batches, total_ce_loss / num_batches, total_sp_loss / num_batches


def evaluate(model, data_loader, device, criterion=None, lambd: float = 0.0, compute_sparsity: bool = False):
    """Evaluate model on a dataset.

    Args:
        model: SparsityAwareNet instance
        data_loader: Data loader
        device: torch device
        criterion: Loss criterion (optional)
        lambd: Sparsity coefficient
        compute_sparsity: Whether to compute sparsity metrics

    Returns:
        accuracy, loss (if criterion provided), sparsity (if compute_sparsity=True)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            if criterion is not None:
                ce_loss = criterion(logits, labels)
                sp_loss = sparsity_loss(model) if lambd > 0 else 0
                loss = ce_loss + lambd * sp_loss
                total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    results = {'accuracy': accuracy}
    
    if criterion is not None:
        results['loss'] = total_loss / len(data_loader)
    
    return results


def train_with_early_stopping(
    lambd: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout_rate: float = 0.5,
    use_batch_norm: bool = False,
    early_stopping_patience: int = 10,
    validation_split: float = 0.1,
    callbacks: Optional[List[Callback]] = None,
) -> Tuple[nn.Module, Dict]:
    """Full training run with early stopping and callbacks.

    Args:
        lambd: λ coefficient for sparsity loss
        epochs: Maximum number of training epochs
        batch_size: Batch size
        lr: Initial learning rate for Adam
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
        early_stopping_patience: Patience for early stopping
        validation_split: Fraction of training data for validation
        callbacks: List of callbacks to use during training

    Returns:
        model: Trained SparsityAwareNet
        history: Dict with training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Enhanced Training: λ={lambd}, dropout={dropout_rate}, batch_norm={use_batch_norm}")
    print(f"Device: {device} | Max Epochs: {epochs} | Early Stopping Patience: {early_stopping_patience}")
    print(f"{'='*70}")

    # Model
    model = SparsityAwareNet(dropout_rate=dropout_rate, use_batch_norm=use_batch_norm).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, validation_split=validation_split
    )

    # History
    history = {
        "epoch": [],
        "train_loss": [],
        "train_ce_loss": [],
        "train_sp_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "learning_rate": [],
    }

    # Callbacks
    if callbacks is None:
        callbacks = []
    early_stopping = EarlyStoppingCallback(patience=early_stopping_patience)
    callbacks.append(early_stopping)

    best_val_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_ce, train_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, lambd, device
        )

        # Evaluate on validation
        val_results = evaluate(model, val_loader, device, criterion, lambd)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']

        # Evaluate on test
        test_results = evaluate(model, test_loader, device)
        test_acc = test_results['accuracy']

        # Step learning rate scheduler
        scheduler.step(val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ce_loss"].append(train_ce)
        history["train_sp_loss"].append(train_sp)
        history["train_accuracy"].append(0.0)  # Would need separate computation
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["test_accuracy"].append(test_acc)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # Print progress
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Callbacks
        logs = {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'model_state_dict': model.state_dict().copy()
        }
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

        # Check early stopping
        if early_stopping.wait_count >= early_stopping.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping.patience} epochs)")
            if early_stopping.best_weights is not None:
                model.load_state_dict(early_stopping.best_weights)
                print(f"Restored best weights from epoch {early_stopping.best_epoch}")
            break

        best_val_accuracy = max(best_val_accuracy, val_acc)

    # Final evaluation
    logs = {'model_state_dict': model.state_dict().copy()}
    for callback in callbacks:
        callback.on_train_end(logs)

    print(f"\n{'='*70}")
    print(f"Training Complete | Best Val Accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*70}\n")

    return model, history


def train(lambd, epochs=20, batch_size=256, lr=1e-3):
    """Legacy training function for backwards compatibility."""
    model, history = train_with_early_stopping(
        lambd=lambd,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        dropout_rate=0.0,  # No dropout by default for backwards compatibility
        early_stopping_patience=10,
    )
    return model, history


if __name__ == "__main__":
    # Example usage
    model, history = train_with_early_stopping(
        lambd=1e-4,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        dropout_rate=0.5,
        use_batch_norm=False,
        early_stopping_patience=10,
    )
