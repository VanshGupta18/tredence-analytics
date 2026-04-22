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
from typing import Dict, List, Tuple, Optional

try:
    from .prunable_linear import PrunableLinear
    from .model import SparsityAwareNet
except ImportError:
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
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        lambd: λ coefficient for sparsity loss
        device: torch device

    Returns:
        avg_loss: average total loss over the epoch
        avg_ce_loss: average cross-entropy loss over the epoch
        avg_sp_loss: average sparsity loss over the epoch
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

        # Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ |sigmoid(gate_scores)|
        ce_loss = criterion(logits, labels)
        sp_loss = sparsity_loss(model)
        loss = ce_loss + lambd * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sp_loss += sp_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_sp_loss = total_sp_loss / num_batches

    return avg_loss, avg_ce_loss, avg_sp_loss


def evaluate(model, data_loader, device, criterion=None, lambd: float = 0.0):
    """Evaluate model on a dataset.

    Args:
        model: SparsityAwareNet instance
        data_loader: Data loader
        device: torch device
        criterion: Loss criterion (optional)
        lambd: Sparsity coefficient

    Returns:
        accuracy, loss (if criterion provided)
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
    
    if criterion is not None:
        return accuracy, total_loss / len(data_loader)
    return accuracy


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
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
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
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_ce, train_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, lambd, device
        )

        # Evaluate on validation
        val_acc, val_loss = evaluate(model, val_loader, device, criterion, lambd)

        # Evaluate on test
        test_acc = evaluate(model, test_loader, device)

        # Step learning rate scheduler
        scheduler.step(val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ce_loss"].append(train_ce)
        history["train_sp_loss"].append(train_sp)
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

        # Track best validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch

        # Check early stopping
        if early_stopping.wait_count >= early_stopping.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping.patience} epochs)")
            if early_stopping.best_weights is not None:
                model.load_state_dict(early_stopping.best_weights)
                print(f"Restored best weights from epoch {early_stopping.best_epoch}")
            break

    # Final evaluation
    logs = {'model_state_dict': model.state_dict().copy()}
    for callback in callbacks:
        callback.on_train_end(logs)

    print(f"\n{'='*70}")
    print(f"Training Complete | Best Val Accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"{'='*70}\n")

    return model, history


def train(lambd, epochs=20, batch_size=256, lr=1e-3):
    """Legacy training function for backwards compatibility with old code."""
    # Use old evaluation method for backwards compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training with λ = {lambd} on {device}")
    print(f"{'='*60}")

    model = SparsityAwareNet(dropout_rate=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, _, test_loader = get_cifar10_loaders(batch_size=batch_size, validation_split=0.0)

    history = {
        "epoch": [],
        "total_loss": [],
        "ce_loss": [],
        "sparsity_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        avg_loss, avg_ce, avg_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, lambd, device
        )
        test_acc = evaluate(model, test_loader, device)

        history["epoch"].append(epoch)
        history["total_loss"].append(avg_loss)
        history["ce_loss"].append(avg_ce)
        history["sparsity_loss"].append(avg_sp)
        history["test_accuracy"].append(test_acc)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Total Loss: {avg_loss:.4f} | "
            f"CE Loss: {avg_ce:.4f} | "
            f"Sparsity Loss: {avg_sp:.2f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return model, history
