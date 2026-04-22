"""
Full network using PrunableLinear layers for CIFAR-10 classification.

Architecture:
    CIFAR-10 (3×32×32) → Flatten → PrunableLinear(3072→512) → ReLU → Dropout
                       → PrunableLinear(512→256) → ReLU → Dropout
                       → PrunableLinear(256→128) → ReLU → Dropout
                       → PrunableLinear(128→10) → Output

Enhanced with:
    - Dropout regularization after each hidden layer
    - Configurable dropout rate
    - Batch normalization support
"""

import torch.nn as nn

from prunable_linear import PrunableLinear


class SparsityAwareNet(nn.Module):
    """A sparsity-aware feedforward network for CIFAR-10 using PrunableLinear layers.

    The network flattens the 3×32×32 input images and passes them through
    four PrunableLinear layers with ReLU activations (except the final output layer).
    
    Includes dropout regularization after each hidden layer for improved generalization.
    
    Args:
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: False)
    """

    def __init__(self, dropout_rate: float = 0.5, use_batch_norm: bool = False) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.flatten = nn.Flatten()

        layers = []
        
        # Layer 1: 3072 → 512
        layers.append(PrunableLinear(3072, 512))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Layer 2: 512 → 256
        layers.append(PrunableLinear(512, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Layer 3: 256 → 128
        layers.append(PrunableLinear(256, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Layer 4: 128 → 10
        layers.append(PrunableLinear(128, 10))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass: flatten input, then pass through PrunableLinear layers."""
        x = self.flatten(x)
        x = self.layers(x)
        return x
