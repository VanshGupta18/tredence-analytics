"""
PrunableLinear — A nn.Module subclass that replaces nn.Linear with learnable gate scores
for sparsity-aware training. Each weight has an associated gate that the network can
learn to drive toward zero, effectively pruning that connection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """A linear layer with learnable gate scores for differentiable pruning.

    Parameters registered via nn.Parameter:
        weights  — shape (out_features, in_features), initialized with Kaiming uniform
        bias     — shape (out_features,), initialized to zeros
        gate_scores — shape (out_features, in_features), initialized to small random
                      values (torch.randn * 0.01)

    Forward pass logic:
        1. gates = sigmoid(gate_scores)          → values ∈ (0, 1)
        2. pruned_weights = weights * gates       → element-wise mask
        3. output = F.linear(input, pruned_weights, bias)  → standard affine transform

    Why gate_scores and not gates as the parameter?
        Because sigmoid squashes the real line to (0,1), giving the optimizer a smooth,
        unconstrained space to work in. Directly learning gates ∈ (0,1) would require
        projection steps.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weights — shape (out_features, in_features), init with Kaiming uniform
        self.weights = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # bias — shape (out_features,), init to zeros
        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate_scores — shape (out_features, in_features), init to small random values
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated (prunable) weights.

        Steps:
            1. gates = sigmoid(gate_scores)          → values ∈ (0, 1)
            2. pruned_weights = weights * gates       → element-wise mask
            3. output = F.linear(input, pruned_weights, bias)
        """
        # Step 1: gates = sigmoid(gate_scores)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: pruned_weights = weights * gates (element-wise mask)
        pruned_weights = self.weights * gates

        # Step 3: output = F.linear(input, pruned_weights, bias)
        output = F.linear(x, pruned_weights, self.bias)

        return output
