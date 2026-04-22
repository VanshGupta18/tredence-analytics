## Project Plan: Sparsity-Aware Neural Network with Prunable Linear Layers

---

### Architecture Overview

```
CIFAR-10 (3×32×32) → Flatten → PrunableLinear(3072→512) → ReLU
                   → PrunableLinear(512→256) → ReLU
                   → PrunableLinear(256→128) → ReLU
                   → PrunableLinear(128→10) → Output
```

---

### Phase 1 — `PrunableLinear` Layer

**What to build:** A `nn.Module` subclass that replaces `nn.Linear`.

**Parameters to register via `nn.Parameter`:**
- `weights` — shape `(out_features, in_features)`, init with Kaiming uniform
- `bias` — shape `(out_features,)`, init to zeros
- `gate_scores` — shape `(out_features, in_features)`, init to small random values (e.g. `torch.randn * 0.01`)

**Forward pass logic (step by step):**
1. `gates = sigmoid(gate_scores)` → values ∈ (0, 1)
2. `pruned_weights = weights * gates` → element-wise mask
3. `output = F.linear(input, pruned_weights, bias)` → standard affine transform

**Why `gate_scores` not `gates` as the parameter?** Because sigmoid squashes the real line to (0,1), giving the optimizer a smooth, unconstrained space to work in. Directly learning gates ∈ (0,1) would require projection steps.

---

### Phase 2 — Sparsity Loss & Training Loop

**Total loss formula:**
```
Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ |sigmoid(gate_scores)|
```

**Sparsity loss implementation:**
```python
def sparsity_loss(model):
    l1 = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            l1 += torch.sigmoid(module.gate_scores).abs().sum()
    return l1
```

**Why L1 and not L2?** L1 penalizes all non-zero values equally regardless of magnitude — this creates a "pressure toward exactly zero" that L2 (which penalizes large values more) cannot produce. L1 is the convex relaxation of the L0 (count of nonzeros) norm, making it the standard tool for inducing sparsity.

**Three λ experiments:**

| Run | λ | Expected outcome |
|-----|---|-----------------|
| Low | `1e-5` | High accuracy, low sparsity |
| Medium | `1e-4` | Balanced trade-off |
| High | `1e-3` | Lower accuracy, high sparsity |

**Training config:**
- Dataset: CIFAR-10 (50k train / 10k test), normalized
- Optimizer: Adam, lr=1e-3
- Epochs: 20 per run
- Batch size: 256

---

### Phase 3 — Evaluation & Report

**Sparsity metric:**
```python
def compute_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            pruned += (gates < threshold).sum().item()
            total += gates.numel()
    return pruned / total  # percentage
```

**Accuracy–Sparsity table** (to appear in report):

| λ | Test Accuracy | % Gates < 1e-2 | Active Weights |
|---|--------------|----------------|----------------|
| 1e-5 | ~% | ~% | ~M |
| 1e-4 | ~% | ~% | ~M |
| 1e-3 | ~% | ~% | ~M |

**Visualization plan:**
- One figure with 3 subplots (one per λ)
- Each: histogram of `sigmoid(gate_scores)` values across all layers
- Expected shape: as λ increases, a sharp spike at 0 grows — the network is "self-pruning"

---

### File Structure

```
project/
├── prunable_linear.py       # PrunableLinear class
├── model.py                 # Full network using PrunableLinear
├── train.py                 # Training loop + sparsity loss
├── evaluate.py              # Metrics + visualization
├── run_experiments.py       # Orchestrates 3 λ runs
└── report.md                # Final Markdown report
```

---

### Report Outline

1. **Introduction** — what learnable pruning is and why it matters
2. **Mathematical foundation** — why L1 → sparsity (subdifferential at 0, comparison to L0/L2)
3. **Implementation details** — PrunableLinear walkthrough
4. **Results table** — accuracy vs. sparsity across λ values
5. **Gate distribution plots** — histogram per λ
6. **Analysis** — accuracy-sparsity trade-off interpretation
7. **Conclusion** — which λ is "best" and why

---