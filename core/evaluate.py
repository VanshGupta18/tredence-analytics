"""
Evaluation metrics + visualization for sparsity-aware neural network.

Provides:
    - compute_sparsity(): percentage of gates below threshold
    - evaluate_all_experiments(): accuracy-sparsity table across λ values
    - plot_gate_distributions(): histogram of sigmoid(gate_scores) per λ

Visualization plan (from plan.md):
    - One figure with 3 subplots (one per λ)
    - Each: histogram of sigmoid(gate_scores) values across all layers
    - Expected shape: as λ increases, a sharp spike at 0 grows — the network is "self-pruning"
"""

import torch
import matplotlib.pyplot as plt

try:
    from .prunable_linear import PrunableLinear
    from .model import SparsityAwareNet
except ImportError:
    from prunable_linear import PrunableLinear
    from model import SparsityAwareNet


def compute_sparsity(model, threshold=1e-2):
    """Compute the percentage of gates below the threshold.

    Args:
        model: SparsityAwareNet instance
        threshold: gate value below which a weight is considered pruned (default: 1e-2)

    Returns:
        sparsity: fraction of gates < threshold (0.0 to 1.0)
    """
    total, pruned = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            pruned += (gates < threshold).sum().item()
            total += gates.numel()
    return pruned / total  # percentage


def get_all_gate_values(model):
    """Collect all sigmoid(gate_scores) values across all PrunableLinear layers.

    Args:
        model: SparsityAwareNet instance

    Returns:
        gate_values: 1D tensor of all gate values
    """
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu()
            all_gates.append(gates.flatten())
    return torch.cat(all_gates)


def count_active_weights(model, threshold=1e-2):
    """Count the number of active (non-pruned) weights.

    Args:
        model: SparsityAwareNet instance
        threshold: gate value below which a weight is considered pruned

    Returns:
        active: number of active weights
        total: total number of weights
    """
    active, total = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            active += (gates >= threshold).sum().item()
            total += gates.numel()
    return active, total


def evaluate_all_experiments():
    """Load all 3 trained models and produce the accuracy-sparsity table.

    Prints and returns the table specified in plan.md:
        | λ | Test Accuracy | % Gates < 1e-2 | Active Weights |

    Returns:
        results: list of dicts with metrics for each λ
    """
    try:
        from .train import get_cifar10_loaders, evaluate
    except ImportError:
        from train import get_cifar10_loaders, evaluate

    lambda_values = [1e-5, 1e-4, 1e-3]
    lambda_names = ["low", "medium", "high"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_cifar10_loaders(batch_size=256)

    results = []

    print(f"\n{'='*70}")
    print("ACCURACY–SPARSITY TABLE")
    print(f"{'='*70}")
    print(f"{'λ':<10} {'Test Accuracy':<18} {'% Gates < 1e-2':<20} {'Active Weights':<18}")
    print("-" * 70)

    for lambd, name in zip(lambda_values, lambda_names):
        model = SparsityAwareNet().to(device)
        model_path = f"results/model_lambda_{name}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Test accuracy
        test_acc = evaluate(model, test_loader, device)

        # Sparsity metric
        sparsity = compute_sparsity(model, threshold=1e-2)

        # Active weights
        active, total = count_active_weights(model, threshold=1e-2)

        result = {
            "lambda": lambd,
            "name": name,
            "test_accuracy": test_acc,
            "sparsity_pct": sparsity * 100,
            "active_weights": active,
            "total_weights": total,
        }
        results.append(result)

        print(
            f"{lambd:<10.0e} "
            f"{test_acc:<18.2f} "
            f"{sparsity * 100:<20.2f} "
            f"{active:,}/{total:,}"
        )

    print(f"{'='*70}")
    return results


def plot_gate_distributions():
    """Plot gate distribution histograms — one subplot per λ.

    Visualization plan (from plan.md):
        - One figure with 3 subplots (one per λ)
        - Each: histogram of sigmoid(gate_scores) values across all layers
        - Expected shape: as λ increases, a sharp spike at 0 grows — "self-pruning"

    Saves the plot to results/gate_distributions.png
    """
    lambda_values = [1e-5, 1e-4, 1e-3]
    lambda_names = ["low", "medium", "high"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Gate Score Distributions Across λ Values", fontsize=16, fontweight="bold")

    for idx, (lambd, name) in enumerate(zip(lambda_values, lambda_names)):
        model = SparsityAwareNet().to(device)
        model_path = f"results/model_lambda_{name}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Collect all gate values
        gate_values = get_all_gate_values(model).numpy()

        ax = axes[idx]
        ax.hist(gate_values, bins=100, color="#2196F3", alpha=0.8, edgecolor="black",
                linewidth=0.3)
        ax.set_title(f"λ = {lambd:.0e} ({name.capitalize()})", fontsize=13)
        ax.set_xlabel("sigmoid(gate_scores)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(0, 1)

        # Add sparsity annotation
        sparsity = compute_sparsity(model, threshold=1e-2)
        ax.annotate(
            f"Sparsity: {sparsity * 100:.1f}%",
            xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig("results/gate_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nGate distribution plot saved to results/gate_distributions.png")


if __name__ == "__main__":
    print("Evaluating all experiments...")
    results = evaluate_all_experiments()
    print("\nGenerating gate distribution plots...")
    plot_gate_distributions()
    print("\nEvaluation complete ✓")
