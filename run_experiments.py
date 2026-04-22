"""
Orchestrates 3 λ runs for the sparsity-aware neural network experiments.

Three λ experiments:
    | Run    | λ     | Expected outcome              |
    |--------|-------|-------------------------------|
    | Low    | 1e-5  | High accuracy, low sparsity   |
    | Medium | 1e-4  | Balanced trade-off            |
    | High   | 1e-3  | Lower accuracy, high sparsity |

Each run trains for 20 epochs with Adam (lr=1e-3) and batch size 256.
Trained models and histories are saved for Phase 3 evaluation.
"""

import os
import torch

from train import train


# Three λ values as specified in the plan
LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]
LAMBDA_NAMES = ["low", "medium", "high"]


def main():
    """Run all three λ experiments and save results."""
    # Create output directory for saved models and histories
    os.makedirs("results", exist_ok=True)

    results = {}

    for lambd, name in zip(LAMBDA_VALUES, LAMBDA_NAMES):
        print(f"\n{'#'*60}")
        print(f"# Experiment: {name.upper()} sparsity (λ = {lambd})")
        print(f"{'#'*60}")

        model, history = train(lambd=lambd, epochs=20, batch_size=256, lr=1e-3)

        # Save trained model
        model_path = f"results/model_lambda_{name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

        # Save training history
        history_path = f"results/history_lambda_{name}.pt"
        torch.save(history, history_path)
        print(f"History saved to {history_path}")

        results[name] = {
            "lambda": lambd,
            "final_accuracy": history["test_accuracy"][-1],
            "final_ce_loss": history["ce_loss"][-1],
            "final_sparsity_loss": history["sparsity_loss"][-1],
        }

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<10} {'λ':<10} {'Test Acc':<15} {'CE Loss':<15} {'Sparsity Loss':<15}")
    print("-" * 65)
    for name in LAMBDA_NAMES:
        r = results[name]
        print(
            f"{name:<10} {r['lambda']:<10.0e} "
            f"{r['final_accuracy']:<15.2f} "
            f"{r['final_ce_loss']:<15.4f} "
            f"{r['final_sparsity_loss']:<15.2f}"
        )


if __name__ == "__main__":
    main()
