"""
Main script to run enhanced training experiments with early stopping and hyperparameter tuning.

Workflow:
    1. Run hyperparameter tuning with various configurations
    2. Generate visualization plots
    3. Update final report with improved results
"""

import os
import sys
import torch


def main():
    """Main orchestration function."""
    print("\n" + "="*80)
    print("ENHANCED TRAINING EXPERIMENTS WITH EARLY STOPPING & HYPERPARAMETER TUNING")
    print("="*80 + "\n")
    
    # Step 1: Run hyperparameter tuning
    print("\nSTEP 1: Running Hyperparameter Tuning...")
    print("-" * 80)
    
    try:
        from hyperparameter_tuning import run_hyperparameter_tuning
        results = run_hyperparameter_tuning()
        print("\n✓ Hyperparameter tuning completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during hyperparameter tuning: {e}")
        return 1
    
    # Step 2: Generate visualizations
    print("\n\nSTEP 2: Generating Visualization Plots...")
    print("-" * 80)
    
    try:
        from visualize_results import generate_all_visualizations
        generate_all_visualizations("results/hyperparams")
        print("\n✓ Visualizations generated successfully!")
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        return 1
    
    # Step 3: Print summary
    print("\n\nSTEP 3: Results Summary")
    print("-" * 80)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        best = max(successful, key=lambda x: x['best_test_accuracy'])
        print(f"\n✓ BEST CONFIGURATION:")
        print(f"  Name: {best['config_name']}")
        print(f"  Test Accuracy: {best['best_test_accuracy']:.2f}%")
        print(f"  Validation Accuracy: {best['best_val_accuracy']:.2f}%")
        print(f"  Dropout Rate: {best['dropout_rate']}")
        print(f"  Lambda (Sparsity): {best['lambd']:.0e}")
        print(f"  Batch Normalization: {best['use_batch_norm']}")
        print(f"  Epochs Trained: {best['epochs_trained']}")
        print(f"  Network Sparsity: {best['sparsity_pct']:.2f}%")
        print(f"\n  Improvement over baseline:")
        baseline = next((r for r in successful if r['config_name'].startswith('baseline_lambda_1e-04')), None)
        if baseline:
            improvement = best['best_test_accuracy'] - baseline['best_test_accuracy']
            print(f"    +{improvement:.2f}% accuracy")
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved to: results/hyperparams/")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
