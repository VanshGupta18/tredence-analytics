"""
Visualization script for training results with plots and curves.

Generates:
    - Training loss curves
    - Validation accuracy curves
    - Comparison plots across hyperparameters
    - Learning rate scheduling curves
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from pathlib import Path


def plot_training_curves(history, config_name, save_path="results/hyperparams"):
    """Plot training metrics for a single configuration.
    
    Args:
        history: Training history dict
        config_name: Configuration name
        save_path: Path to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Plot 1: Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CE Loss and Sparsity Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['epoch'], history['train_ce_loss'], label='CE Loss', marker='o', markersize=3)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(history['epoch'], history['train_sp_loss'], label='Sparsity Loss', 
                  color='red', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CE Loss')
    ax2_twin.set_ylabel('Sparsity Loss', color='red')
    ax2.set_title('Loss Components')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation & Test Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy', marker='o', markersize=3)
    ax3.plot(history['epoch'], history['test_accuracy'], label='Test Accuracy', marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Validation & Test Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate Schedule
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(history['epoch'], history['learning_rate'], marker='o', markersize=3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate (log scale)')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Accuracy Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    epochs = history['epoch']
    ax5.fill_between(epochs, history['val_accuracy'], history['test_accuracy'], 
                     alpha=0.3, label='Gap')
    ax5.plot(epochs, history['val_accuracy'], label='Val Acc', marker='o', markersize=3)
    ax5.plot(epochs, history['test_accuracy'], label='Test Acc', marker='s', markersize=3)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Accuracy Gap (Generalization)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Training Progress Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    best_val_idx = history['val_accuracy'].index(max(history['val_accuracy']))
    summary_text = f"""
Training Summary
{'='*40}

Best Epoch: {history['epoch'][best_val_idx]}
Best Val Accuracy: {max(history['val_accuracy']):.2f}%
Final Test Accuracy: {history['test_accuracy'][-1]:.2f}%
Total Epochs: {len(history['epoch'])}

Final Metrics:
  Train Loss: {history['train_loss'][-1]:.4f}
  Val Loss: {history['val_loss'][-1]:.4f}
  Learning Rate: {history['learning_rate'][-1]:.6f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Training Curves: {config_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plot_path = f"{save_path}/curves_{config_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_comparison_matrix(results_summary_path, save_path="results/hyperparams"):
    """Create comparison matrix of all hyperparameter configurations.
    
    Args:
        results_summary_path: Path to results_summary.json
        save_path: Path to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Load results
    with open(results_summary_path, 'r') as f:
        results = json.load(f)
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    names = [r['config_name'] for r in successful_results]
    test_accs = [r['best_test_accuracy'] for r in successful_results]
    sparsities = [r['sparsity_pct'] for r in successful_results]
    dropouts = [r['dropout_rate'] for r in successful_results]
    lambdas = [r['lambd'] for r in successful_results]
    epochs_trained = [r['epochs_trained'] for r in successful_results]
    
    # Plot 1: Test Accuracy Comparison
    ax = axes[0, 0]
    colors = ['red' if d == 0.0 else 'blue' if d < 0.5 else 'green' 
              for d in dropouts]
    bars = ax.barh(range(len(names)), test_accs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=9)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Across Configurations')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(test_accs):
        ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=8)
    
    # Plot 2: Sparsity Comparison
    ax = axes[0, 1]
    bars = ax.barh(range(len(names)), sparsities, color='purple', alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=9)
    ax.set_xlabel('Sparsity (%)')
    ax.set_title('Network Sparsity Across Configurations')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(sparsities):
        ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=8)
    
    # Plot 3: Accuracy vs Sparsity Trade-off
    ax = axes[1, 0]
    scatter = ax.scatter(sparsities, test_accs, s=100, c=dropouts, cmap='RdYlGn', alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(f'{dropouts[i]:.1f}', (sparsities[i], test_accs[i]), 
                   fontsize=8, ha='center', va='center')
    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy-Sparsity Trade-off')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dropout Rate')
    
    # Plot 4: Training Efficiency
    ax = axes[1, 1]
    colors_lambda = ['red' if l == 1e-5 else 'blue' if l == 1e-4 else 'green'
                     for l in lambdas]
    scatter = ax.scatter(epochs_trained, test_accs, s=100, c=colors_lambda, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(f'{lambdas[i]:.0e}', (epochs_trained[i], test_accs[i]),
                   fontsize=8, ha='center', va='center')
    ax.set_xlabel('Epochs Trained (Early Stopping)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Training Efficiency')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Hyperparameter Tuning Results Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = f"{save_path}/comparison_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison matrix: {plot_path}")


def generate_all_visualizations(hyperparams_dir="results/hyperparams"):
    """Generate all visualization plots from hyperparameter tuning results.
    
    Args:
        hyperparams_dir: Directory containing hyperparameter results
    """
    print(f"\n{'='*80}")
    print("Generating Visualization Plots")
    print(f"{'='*80}\n")
    
    # Get all history files
    history_files = list(Path(hyperparams_dir).glob("history_*.pt"))
    
    print(f"Found {len(history_files)} history files")
    print("Generating training curves for each configuration...")
    
    for history_file in sorted(history_files):
        config_name = history_file.stem.replace('history_', '')
        print(f"\n  {config_name}")
        
        history = torch.load(history_file)
        plot_training_curves(history, config_name, hyperparams_dir)
    
    # Generate comparison matrix
    results_summary_path = Path(hyperparams_dir) / "results_summary.json"
    if results_summary_path.exists():
        print(f"\nGenerating comparison matrix...")
        plot_comparison_matrix(str(results_summary_path), hyperparams_dir)
    
    print(f"\n{'='*80}")
    print("Visualization generation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    generate_all_visualizations()
