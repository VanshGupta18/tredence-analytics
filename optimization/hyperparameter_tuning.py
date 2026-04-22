"""
Hyperparameter tuning experiments with early stopping and improved configurations.

Runs systematic hyperparameter search with:
    - Different dropout rates
    - Different λ (sparsity) values
    - Early stopping for each run
    - Comprehensive results tracking
"""

import os
import sys
import torch
import json

# Add parent directory to path to import from core/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train import train_with_early_stopping
from core.evaluate import compute_sparsity, count_active_weights
from core.model import SparsityAwareNet


def run_hyperparameter_tuning():
    """Run systematic hyperparameter tuning experiments."""
    
    # Hyperparameter grid
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    lambda_values = [1e-5, 1e-4, 1e-3]
    batch_sizes = [128, 256]
    learning_rates = [1e-3, 5e-4]
    
    # We'll do a focused search: combine dropout with lambda
    configs = []
    
    # Config 1: No dropout, different lambdas (baseline)
    for lambd in lambda_values:
        configs.append({
            'name': f'baseline_lambda_{lambd:.0e}',
            'lambd': lambd,
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'batch_size': 256,
            'lr': 1e-3,
        })
    
    # Config 2: With dropout, different rates
    for dropout in dropout_rates[1:]:  # Skip 0.0 as it's already in baseline
        for lambd in [1e-4]:  # Focus on medium lambda for dropout experiments
            configs.append({
                'name': f'dropout_{dropout}_lambda_{lambd:.0e}',
                'lambd': lambd,
                'dropout_rate': dropout,
                'use_batch_norm': False,
                'batch_size': 256,
                'lr': 1e-3,
            })
    
    # Config 3: Batch norm + dropout
    for dropout in [0.3, 0.5]:
        for lambd in [1e-4]:
            configs.append({
                'name': f'batchnorm_dropout_{dropout}_lambda_{lambd:.0e}',
                'lambd': lambd,
                'dropout_rate': dropout,
                'use_batch_norm': True,
                'batch_size': 256,
                'lr': 1e-3,
            })
    
    # Create results directory
    os.makedirs("results/hyperparams", exist_ok=True)
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {len(configs)} configurations")
    print(f"{'='*80}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Config {i}/{len(configs)}: {config['name']}")
        print(f"{'#'*80}")
        
        try:
            # Train model with early stopping
            model, history = train_with_early_stopping(
                lambd=config['lambd'],
                epochs=150,  # Allow many epochs with early stopping
                batch_size=config['batch_size'],
                lr=config['lr'],
                dropout_rate=config['dropout_rate'],
                use_batch_norm=config['use_batch_norm'],
                early_stopping_patience=15,
                validation_split=0.1,
            )
            
            # Compute sparsity metrics
            model = model.to(device)
            sparsity = compute_sparsity(model, threshold=1e-2)
            active, total = count_active_weights(model, threshold=1e-2)
            
            # Get best metrics
            best_val_acc = max(history['val_accuracy'])
            best_test_acc = history['test_accuracy'][history['val_accuracy'].index(best_val_acc)]
            final_epochs = len(history['epoch'])
            
            result = {
                'config_name': config['name'],
                'lambd': config['lambd'],
                'dropout_rate': config['dropout_rate'],
                'use_batch_norm': config['use_batch_norm'],
                'batch_size': config['batch_size'],
                'learning_rate': config['lr'],
                'best_val_accuracy': round(best_val_acc, 2),
                'best_test_accuracy': round(best_test_acc, 2),
                'epochs_trained': final_epochs,
                'final_val_loss': round(history['val_loss'][-1], 4),
                'sparsity_pct': round(sparsity * 100, 2),
                'active_weights': active,
                'total_weights': total,
                'status': 'SUCCESS'
            }
            
            results.append(result)
            
            # Save model
            model_save_path = f"results/hyperparams/model_{config['name']}.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"\n✓ Model saved: {model_save_path}")
            
            # Save history
            history_save_path = f"results/hyperparams/history_{config['name']}.pt"
            torch.save(history, history_save_path)
            print(f"✓ History saved: {history_save_path}")
            
            print(f"\n  Best Val Acc: {result['best_val_accuracy']:.2f}%")
            print(f"  Best Test Acc: {result['best_test_accuracy']:.2f}%")
            print(f"  Epochs: {final_epochs}")
            print(f"  Sparsity: {result['sparsity_pct']:.2f}%")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            result = {
                'config_name': config['name'],
                'lambd': config['lambd'],
                'dropout_rate': config['dropout_rate'],
                'status': 'FAILED',
                'error': str(e)
            }
            results.append(result)
    
    # Save results summary
    results_path = "results/hyperparams/results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\n{'='*80}")
    print(f"Results summary saved to: {results_path}")
    print(f"{'='*80}\n")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<40} {'Val Acc':<10} {'Test Acc':<10} {'Epochs':<10} {'Sparse %':<10}")
    print("-" * 80)
    
    for r in results:
        if r['status'] == 'SUCCESS':
            config_name = r['config_name'][:38]
            print(
                f"{config_name:<40} "
                f"{r['best_val_accuracy']:<10.2f} "
                f"{r['best_test_accuracy']:<10.2f} "
                f"{r['epochs_trained']:<10} "
                f"{r['sparsity_pct']:<10.2f}"
            )
        else:
            print(f"{r['config_name']:<40} FAILED")
    
    print(f"{'='*80}\n")
    
    # Find best configuration
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    if successful_results:
        best = max(successful_results, key=lambda x: x['best_test_accuracy'])
        print(f"BEST CONFIGURATION:")
        print(f"  Name: {best['config_name']}")
        print(f"  Test Accuracy: {best['best_test_accuracy']:.2f}%")
        print(f"  Dropout: {best['dropout_rate']}")
        print(f"  Lambda: {best['lambd']:.0e}")
        print(f"  Batch Norm: {best['use_batch_norm']}")
        print(f"  Epochs: {best['epochs_trained']}")
    
    return results


if __name__ == "__main__":
    results = run_hyperparameter_tuning()
