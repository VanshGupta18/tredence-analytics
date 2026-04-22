# Ready-to-Use Colab Script

Save this as a new notebook in Google Colab and run each cell in order.

---

## Cell 1: Install PyTorch for GPU

```python
# Install PyTorch with CUDA 11.8 support
!pip install torch torchvision -q
print("✅ PyTorch installed")
```

---

## Cell 2: Verify GPU Setup

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected! Go to Runtime → Change runtime type → GPU")
```

---

## Cell 3: Mount Google Drive (Optional - for saving results)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✅ Drive mounted at /content/drive")
```

---

## Cell 4: Clone or Upload Project

### Option A: Clone from GitHub
```python
%cd /content
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
%cd tredence-analytics
!ls -la
```

### Option B: Upload ZIP (if not on GitHub)
```python
# Upload zip file manually in Colab
# Then unzip:
!unzip -q tredence-analytics.zip
%cd tredence-analytics
!ls -la
```

---

## Cell 5: Verify Project Setup

```python
import os
os.chdir('/content/tredence-analytics')  # Adjust path if needed

# Check files exist
required_files = ['train.py', 'model.py', 'prunable_linear.py', 'run_enhanced_experiments.py']
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    print(f"❌ Missing files: {missing}")
else:
    print("✅ All required files present")
    
# Check data
if os.path.exists('data/cifar-10-batches-py'):
    print("✅ CIFAR-10 data found")
else:
    print("⚠️  CIFAR-10 data not found - will download on first run")
```

---

## Cell 6: Test Single Training Run (Quick Validation)

```python
# This runs ONE configuration for 20 epochs to verify everything works
# Should take ~5-10 minutes

from train import train_with_early_stopping
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

print("\n🚀 Starting quick validation run...")
print("Config: λ=1e-4, dropout=0.5, epochs=20")

model, history = train_with_early_stopping(
    lambd=1e-4,
    dropout_rate=0.5,
    use_batch_norm=False,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    early_stopping_patience=10,
    validation_split=0.1
)

print(f"\n✅ Quick run complete!")
print(f"Best test accuracy: {max(history['test_accuracy']):.2f}%")
print(f"Total time: {history.get('total_time', 'N/A')} seconds")
```

---

## Cell 7: Full Hyperparameter Tuning (Main Run)

```python
# This runs all 8 configurations - takes ~1-2 hours on T4
# Good for overnight or Pro tier

print("🚀 Starting full hyperparameter tuning...")
print("This will test 8 configurations with 100 epochs each")
print("Estimated time: 1-2 hours (T4), 45 min (A100)")

!python3 run_enhanced_experiments.py

print("\n✅ Hyperparameter tuning complete!")
```

---

## Cell 8: Generate Visualizations

```python
# Creates comparison plots
print("📊 Generating visualizations...")

!python3 visualize_results.py

print("✅ Visualizations generated!")
print("\nGenerated files:")
import os
import glob

plots = glob.glob('results/hyperparams/*.png')
for plot in sorted(plots):
    print(f"  - {os.path.basename(plot)}")
```

---

## Cell 9: Display and Save Results

```python
import json
import pandas as pd

# Load results
results_file = 'results/hyperparams/results_summary.json'

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Display as table
    df = pd.DataFrame(results).T
    print("📊 HYPERPARAMETER TUNING RESULTS")
    print("=" * 100)
    print(df.to_string())
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]['best_test_accuracy'])
    print(f"\n🏆 Best configuration: {best[0]}")
    print(f"   Accuracy: {best[1]['best_test_accuracy']:.2f}%")
    print(f"   Sparsity: {best[1]['sparsity_pct']:.2f}%")
else:
    print("Results file not found. Run hyperparameter tuning first (Cell 7)")
```

---

## Cell 10: View Generated Plots

```python
# Display plots in Colab
from IPython.display import Image, display
import glob

plot_files = glob.glob('results/hyperparams/*.png')

for plot_file in sorted(plot_files):
    print(f"\n📊 {os.path.basename(plot_file)}")
    print("-" * 50)
    display(Image(plot_file))
```

---

## Cell 11: Save Results to Drive

```python
# Copy results to Google Drive for persistence
import shutil

source = '/content/tredence-analytics/results'
dest = '/content/drive/MyDrive/tredence-results'

# Create destination if needed
os.makedirs(dest, exist_ok=True)

# Copy
shutil.copytree(source, dest, dirs_exist_ok=True)
print(f"✅ Results saved to Google Drive: {dest}")

# List saved files
import subprocess
result = subprocess.run(['du', '-sh', dest], capture_output=True, text=True)
print(f"Total size: {result.stdout}")
```

---

## Cell 12: Download Results (Local Machine)

```python
# If not using Drive, download from Colab
import zipfile

# Create zip
with zipfile.ZipFile('tredence-results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('results/hyperparams'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'results')
            zipf.write(file_path, arcname)

print("✅ Results zipped. Download from Colab → Files → tredence-results.zip")
```

---

## Cell 13: Custom Training (Experimentation)

```python
# Use this to run custom configurations
from train import train_with_early_stopping

# Your custom config
custom_configs = [
    {'lambd': 1e-5, 'dropout_rate': 0.3, 'use_batch_norm': False, 'name': 'light_sparsity'},
    {'lambd': 1e-3, 'dropout_rate': 0.7, 'use_batch_norm': True, 'name': 'heavy_regularization'},
]

for config in custom_configs:
    print(f"\n🚀 Training: {config['name']}")
    name = config.pop('name')
    
    model, history = train_with_early_stopping(**config, epochs=50)
    
    acc = max(history['test_accuracy'])
    print(f"✅ {name}: {acc:.2f}% accuracy")
```

---

## ✅ Checklist

- [ ] Cell 1: Install PyTorch
- [ ] Cell 2: Verify GPU (should show T4, A100, or similar)
- [ ] Cell 3: Mount Drive (optional)
- [ ] Cell 4: Clone/Upload project
- [ ] Cell 5: Verify project setup
- [ ] Cell 6: Quick validation run (~10 min)
- [ ] Cell 7: Full training (~1-2 hours)
- [ ] Cell 8: Generate visualizations
- [ ] Cell 9: View results table
- [ ] Cell 10: View plots
- [ ] Cell 11: Save to Drive (optional)

---

## 🎯 Recommended Flow

**Option A: Quick Test (15 minutes)**
- Run Cells 1-6
- Verify GPU is working
- See if training completes

**Option B: Full Training (2-3 hours)**
- Run Cells 1-12
- Get all results and plots
- Save to Drive for analysis

**Option C: Custom Experiments (On-Demand)**
- Run Cells 1-5, then Cell 13
- Test your own configurations
- Iterate based on results

---

## 🚀 Pro Tips

1. **Save frequently**: Run Cell 11 periodically to backup to Drive
2. **Monitor GPU**: Add `!nvidia-smi` between runs to check memory
3. **Reduce batch size if OOM**: Change `batch_size=128` in Cell 6
4. **Longer session**: Use Colab Pro for 24-hour sessions
5. **Background tab**: Keep Colab tab active while training (use a timer script if needed)

---

## 📊 Expected Outputs

After complete run:
- ✅ 8 trained models in `results/hyperparams/`
- ✅ Training histories for each config
- ✅ `comparison_matrix.png` showing all configs
- ✅ 6-panel training curves per config
- ✅ `results_summary.json` with metrics
- ✅ Best config identified

---

## 💡 Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU | Runtime → Change runtime type → GPU |
| Out of memory | Reduce batch_size in train calls |
| Download slow | Use Drive mount (Cell 3) instead |
| Timeout (>12h) | Use Colab Pro or Lambda Labs |
| Missing files | Ensure full project uploaded/cloned |

Good luck! 🚀
