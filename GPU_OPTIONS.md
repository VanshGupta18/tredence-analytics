# GPU Training Options - Setup Guide

## 🚀 Quick Comparison

| Platform | Cost | GPU | Speed | Hours/Month | Best For |
|----------|------|-----|-------|------------|----------|
| **Google Colab Free** | Free | T4 | 25-30x | ~8-10 | Quick experiments |
| **Google Colab Pro** | $10/mo | T4/A100 | 25-100x | Unlimited | Regular training |
| **Kaggle Kernels** | Free | K80 | 15-20x | ~40 | Quick runs |
| **Lambda Labs** | $0.44/hr | A100 | 80-100x | Pay-as-go | Production |
| **AWS EC2** | $0.35-$2.48/hr | Various | 25-100x | Pay-as-go | Scalable |
| **Local GPU** | One-time | RTX3090+ | 30-50x | Unlimited | If you have one |

---

## 🥇 RECOMMENDED: Google Colab (Free T4)

### Why Colab?
✅ **Free** (with T4 GPU)
✅ **No setup** needed
✅ **Pre-installed** PyTorch, torchvision
✅ **Easy to use** (Jupyter notebook)
✅ **Sufficient for** hyperparameter tuning (~10 hours)

### Setup Steps

#### Step 1: Create Colab Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook** (or File → New notebook)
3. Name it: "Tredence Analytics - GPU Training"

#### Step 2: Enable GPU
1. Menu: **Runtime** → **Change runtime type**
2. Select **GPU** (should show T4)
3. Click **Save**

#### Step 3: Install and Setup
```python
# Cell 1: Install dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# Cell 2: Mount Google Drive (to save results)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Clone or upload project
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
# OR upload as ZIP and unzip

# Cell 4: Navigate to project
import os
os.chdir('/content/drive/MyDrive/tredence-analytics')  # or /content/tredence-analytics
```

#### Step 4: Run Training
```python
# Cell 5: Run enhanced experiments
!python3 run_enhanced_experiments.py

# Or individual components
from train import train_with_early_stopping
model, history = train_with_early_stopping(
    lambd=1e-4,
    dropout_rate=0.5,
    epochs=100
)
```

---

## 🥈 ALTERNATIVE: Google Colab Pro ($10/month)

### Advantages over Free Tier
✅ Access to **A100 GPU** (3-4x faster than T4)
✅ **More session time** (24 hour max vs 12 hour)
✅ **Higher memory** (40GB vs 12GB)
✅ **Faster TPU** access

### Cost Analysis
```
Monthly cost:        $10
Estimated training:  5-10 hours per run
Cost per training:   ~$0.50-$1.50
Value:               Saves 40+ hours of CPU time
```

### Setup: Same as free tier!

---

## 🥉 ALTERNATIVE: Kaggle Kernels (Free K80)

### Advantages
✅ **Completely free**
✅ **K80 GPU** (slower than T4 but still 15-20x faster)
✅ **40 hours/week** GPU time
✅ **Pre-loaded** with PyTorch

### Setup Steps

1. **Create Kaggle Account** at [kaggle.com](https://kaggle.com)
2. **New Notebook** (Kernel)
3. **Change GPU settings**: 
   - Accelerator: GPU (K80)
   - Internet: Enable
4. **Upload project** as dataset or use git

### Code Example
```python
# Mount Kaggle dataset
import os
os.chdir('/kaggle/input/your-dataset')

# Or clone from GitHub
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
os.chdir('tredence-analytics')

# Run training
!python3 run_enhanced_experiments.py
```

---

## 💻 ALTERNATIVE: Lambda Labs ($0.44/hr with A100)

### Best For
✅ Production runs
✅ Fastest execution
✅ No hour limits

### Setup
1. **Sign up** at [lambdalabs.com](https://lambdalabs.com)
2. **Launch instance** (A100)
3. **SSH into instance**
4. **Clone and run**:
```bash
git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
cd tredence-analytics
python3 run_enhanced_experiments.py
```

### Cost for Full Run
```
Lambda A100: $0.44/hour
Estimated training: 1-2 hours
Total cost: $0.50-$1.00
```

---

## 📊 Performance Comparison

### Training Time Estimates

```
Configuration: 8 hyperparams × 50 epochs = 400 total epochs

CPU (4-core):                ~24-48 hours
GPU T4 (Colab):              ~1-2 hours        (25-30x faster)
GPU A100 (Lambda):           ~30-45 minutes    (50-80x faster)
```

---

## ⚡ Modified Code for GPU

Your current code works on GPU without changes, but here's how to optimize:

### Use CUDA if Available
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Models automatically use GPU if available
model = SparsityAwareNet(dropout_rate=0.5)
model = model.to(device)
```

### Check GPU in Colab
```python
!nvidia-smi  # Shows GPU type and memory
```

### Batch Size Optimization
```python
# GPU can handle larger batches than CPU
# For T4 (15GB memory): batch_size = 256-512
# For A100 (40GB memory): batch_size = 512-1024

batch_size = 512  # Optimal for T4
```

---

## 🔄 Complete Colab Notebook Template

Create a new notebook and use this:

```python
# ====== SETUP ======
# Cell 1: Install
!pip install torch torchvision -q

# Cell 2: Check GPU
!nvidia-smi

# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# ====== PROJECT SETUP ======
# Cell 4: Clone or upload
%cd /content/drive/MyDrive/
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
%cd tredence-analytics

# ====== VERIFY SETUP ======
# Cell 5: Test imports
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# ====== RUN TRAINING ======
# Cell 6: Option A - Quick test (single config)
from train import train_with_early_stopping
model, history = train_with_early_stopping(
    lambd=1e-4,
    dropout_rate=0.5,
    epochs=20,  # Start with 20 to test
    batch_size=256
)

print(f"Best test accuracy: {history['test_accuracy'][-1]:.2f}%")

# Cell 7: Option B - Full hyperparameter tuning
!python3 run_enhanced_experiments.py

# ====== SAVE RESULTS ======
# Cell 8: Copy results to Drive
!cp -r results/hyperparams /content/drive/MyDrive/tredence-results/

print("✅ Training complete! Results saved to Drive.")
```

---

## 📋 Step-by-Step: Colab Free (Quickest Start)

### 1. Go to Colab
```
https://colab.research.google.com
```

### 2. Create New Notebook
Click "New notebook"

### 3. Enable GPU
Menu → Runtime → Change runtime type → GPU

### 4. Run This
```python
# Install PyTorch for GPU
!pip install torch torchvision -q

# Clone your project
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git

# Run
import os
os.chdir('tredence-analytics')
!python3 run_enhanced_experiments.py
```

**That's it!** Training will start on free T4 GPU.

---

## 🎯 Optimization Tips

### For Colab (Limited Time)
```python
# Run only critical configs, not all 8
configs_to_test = [
    ('baseline_lambda_1e-04', 1e-4, 0.0, False),
    ('dropout_0.5_lambda_1e-04', 1e-4, 0.5, False),
    ('batchnorm_dropout_0.5', 1e-4, 0.5, True),
]

# Reduce epochs for quick results
epochs = 50  # Instead of 100
```

### For Lambda Labs (Pay-as-You-Go)
```python
# Maximize GPU usage
batch_size = 1024  # A100 can handle this
epochs = 200
# Lower cost per epoch due to speed
```

### For Local GPU
```python
# If you have NVIDIA GPU locally:
# Install CUDA: https://developer.nvidia.com/cuda-downloads
# Then run: python3 run_enhanced_experiments.py
```

---

## 🚨 Common Issues & Fixes

### Issue: "No GPU Available"
```python
# Solution: Check runtime type
!nvidia-smi  # Should show GPU info
# If not, go to Runtime → Change runtime type → GPU
```

### Issue: "Out of Memory"
```python
# Reduce batch size in train.py
batch_size = 128  # Instead of 256
# Or use gradient accumulation
```

### Issue: "Module not found"
```python
# Install missing dependencies
!pip install -r requirements.txt
```

---

## 📊 Recommended: Free Colab → Pro Colab

### Phase 1: Test (Free Colab)
- Quick validation run: 1 config, 20 epochs
- Estimated: 10 minutes
- Cost: $0

### Phase 2: Full Training (Pro Colab if needed)
- All 8 configs, 100 epochs each
- Estimated: 10 hours
- Cost: $10 one-time or pay-as-you-go

### Phase 3: CNN Extension (Lambda Labs)
- Faster CNN implementation
- Estimated: 1 hour on A100
- Cost: $0.50-$1.00

---

## 📞 Quick Decision

**Choose based on your situation:**

✅ **Just want to try**: Use **Free Colab T4** (takes 1-2 hours, results next day)

✅ **Want faster results**: Get **Colab Pro** ($10/month, A100 GPU)

✅ **Production/many runs**: Use **Lambda Labs** ($0.50 per run)

✅ **Have local GPU**: Run **locally** (no cost, unlimited)

All options require **NO code changes** - your existing scripts work as-is!

---

## 🎓 Expected Runtimes

```
Setup + Full Training + Results (8 configs, 100 epochs):

Free Colab T4:       ~2 hours
Colab Pro A100:      ~45 minutes
Lambda A100:         ~30 minutes  
Local RTX 3090:      ~20 minutes
CPU (4-core):        ~36 hours
```

**Recommendation**: Start with **Free Colab** to validate, then use **Colab Pro** or **Lambda** for full runs.
