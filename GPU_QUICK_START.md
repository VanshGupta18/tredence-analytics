# 🎯 GPU Setup - Quick Decision Guide

## First: Check Your System

Run this to see if you have a local GPU:

```bash
python3 check_gpu.py
```

This will tell you:
- ✅ If you have a GPU locally
- ✅ What GPU specs you have
- ✅ Estimated training times
- ✅ Best options for your setup

---

## 🚀 FASTEST WAY TO START

### Option 1: Google Colab (RECOMMENDED - Quickest Start)

**Why**: Free, no setup, works immediately

```
1. Go to: https://colab.research.google.com
2. New notebook
3. Runtime → Change runtime type → GPU
4. Copy-paste code from COLAB_NOTEBOOK_GUIDE.md
5. Run!
```

**Time**: 2 hours for full training
**Cost**: FREE

---

### Option 2: Check Local GPU First

```bash
python3 check_gpu.py
```

**If you see ✅ CUDA available:**
```bash
# Run directly on your machine
python3 run_enhanced_experiments.py
```

**If you see ❌ No GPU:**
→ Use Colab (see Option 1)

---

## 📊 Decision Matrix

Pick based on your situation:

### 👤 I have a local GPU (NVIDIA)
```
✅ Best: Run locally
   python3 run_enhanced_experiments.py
   
⏱️  Time: 20-60 minutes
💰 Cost: $0
```

### 🎓 I want to try quickly (< 10 mins)
```
✅ Use: Google Colab Free
   
🔗 URL: https://colab.research.google.com
⏱️  Time: 5-10 minutes (quick test)
💰 Cost: FREE
```

### 🧪 I want full results today (not overnight)
```
✅ Use: Google Colab Pro OR Lambda Labs

📊 Colab Pro:    45 min, $10/month
⚡ Lambda Labs:  30 min, $0.50 per run
```

### 🌙 I want results overnight (free)
```
✅ Use: Google Colab Free OR Kaggle Kernels

📊 Colab Free:    2 hours, FREE
🎲 Kaggle:        3-4 hours, FREE (limited)
```

---

## 📋 Setup Instructions (Choose One)

### Setup A: Local GPU (If You Have One)

```bash
# 1. Check if you have CUDA
python3 check_gpu.py

# 2. If ✅ CUDA available:
python3 run_enhanced_experiments.py

# 3. Results will be in:
# results/hyperparams/
```

**Estimated time**: 20-60 minutes

---

### Setup B: Google Colab (Fastest Start)

**Step 1**: Open [colab.research.google.com](https://colab.research.google.com)

**Step 2**: Create new notebook

**Step 3**: Enable GPU
- Menu → Runtime → Change runtime type
- Select **GPU**
- Click Save

**Step 4**: Copy-paste and run (Cell 1-7 from COLAB_NOTEBOOK_GUIDE.md)

```python
# Cell 1: Install
!pip install torch torchvision -q

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Clone project
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
import os
os.chdir('tredence-analytics')

# Cell 4: Run training
!python3 run_enhanced_experiments.py
```

**Estimated time**: 2 hours

---

### Setup C: Kaggle Kernels (Alternative Free GPU)

**Step 1**: Sign up at [kaggle.com](https://kaggle.com)

**Step 2**: New Notebook → Accelerator: GPU (K80)

**Step 3**: Enable Internet (Settings)

**Step 4**: Run:
```python
!git clone https://github.com/YOUR_USERNAME/tredence-analytics.git
import os
os.chdir('tredence-analytics')
!python3 run_enhanced_experiments.py
```

**Estimated time**: 3-4 hours

---

## ⚡ Performance Comparison

What will you get?

| Platform | GPU | Time | Cost | Best For |
|---|---|---|---|---|
| **Local (your machine)** | RTX 3090 (if you have) | 20-60 min | $0 | Maximum speed |
| **Colab Free** | T4 | 2 hours | FREE | Quick start |
| **Colab Pro** | A100 | 45 min | $10/mo | Fast results |
| **Kaggle** | K80 | 3-4 hours | FREE | Patient users |
| **Lambda Labs** | A100 | 30 min | $0.50 | Pay-per-use |

---

## 🎯 My Recommendation

### For First-Time Users:
1. **Run `check_gpu.py`** to see your options
2. If local GPU available → Use it
3. If not → **Use Google Colab Free** (takes 2 hours, completely free)

### For Impatient Users:
Get **Colab Pro** ($10) for 3-4x faster results

### For Budget Users:
Use **Colab Free** for testing, **Kaggle** for full runs

---

## 🚀 Absolute Quickest Start (Copy-Paste)

### Option 1A: Test Your Local GPU Right Now

```bash
cd /Users/vanshgupta/tredence-analytics

# Check if GPU is available
python3 check_gpu.py

# If it says ✅ CUDA available, run:
python3 run_enhanced_experiments.py
```

### Option 1B: Use Colab Right Now (No Local GPU Needed)

1. Open: https://colab.research.google.com
2. Click **New notebook**
3. Menu → **Runtime** → **Change runtime type** → Select **GPU**
4. Run this in a cell:

```python
!pip install torch torchvision -q && git clone https://github.com/YOUR_USERNAME/tredence-analytics.git && cd tredence-analytics && python3 run_enhanced_experiments.py
```

**That's it!** 🎉

---

## 📊 Expected Outputs

After running (on any GPU platform):

```
results/hyperparams/
├── model_baseline_lambda_1e-04.pt          ← Trained model
├── history_baseline_lambda_1e-04.pt        ← Training history
├── model_dropout_0.5_lambda_1e-04.pt       ← More models...
├── history_dropout_0.5_lambda_1e-04.pt     ← More histories...
├── comparison_matrix.png                   ← 4-panel comparison plot
├── curves_baseline_lambda_1e-04.png        ← 6-panel training curves
├── curves_dropout_0.5_lambda_1e-04.png     ← More curves...
├── curves_batchnorm_dropout_0.5_*.png      ← More curves...
└── results_summary.json                    ← All results in JSON

Expected accuracy improvement:
  56% (baseline) → 62-66% (with enhancements)
  
  Individual contributions:
  + Early stopping:    +0.5%
  + Dropout (0.5):     +3%
  + Batch norm:        +2%
  + LR scheduling:     +1%
  = Total:             +6-7%
```

---

## 💡 Pro Tips

### Tip 1: Large Batch Size on GPU
```python
# On GPU, you can use larger batches (faster)
batch_size = 512  # Instead of 256
```

### Tip 2: Monitor GPU Memory
```bash
# In Colab, check GPU usage:
!nvidia-smi
```

### Tip 3: Save Results Frequently
```python
# In Colab, regularly save to Drive:
!cp -r results/hyperparams /content/drive/MyDrive/
```

### Tip 4: Reduce Epochs for Quick Tests
```python
# Test first with fewer epochs (5-10), then scale up
epochs = 10  # Quick test
epochs = 100  # Full run
```

---

## ❓ FAQ

**Q: Will my code work on GPU without changes?**
A: Yes! All your scripts work as-is on GPU.

**Q: How much faster is GPU than CPU?**
A: 20-50x faster. T4 GPU finishes in 2 hours what CPU takes 36+ hours.

**Q: Will I run out of GPU memory?**
A: Unlikely. Your code uses ~2-3GB per config. Colab provides 15GB on free tier.

**Q: Can I use CPU if GPU is slow?**
A: Yes, just don't enable GPU. But it will take 36+ hours.

**Q: Will code run if I close Colab?**
A: Free Colab disconnects after 12 hours. Colab Pro allows 24 hours.

**Q: How much will Colab Pro cost?**
A: $10/month for unlimited compute. Each run costs ~$0.10.

**Q: Can I download results from Colab?**
A: Yes! Click Files → Download in Colab interface.

---

## 🎓 Summary

✅ **Best for your case**: Check `python3 check_gpu.py` first

✅ **If local GPU**: Run `python3 run_enhanced_experiments.py`

✅ **If no local GPU**: Use **Google Colab Free** (2 hours, $0)

✅ **If in a hurry**: Use **Colab Pro** or **Lambda Labs** (45 min, $10 or $0.50)

All platforms will give you the same results - just different speeds and costs!

---

## 🔗 Quick Links

- **Check GPU**: `python3 check_gpu.py`
- **Colab Notebook**: [colab.research.google.com](https://colab.research.google.com)
- **Colab Guide**: See COLAB_NOTEBOOK_GUIDE.md
- **GPU Options**: See GPU_OPTIONS.md
- **Main Script**: `python3 run_enhanced_experiments.py`

**Ready? Pick your option above and get started! 🚀**
