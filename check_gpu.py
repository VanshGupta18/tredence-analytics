#!/usr/bin/env python3
"""
GPU Detection & Configuration Script
Run this to check what GPU options are available on your system
"""

import sys
import subprocess

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print(f"\n{'='*60}")
        print("🔍 PYTORCH GPU CHECK")
        print(f"{'='*60}")
        
        print(f"\n📦 PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1e9
                print(f"\n   GPU {i}: {props.name}")
                print(f"   - Memory: {total_memory:.1f} GB")
                print(f"   - Compute Capability: {props.major}.{props.minor}")
                print(f"   - Multi-Processor Count: {props.multi_processor_count}")
                
                # Benchmark
                try:
                    x = torch.randn(10000, 10000, device=f'cuda:{i}')
                    y = torch.randn(10000, 10000, device=f'cuda:{i}')
                    
                    import time
                    start = time.time()
                    z = torch.mm(x, y)
                    end = time.time()
                    
                    tflops = (2 * 10000 * 10000 * 10000) / ((end - start) * 1e12)
                    print(f"   - Performance: ~{tflops:.1f} TFLOPS")
                except Exception as e:
                    print(f"   - Performance: (benchmark failed)")
            
            return True
        else:
            print("\n❌ CUDA is NOT available")
            print("\nTo enable GPU:")
            print("  1. On macOS: No CUDA support (use Metal acceleration)")
            print("  2. On Windows/Linux: Install NVIDIA CUDA toolkit")
            print("     - https://developer.nvidia.com/cuda-downloads")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nInstall with: pip install torch torchvision")
        return False

def check_cpu():
    """Check CPU specs"""
    print(f"\n{'='*60}")
    print("💻 CPU INFO")
    print(f"{'='*60}\n")
    
    try:
        import psutil
        import multiprocessing
        
        print(f"Cores: {multiprocessing.cpu_count()}")
        print(f"CPU Freq: {psutil.cpu_freq().current:.1f} MHz")
        
        # Estimate training time
        cpu_cores = multiprocessing.cpu_count()
        estimated_hours = 36 / (cpu_cores / 4)  # Rough estimate
        print(f"\n⏱️  Estimated training time (CPU):")
        print(f"   ~{estimated_hours:.0f}-{estimated_hours*2:.0f} hours for full run")
    except:
        print("(psutil not installed for detailed CPU info)")

def check_tensorflow():
    """Check TensorFlow GPU support (alternative)"""
    try:
        import tensorflow as tf
        print(f"\n{'='*60}")
        print("🔍 TENSORFLOW GPU CHECK")
        print(f"{'='*60}\n")
        
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
        
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
    except ImportError:
        pass

def recommendations():
    """Provide recommendations"""
    import torch
    
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    if torch.cuda.is_available():
        print("✅ You have a GPU! Options:")
        print("   1. ⭐ Run locally (fastest, no cost)")
        print("      Command: python3 run_enhanced_experiments.py")
        print("\n   2. Consider Colab Pro for backup sessions")
        print("      URL: https://colab.research.google.com")
        
    else:
        print("❌ No GPU detected. Options:")
        print("\n   1. ⭐ Google Colab (Free T4 GPU)")
        print("      - No setup needed")
        print("      - URL: https://colab.research.google.com")
        print("      - Time: ~2 hours for full run")
        print("      - Cost: FREE")
        
        print("\n   2. Google Colab Pro ($10/month)")
        print("      - Access to A100 GPU (3-4x faster)")
        print("      - Time: ~45 minutes for full run")
        print("      - Cost: $10/month")
        
        print("\n   3. Kaggle Kernels (Free, K80)")
        print("      - Time: ~3-4 hours for full run")
        print("      - Cost: FREE (40 hrs/week limit)")
        print("      - URL: https://kaggle.com")
        
        print("\n   4. Lambda Labs ($0.44/hr, A100)")
        print("      - Time: ~30 minutes for full run")
        print("      - Cost: ~$0.50 per run")
        print("      - URL: https://lambdalabs.com")
        
        print("\n   5. Local GPU (Best if available)")
        print("      - Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("      - Then run: python3 run_enhanced_experiments.py")

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  🚀 GPU CONFIGURATION CHECKER".ljust(59) + "║")
    print("║  Checking your system for GPU support...".ljust(59) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check CPU
    check_cpu()
    
    # Check TensorFlow
    check_tensorflow()
    
    # Recommendations
    recommendations()
    
    print(f"\n{'='*60}\n")
    
    return 0 if has_gpu else 1

if __name__ == "__main__":
    sys.exit(main())
