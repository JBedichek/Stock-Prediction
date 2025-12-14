#!/usr/bin/env python3
"""
Debug script to check Triton/CUDA compilation setup.
Run this to diagnose torch.compile max-autotune issues.
"""

import subprocess
import os
import sys

print("=" * 80)
print("TRITON COMPILATION DIAGNOSTICS")
print("=" * 80)

# Check gcc
print("\n1. Checking gcc...")
try:
    result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
    print(f"✅ gcc found: {result.stdout.splitlines()[0]}")
except FileNotFoundError:
    print("❌ gcc not found! Install with: sudo apt install build-essential")
    sys.exit(1)

# Check CUDA
print("\n2. Checking CUDA...")
cuda_lib_paths = [
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
]

libcuda_found = False
for path in cuda_lib_paths:
    libcuda = os.path.join(path, "libcuda.so")
    if os.path.exists(libcuda):
        print(f"✅ libcuda.so found at: {path}")
        libcuda_found = True
        break

if not libcuda_found:
    print("❌ libcuda.so not found!")
    print("   This is needed for Triton compilation.")
    print("   Install NVIDIA driver with: sudo apt install nvidia-driver-XXX")

# Check nvidia-smi
print("\n3. Checking NVIDIA driver...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ NVIDIA driver installed")
        # Extract driver version
        for line in result.stdout.splitlines():
            if "Driver Version:" in line:
                print(f"   {line.strip()}")
    else:
        print("❌ nvidia-smi failed")
except FileNotFoundError:
    print("❌ nvidia-smi not found - NVIDIA driver may not be installed")

# Check PyTorch CUDA
print("\n4. Checking PyTorch CUDA...")
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("   ⚠️  CUDA not available in PyTorch")

# Check Triton
print("\n5. Checking Triton...")
try:
    import triton
    print(f"✅ Triton version: {triton.__version__}")
except ImportError:
    print("❌ Triton not installed")
    print("   Install with: pip install triton")

# Try a simple compile test
print("\n6. Testing torch.compile...")
try:
    import torch.nn as nn

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().cuda()

    # Try default mode (no Triton)
    print("   Testing mode='default'...")
    compiled_default = torch.compile(model, mode='default')
    x = torch.randn(2, 10).cuda()
    out = compiled_default(x)
    print("   ✅ mode='default' works")

    # Try reduce-overhead (no Triton)
    print("   Testing mode='reduce-overhead'...")
    compiled_reduce = torch.compile(model, mode='reduce-overhead')
    out = compiled_reduce(x)
    print("   ✅ mode='reduce-overhead' works")

    # Try max-autotune (uses Triton)
    print("   Testing mode='max-autotune'...")
    compiled_max = torch.compile(model, mode='max-autotune')
    out = compiled_max(x)
    print("   ✅ mode='max-autotune' works!")

except Exception as e:
    print(f"   ❌ Compilation failed: {e}")
    print("\n   To see full error, set: TORCHDYNAMO_VERBOSE=1")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if not libcuda_found:
    print("\n❌ CRITICAL: libcuda.so not found")
    print("   Fix: sudo apt install nvidia-driver-XXX")
    print("   Then reboot")
else:
    print("\n✅ All checks passed!")
    print("   If max-autotune still fails, try:")
    print("   1. Set LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
    print("   2. Or use mode='reduce-overhead' instead (still fast!)")

print()
