"""Diagnose why torch.cuda.is_available() returns False on the compute node."""
import os
import ctypes
import sys

print("PY:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

# Try to load the CUDA driver lib directly
for name in ["libcuda.so.1", "libcuda.so", "libcudart.so.12"]:
    try:
        h = ctypes.CDLL(name)
        print(f"loaded: {name}")
    except Exception as e:
        print(f"FAILED to load {name}: {e}")

# torch internal nvml init diagnostics
try:
    if hasattr(torch.cuda, "_initialized"):
        print("torch.cuda._initialized:", torch.cuda._initialized)
    print("Trying torch.cuda.init()...")
    torch.cuda.init()
    print("torch.cuda.init() ok, count now:", torch.cuda.device_count())
except Exception as e:
    print("init error:", repr(e))
