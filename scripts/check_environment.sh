#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "NVIDIA driver-visible GPU information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Repository disk space:"
df -h "$ROOT"

python - <<'PY'
import platform
import sys

import pyarrow
import torch

expected = "2.13.0+cu130"
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Torch CUDA runtime: {torch.version.cuda}")
print(f"PyArrow: {pyarrow.__version__}")
if torch.__version__ != expected:
    raise SystemExit(f"ERROR: expected torch {expected}, got {torch.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: CUDA GPU is not visible to PyTorch")
device = torch.device("cuda:0")
properties = torch.cuda.get_device_properties(device)
print(f"GPU: {properties.name}")
print(f"Compute capability: {properties.major}.{properties.minor}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: BF16 is not supported")
x = torch.tensor([2.0], device=device)
y = (x.square() + 1).item()
torch.cuda.synchronize()
if y != 5.0:
    raise SystemExit("ERROR: CUDA tensor smoke calculation failed")
print("CUDA tensor smoke: PASS")
PY
