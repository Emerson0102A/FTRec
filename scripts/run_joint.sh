#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for seed in 42 43 44 45 46; do
  ftrec-pretrain --config configs/experiment/joint.yaml --seed "$seed" "$@"
done
