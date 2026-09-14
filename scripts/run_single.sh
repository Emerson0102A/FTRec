#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for seed in 42 43 44; do
  for domain in 0 1 2 3 4; do
    ftrec-pretrain --config configs/experiment/single.yaml --seed "$seed" --domain "$domain" "$@"
  done
done
