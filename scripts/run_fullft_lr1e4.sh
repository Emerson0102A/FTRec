#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec ftrec-adapt --config configs/experiment/fullft_lr1e4.yaml "$@"
