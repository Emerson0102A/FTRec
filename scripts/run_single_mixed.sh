#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

seed=42
extra_args=()
while (($#)); do
  case "$1" in
    --seed)
      if (($# < 2)); then
        echo "--seed requires a value" >&2
        exit 2
      fi
      seed="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

for domain in 0 1 2 3 4; do
  ftrec-pretrain \
    --config configs/experiment/single_mixed.yaml \
    --seed "$seed" \
    --domain "$domain" \
    "${extra_args[@]}"
done
