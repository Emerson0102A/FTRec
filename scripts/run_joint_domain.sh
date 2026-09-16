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

ftrec-pretrain \
  --config configs/experiment/joint_domain.yaml \
  --seed "$seed" \
  "${extra_args[@]}"
