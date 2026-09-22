#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"

action="${ACTION:-train}"
include_fullft="${INCLUDE_FULLFT:-0}"
extra_args=("$@")

if [[ "$action" != "train" && "$action" != "dry-run" ]]; then
  echo "ACTION must be train or dry-run" >&2
  exit 2
fi
if [[ "$include_fullft" != "0" && "$include_fullft" != "1" ]]; then
  echo "INCLUDE_FULLFT must be 0 or 1" >&2
  exit 2
fi

configs=(
  configs/experiment/attribute_structured_title_fused_target_only_content.yaml
  configs/experiment/attribute_structured_title_fused_target_only_lora.yaml
  configs/experiment/attribute_structured_title_fused_target_only_joint.yaml
)
if [[ "$include_fullft" == "1" ]]; then
  configs+=(
    configs/experiment/attribute_structured_title_fused_target_only_fullft.yaml
  )
fi

action_args=()
if [[ "$action" == "dry-run" ]]; then
  action_args+=(--dry-run)
fi

for config in "${configs[@]}"; do
  python -m ftrec.cli.adapt \
    --config "$config" \
    --model-config configs/model/sasrec_structured_title_fused.yaml \
    "${action_args[@]}" \
    "${extra_args[@]}"
done
