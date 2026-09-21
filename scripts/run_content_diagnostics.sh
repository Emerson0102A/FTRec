#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

run_diagnostic() {
  local experiment_config="$1"
  local model_config="$2"
  local run_root="$3"
  shift 3

  python -m ftrec.cli.diagnose_content \
    --config "$experiment_config" \
    --model-config "$model_config" \
    --checkpoint "$run_root/pretrain/joint_proportional/all-domains/seed-42/best.pt" \
    --output "$run_root/pretrain/joint_proportional/all-domains/seed-42/content-diagnostics.json" \
    "$@"
}

run_diagnostic \
  configs/experiment/attribute_llm2attr_pretrain.yaml \
  configs/model/sasrec_llm2attr.yaml \
  runs-attributes/llm2attr-dual \
  --tune-adaptive-fusion

run_diagnostic \
  configs/experiment/attribute_llm2attr_fused_pretrain.yaml \
  configs/model/sasrec_llm2attr_fused.yaml \
  runs-attributes/llm2attr-fused

run_diagnostic \
  configs/experiment/attribute_structured_title_pretrain.yaml \
  configs/model/sasrec_structured_title.yaml \
  runs-attributes/structured-title-dual \
  --tune-adaptive-fusion

run_diagnostic \
  configs/experiment/attribute_structured_title_fused_pretrain.yaml \
  configs/model/sasrec_structured_title_fused.yaml \
  runs-attributes/structured-title-fused

run_diagnostic \
  configs/experiment/joint_proportional.yaml \
  configs/model/sasrec.yaml \
  runs-lr1e-4
