#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"

seed="${SEED:-42}"
model_set="${MODEL_SET:-core}"
action="${ACTION:-train}"
extra_args=("$@")

models=(
  "id|configs/model/sasrec.yaml"
  "llm2attr-dual|configs/model/sasrec_llm2attr.yaml"
  "structured-title-fused|configs/model/sasrec_structured_title_fused.yaml"
)
if [[ "$model_set" == "all" ]]; then
  models+=(
    "llm2attr-fused|configs/model/sasrec_llm2attr_fused.yaml"
    "structured-title-dual|configs/model/sasrec_structured_title.yaml"
  )
elif [[ "$model_set" != "core" ]]; then
  echo "MODEL_SET must be core or all" >&2
  exit 2
fi
if [[ "$action" != "train" && "$action" != "diagnose" ]]; then
  echo "ACTION must be train or diagnose" >&2
  exit 2
fi

for method in joint_domain joint_mixed_matched; do
  for specification in "${models[@]}"; do
    IFS='|' read -r model_name model_config <<<"$specification"
    run_dir="runs-sequence-ablation/$model_name/pretrain/$method/all-domains/seed-$seed"
    if [[ "$action" == "train" ]]; then
      python -m ftrec.cli.pretrain \
        --config configs/experiment/attribute_context_ablation.yaml \
        --model-config "$model_config" \
        --method "$method" \
        --seed "$seed" \
        --output-dir "$run_dir" \
        "${extra_args[@]}"
      continue
    fi

    tuning_args=()
    if [[ "$model_name" == *-dual ]]; then
      tuning_args+=(--tune-adaptive-fusion)
    fi
    python -m ftrec.cli.diagnose_content \
      --config configs/experiment/attribute_context_ablation.yaml \
      --model-config "$model_config" \
      --method "$method" \
      --checkpoint "$run_dir/best.pt" \
      --output "$run_dir/content-diagnostics.json" \
      "${tuning_args[@]}" \
      "${extra_args[@]}"
  done
done
