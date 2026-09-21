#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"

seed="${SEED:-42}"
action="${ACTION:-train}"
extra_args=("$@")

if [[ "$action" != "train" && "$action" != "diagnose" ]]; then
  echo "ACTION must be train or diagnose" >&2
  exit 2
fi

# hard_top1 is already available at runs-attributes/llm2attr-fused. These are
# the three new alternatives, all trained with the same data and protocol.
models=(
  "mean-all|configs/model/sasrec_llm2attr_fused_mean.yaml"
  "soft-attention|configs/model/sasrec_llm2attr_fused_soft_attention.yaml"
  "domain-title-attention|configs/model/sasrec_llm2attr_fused_domain_title_attention.yaml"
)

for specification in "${models[@]}"; do
  IFS='|' read -r model_name model_config <<<"$specification"
  run_dir="runs-attribute-pooling/$model_name/pretrain/joint_proportional/all-domains/seed-$seed"
  if [[ "$action" == "train" ]]; then
    python -m ftrec.cli.pretrain \
      --config configs/experiment/attribute_llm2attr_fused_pretrain.yaml \
      --model-config "$model_config" \
      --seed "$seed" \
      --output-dir "$run_dir" \
      "${extra_args[@]}"
    continue
  fi

  python -m ftrec.cli.diagnose_content \
    --config configs/experiment/attribute_llm2attr_fused_pretrain.yaml \
    --model-config "$model_config" \
    --checkpoint "$run_dir/best.pt" \
    --output "$run_dir/content-diagnostics.json" \
    "${extra_args[@]}"
done
