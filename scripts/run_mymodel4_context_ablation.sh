#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"

seed="${SEED:-42}"
extra_args=("$@")

# Both arms use the same model, target cohort, split, candidates, seed, and
# optimizer. Only whether the shared lower block sees domain-only or mixed
# history changes.
experiments=(
  "single-domain|joint_domain"
  "mixed-domain|joint_mixed_matched"
)

for specification in "${experiments[@]}"; do
  IFS='|' read -r context_name method <<<"$specification"
  run_dir="runs-mymodel4-context-ablation/$context_name/seed-$seed"
  python -m ftrec.cli.pretrain \
    --config configs/experiment/mymodel4_context_ablation.yaml \
    --model-config configs/model/sasrec_llm2attr_mymodel4_behavior.yaml \
    --method "$method" \
    --seed "$seed" \
    --output-dir "$run_dir" \
    "${extra_args[@]}"
done
