#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

action="${ACTION:-train}"
phase="${PHASE:-all}"
seed="${SEED:-42}"
arms="${ARMS:-id random shuffled attribute semantic}"
domains="${DOMAINS:-0 1 2 3 4}"
run_root="${RUN_ROOT:-runs-attributes/semantic-ft-ablation}"

case "$action" in train|dry-run|plan) ;; *) echo "ACTION must be train, dry-run or plan" >&2; exit 2 ;; esac
case "$phase" in all|pretrain|adapt) ;; *) echo "PHASE must be all, pretrain or adapt" >&2; exit 2 ;; esac

run() {
  if [[ "$action" == plan ]]; then
    printf '%q ' "$@"
    printf '\n'
  elif [[ "$action" == dry-run ]]; then
    "$@" --dry-run
  else
    "$@"
  fi
}

for arm in $arms; do
  case "$arm" in
    id) model_config=configs/model/sasrec.yaml ;;
    random) model_config=configs/model/sasrec_structured_fused_random.yaml ;;
    shuffled) model_config=configs/model/sasrec_structured_fused_shuffled.yaml ;;
    attribute) model_config=configs/model/sasrec_structured_fused_attribute_only.yaml ;;
    semantic) model_config=configs/model/sasrec_structured_title_fused.yaml ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
  base="$run_root/$arm/pretrain/joint_proportional/all-domains/seed-$seed/best.pt"
  if [[ "$phase" == all || "$phase" == pretrain ]]; then
    run python -m ftrec.cli.pretrain \
      --config configs/experiment/attribute_structured_title_fused_pretrain.yaml \
      --model-config "$model_config" --seed "$seed" \
      --output-dir "${base%/*}"
  fi
  if [[ "$phase" == all || "$phase" == adapt ]]; then
    for domain in $domains; do
      run python -m ftrec.cli.adapt \
        --config configs/experiment/attribute_structured_title_fused_target_only_lora.yaml \
        --model-config "$model_config" --pretrain-method joint_proportional \
        --domain "$domain" --rank 5 --seed "$seed" \
        --base-checkpoint "$base" \
        --output-dir "$run_root/$arm/adapt/lora_all/joint_proportional/domain-$domain/rank-5/seed-$seed"
    done
  fi
done
