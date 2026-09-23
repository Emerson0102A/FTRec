#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

action="${ACTION:-train}"
phase="${PHASE:-all}"
arms="${ARMS:-id semantic}"
seeds="${SEEDS:-42}"
domains="${DOMAINS:-0 1 2 3 4}"
read -r -a snapshot_epochs <<< "${SNAPSHOT_EPOCHS:-5 10 20 30 40 50 75 100 150 200 250 300}"
pretrain_epochs="${PRETRAIN_EPOCHS:-300}"
ft_epochs="${FT_EPOCHS:-10}"
root="${RUN_ROOT:-runs-attributes/checkpoint-transfer}"

case "$action" in train|plan) ;; *) echo "ACTION must be train or plan" >&2; exit 2 ;; esac
case "$phase" in all|pretrain|adapt) ;; *) echo "PHASE must be all, pretrain or adapt" >&2; exit 2 ;; esac
[[ ${#snapshot_epochs[@]} -gt 0 ]] || { echo "SNAPSHOT_EPOCHS must not be empty" >&2; exit 2; }
[[ "$pretrain_epochs" =~ ^[0-9]+$ && "$ft_epochs" =~ ^[0-9]+$ ]] || {
  echo "PRETRAIN_EPOCHS and FT_EPOCHS must be positive integers" >&2; exit 2;
}
(( pretrain_epochs > 0 && ft_epochs > 0 )) || {
  echo "PRETRAIN_EPOCHS and FT_EPOCHS must be positive integers" >&2; exit 2;
}

run() {
  if [[ "$action" == plan ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for arm in $arms; do
  case "$arm" in
    id) model_config=configs/model/sasrec.yaml ;;
    semantic) model_config=configs/model/sasrec_structured_title_fused.yaml ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
  for seed in $seeds; do
    base="$root/$arm/pretrain/joint_proportional/all-domains/seed-$seed"
    if [[ "$phase" == all || "$phase" == pretrain ]]; then
      run python -m ftrec.cli.pretrain \
        --config configs/experiment/attribute_structured_title_fused_pretrain.yaml \
        --model-config "$model_config" --seed "$seed" \
        --output-dir "$base" --epochs "$pretrain_epochs" \
        --patience "$((pretrain_epochs + 1))" \
        --snapshot-epochs "${snapshot_epochs[@]}"
    fi
    if [[ "$phase" == all || "$phase" == adapt ]]; then
      for epoch in "${snapshot_epochs[@]}"; do
        snapshot=$(printf '%s/snapshots/epoch-%04d.pt' "$base" "$epoch")
        if [[ "$action" == train && ! -f "$snapshot" ]]; then
          echo "missing pretrain snapshot: $snapshot" >&2
          exit 1
        fi
        for domain in $domains; do
          run python -m ftrec.cli.adapt \
            --config configs/experiment/attribute_structured_title_fused_target_only_fullft.yaml \
            --model-config "$model_config" --method fullft \
            --pretrain-method joint_proportional --domain "$domain" --seed "$seed" \
            --base-checkpoint "$snapshot" \
            --output-dir "$root/$arm/adapt/$(printf 'epoch-%04d' "$epoch")/domain-$domain/seed-$seed" \
            --epochs "$ft_epochs" --patience "$((ft_epochs + 1))"
        done
      done
    fi
  done
done
