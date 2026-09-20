#!/usr/bin/env bash
set -euo pipefail

ROOT="${FTREC_ROOT:-/root/autodl-tmp/FTRec}"
LLM2ATTR_ROOT="${LLM2ATTR_ROOT:-${ROOT}/LLM2Attr}"
SOURCE_DATA="${MDSR_SOURCE_DIR:-${ROOT}/data/MDSR-Amazon}"
META_DATA="${AMAZON_META_DIR:-${ROOT}/Dataset}"
CATALOG="${ATTRIBUTE_CATALOG:-${ROOT}/data/attribute_experiment/catalog.jsonl.gz}"
MODE="${MODE:-smoke}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"
cd "${ROOT}"

python -m ftrec.attributes.runtime \
  --llm2attr-root "${LLM2ATTR_ROOT}" \
  --model "Qwen/Qwen2.5-0.5B"

if [[ ! -f "${CATALOG}" ]]; then
  python -m ftrec.attributes.catalog \
    --dataset-dir "${META_DATA}" \
    --mappings "${SOURCE_DATA}/mappings.pkl" \
    --output "${CATALOG}"
fi

case "${MODE}" in
  smoke)
    OUTPUT="${LLM2ATTR_OUTPUT:-${ROOT}/data/attribute_experiment/llm2attr-smoke}"
    LIMIT_ARGS=(--max-items 64)
    ;;
  full)
    OUTPUT="${LLM2ATTR_OUTPUT:-${ROOT}/data/attribute_experiment/llm2attr}"
    LIMIT_ARGS=()
    ;;
  *)
    echo "ERROR: MODE must be smoke or full, got: ${MODE}" >&2
    exit 2
    ;;
esac

python -m ftrec.attributes.llm2attr \
  --catalog "${CATALOG}" \
  --output "${OUTPUT}" \
  --llm2attr-root "${LLM2ATTR_ROOT}" \
  --model "Qwen/Qwen2.5-0.5B" \
  --device cuda \
  --torch-dtype bfloat16 \
  --attention sdpa \
  --batch-size "${BATCH_SIZE}" \
  "${LIMIT_ARGS[@]}"

echo "LLM2Attr artifact written to ${OUTPUT}"
