#!/usr/bin/env bash
set -euo pipefail

ROOT="${FTREC_ROOT:-/root/autodl-tmp/FTRec}"
SOURCE_DATA="${MDSR_SOURCE_DIR:-${ROOT}/data/MDSR-Amazon}"
META_DATA="${AMAZON_META_DIR:-${ROOT}/Dataset}"
CATALOG="${ATTRIBUTE_CATALOG:-${ROOT}/data/attribute_experiment/catalog.jsonl.gz}"
MODE="${MODE:-smoke}"
EVIDENCE="${EVIDENCE:-title}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"
cd "${ROOT}"

if [[ ! -f "${CATALOG}" ]]; then
  python -m ftrec.attributes.catalog \
    --dataset-dir "${META_DATA}" \
    --mappings "${SOURCE_DATA}/mappings.pkl" \
    --output "${CATALOG}"
fi

case "${EVIDENCE}" in
  title)
    INPUT_FIELDS="title"
    OUTPUT_NAME="structured-title"
    ;;
  enhanced)
    INPUT_FIELDS="title,main_category,categories,features,description,details,store"
    OUTPUT_NAME="structured-enhanced"
    ;;
  *)
    echo "ERROR: EVIDENCE must be title or enhanced, got: ${EVIDENCE}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  smoke)
    OUTPUT="${STRUCTURED_OUTPUT:-${ROOT}/data/attribute_experiment/${OUTPUT_NAME}-smoke}"
    LIMIT_ARGS=(--max-items 64)
    ;;
  full)
    OUTPUT="${STRUCTURED_OUTPUT:-${ROOT}/data/attribute_experiment/${OUTPUT_NAME}}"
    LIMIT_ARGS=()
    ;;
  *)
    echo "ERROR: MODE must be smoke or full, got: ${MODE}" >&2
    exit 2
    ;;
esac

python -m ftrec.attributes.structured \
  --catalog "${CATALOG}" \
  --output "${OUTPUT}" \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --input-fields "${INPUT_FIELDS}" \
  --device cuda \
  --torch-dtype bfloat16 \
  --attention sdpa \
  --batch-size "${BATCH_SIZE}" \
  "${LIMIT_ARGS[@]}"

echo "Structured-LLM artifact written to ${OUTPUT}"
