#!/usr/bin/env bash
set -euo pipefail

ROOT="${FTREC_ROOT:-/root/autodl-tmp/FTRec}"
LLM2ATTR_ROOT="${LLM2ATTR_ROOT:-${ROOT}/LLM2Attr}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
python -m ftrec.attributes.runtime \
  --llm2attr-root "${LLM2ATTR_ROOT}" \
  --model "Qwen/Qwen2.5-0.5B"
