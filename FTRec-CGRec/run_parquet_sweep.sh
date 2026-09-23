#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PARQUET_DIR [RUN_DIR] [extra train_parquet.py flags...]" >&2
    exit 2
fi

parquet_dir=$1
shift
run_dir=runs/cgrec-parquet
if [[ $# -gt 0 && $1 != --* ]]; then
    run_dir=$1
    shift
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for domain in 0 1 2 3 4; do
    for seed in 42 43 44 45 46; do
        result="${run_dir}/domain-${domain}/seed-${seed}/results.json"
        if [[ -s "${result}" ]]; then
            echo "Keeping completed target domain ${domain}, seed ${seed}: ${result}" >&2
            continue
        fi
        echo "Running target domain ${domain}, seed ${seed}" >&2
        python "${script_dir}/train_parquet.py" \
            --parquet_dir "${parquet_dir}" \
            --run_dir "${run_dir}" \
            --target_domain "${domain}" \
            --seed "${seed}" \
            "$@"
    done
done
python "${script_dir}/summarize_parquet.py" --run_dir "${run_dir}"
