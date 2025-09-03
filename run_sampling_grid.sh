#!/usr/bin/env bash
set -euo pipefail

INPUT="datasets/wildchat_private/full.json"
OUT_DIR="data"
LOG_DIR="res/sampling_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

vals=(10 20 50 100 200 500)

for minc in "${vals[@]}"; do
  for maxc in "${vals[@]}"; do
    if (( minc >= maxc )); then
      continue
    fi

    out_path="${OUT_DIR}/wildchat_internal_aug1_sample.min${minc}_max${maxc}.json"
    log_path="${LOG_DIR}/sample_datasets.min${minc}_max${maxc}.log"

    echo "[$(date -Is)] Running min=${minc}, max=${maxc} → ${out_path}"
    python -u src/scripts/sample_datasets.py \
      --input_path "${INPUT}" \
      --output_path "${out_path}" \
      --temporal_sample_size 5000 \
      --n_users 500 \
      --conversations_per_user 40 \
      --min_conversations "${minc}" \
      --max_conversations "${maxc}" \
      --seed 42 > "${log_path}" 2>&1
  done
done

echo "[$(date -Is)] Done."
