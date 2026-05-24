#!/usr/bin/env bash
set -Eeuo pipefail

# Run adversarial-only and paired robustness repair over the retained panels.
# Set REGENERATE_MISSING=1 to rerun eval-only predictions from existing checkpoints.

cd "${DATASET_ARTIFACTS_REPO:-$(pwd)}"

RESULT_DIRS=(
  results/run7_electra_small
  results/run5_6_bert_base
  results/run5_electra_base
  results/run5_roberta_base
  results/run6_deberta_v3_base
  results/run6_distilbert
  results/run6_albert_base
)

ARGS=()
for d in "${RESULT_DIRS[@]}"; do
  if [ -d "$d" ]; then
    ARGS+=(--results-dir "$d")
  else
    echo "[warn] missing result directory: $d" >&2
  fi
done

if [ "${#ARGS[@]}" -eq 0 ]; then
  echo "ERROR: no retained result directories were found" >&2
  exit 1
fi

if [ "${REGENERATE_MISSING:-0}" = "1" ]; then
  ARGS+=(--regenerate-missing)
fi

OUT_DIR="${OUT_DIR:-cartography_adversarial_repair_$(date -u +%Y%m%d_%H%M%S)}"

python3 scripts/repair_adversarial_metrics.py   "${ARGS[@]}"   --dataset-dir "${DATASET_DIR:-data/qa}"   --out-dir "$OUT_DIR"
