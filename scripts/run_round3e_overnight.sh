#!/usr/bin/env bash
set -euo pipefail

ELECTRA_CONFIG=${ELECTRA_CONFIG:-configs/panel.round3e_electra.yaml}
BERT_CONFIG=${BERT_CONFIG:-configs/panel.round3e_bert.yaml}
ELECTRA_GPU_IDS=${ELECTRA_GPU_IDS:-0,1,2,3}
BERT_GPU_IDS=${BERT_GPU_IDS:-4,5,6,7}
ELECTRA_WORKERS=${ELECTRA_WORKERS:-4}
BERT_WORKERS=${BERT_WORKERS:-4}
LOG_DIR=${LOG_DIR:-results/round3e_overnight_logs}

mkdir -p "$LOG_DIR"

if [ ! -f data/qa/squad_train.jsonl ] || [ ! -f data/qa/squad_dev.jsonl ] || [ ! -f data/qa/addsent.jsonl ] || [ ! -f data/qa/addonesent.jsonl ]; then
  uv run --no-sync python scripts/materialize_qa_data.py --out-dir data/qa
fi

run_panel() {
  local name=$1
  local config=$2
  local gpu_ids=$3
  local workers=$4
  local log_file="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "[$name] config=$config gpu_ids=$gpu_ids workers=$workers log=$log_file"
  DATASET_ARTIFACTS_GPU_IDS="$gpu_ids" \
    scripts/run_full_panel.sh "$config" \
      --gpu-ids "$gpu_ids" \
      --parallel-workers "$workers" \
    2>&1 | tee "$log_file"
}

run_panel electra "$ELECTRA_CONFIG" "$ELECTRA_GPU_IDS" "$ELECTRA_WORKERS" &
electra_pid=$!

run_panel bert "$BERT_CONFIG" "$BERT_GPU_IDS" "$BERT_WORKERS" &
bert_pid=$!

status=0
if ! wait "$electra_pid"; then
  echo "[electra] failed" >&2
  status=1
fi
if ! wait "$bert_pid"; then
  echo "[bert] failed" >&2
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "Round 3E overnight panels complete."
else
  echo "One or more Round 3E overnight panels failed." >&2
fi

exit "$status"
