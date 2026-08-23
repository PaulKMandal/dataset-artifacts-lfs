#!/usr/bin/env bash
set -euo pipefail

SERVER="${DATASET_ARTIFACTS_SERVER:?Set DATASET_ARTIFACTS_SERVER, e.g. user@gpu-host}"
REMOTE_DIR="${DATASET_ARTIFACTS_REMOTE_DIR:-~/dataset-artifacts}"
REMOTE_RESULTS_SUBDIR="${DATASET_ARTIFACTS_REMOTE_RESULTS_SUBDIR:-results}"
LOCAL_RESULTS_DIR="${DATASET_ARTIFACTS_LOCAL_RESULTS_DIR:-./server_results}"
PULL_MODELS="${PULL_MODELS:-0}"

mkdir -p "$LOCAL_RESULTS_DIR"

if [[ "$PULL_MODELS" == "1" ]]; then
  rsync -az --info=stats2,progress2 \
    "$SERVER:$REMOTE_DIR/$REMOTE_RESULTS_SUBDIR/" \
    "$LOCAL_RESULTS_DIR/$REMOTE_RESULTS_SUBDIR/"
else
  rsync -az --prune-empty-dirs --info=stats2,progress2 \
    --include='*/' \
    --include='*.json' \
    --include='*.jsonl' \
    --include='*.csv' \
    --include='*.tsv' \
    --include='*.txt' \
    --include='*.log' \
    --include='*.png' \
    --include='*.yaml' \
    --include='*.yml' \
    --exclude='checkpoint-*' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.safetensors' \
    --exclude='optimizer.pt' \
    --exclude='scheduler.pt' \
    --exclude='rng_state*.pth' \
    --exclude='pytorch_model*' \
    --exclude='model.safetensors' \
    --exclude='*' \
    "$SERVER:$REMOTE_DIR/$REMOTE_RESULTS_SUBDIR/" \
    "$LOCAL_RESULTS_DIR/$REMOTE_RESULTS_SUBDIR/"
fi

echo "Pulled results into $LOCAL_RESULTS_DIR/$REMOTE_RESULTS_SUBDIR"
