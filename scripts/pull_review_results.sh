#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:-${DATASET_ARTIFACTS_SERVER:-gpu-host}}"
REMOTE_ROOT="${2:-${DATASET_ARTIFACTS_REMOTE_RESULTS_DIR:-/workspace/dataset-artifacts/results/round3e_electra_small}}"
LOCAL_ROOT="${3:-${DATASET_ARTIFACTS_LOCAL_RESULTS_DIR:-server_results/results/round3e_electra_small}}"

mkdir -p "$LOCAL_ROOT"

rsync -aP --prune-empty-dirs \
  --include='*/' \
  --include='/audit/***' \
  --include='/metrics/***' \
  --include='/logs/***' \
  --include='/cartography/*.csv' \
  --include='/configs/***' \
  --include='/RUN_MANIFESTS/***' \
  --include='/CODE_AGENT_REPORT.md' \
  --exclude='*' \
  "$SERVER:$REMOTE_ROOT/" \
  "$LOCAL_ROOT/"

echo "Pulled review results into $LOCAL_ROOT"
