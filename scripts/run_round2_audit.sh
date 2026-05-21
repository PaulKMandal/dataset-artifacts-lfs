#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-results/panel_electra_small}"
AUDIT_DIR="$RESULTS_DIR/audit"

python scripts/audit_current_results.py \
  --results-dir "$RESULTS_DIR" \
  --out-dir "$AUDIT_DIR"

python scripts/aggregate_metrics.py \
  --metrics-dir "$RESULTS_DIR/metrics/raw" \
  --out-dir "$RESULTS_DIR/metrics" \
  --eval-split-metrics "$AUDIT_DIR/eval_split_metrics.csv"

python scripts/audit_current_results.py \
  --results-dir "$RESULTS_DIR" \
  --out-dir "$AUDIT_DIR"

python scripts/validate_results_tree.py --results-dir "$RESULTS_DIR"
