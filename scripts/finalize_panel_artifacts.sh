#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:?Usage: scripts/finalize_panel_artifacts.sh RESULTS_DIR}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR/metrics" "$RESULTS_DIR/figures" "$RESULTS_DIR/logs"

if [[ -d "$RESULTS_DIR/metrics/raw" ]]; then
  python scripts/aggregate_metrics.py \
    --metrics-dir "$RESULTS_DIR/metrics/raw" \
    --out-dir "$RESULTS_DIR/metrics"
else
  echo "WARNING: $RESULTS_DIR/metrics/raw not found; skipping metric aggregation" >&2
fi

if [[ -f "$RESULTS_DIR/metrics/main_table.csv" ]]; then
  seed_args=()
  if [[ -f "$RESULTS_DIR/metrics/seed_level_metrics.csv" ]]; then
    seed_args=(--seed-table "$RESULTS_DIR/metrics/seed_level_metrics.csv")
  fi
  python scripts/plot_panel_metrics.py \
    --main-table "$RESULTS_DIR/metrics/main_table.csv" \
    "${seed_args[@]}" \
    --out-dir "$RESULTS_DIR/figures"
else
  echo "WARNING: main_table.csv not found; skipping panel figures" >&2
fi

if [[ -x scripts/audit_evalsets.py ]]; then
  python scripts/audit_evalsets.py --results-dir "$RESULTS_DIR" || true
fi

if [[ -x scripts/validate_results_tree.py ]]; then
  python scripts/validate_results_tree.py --results-dir "$RESULTS_DIR" || true
fi

find "$RESULTS_DIR" -maxdepth 3 \( -path '*/metrics/*' -o -path '*/figures/*' -o -path '*/logs/*' -o -path '*/cartography/*.csv' \) -type f | sort > "$RESULTS_DIR/logs/review_artifact_manifest.txt"

echo "Finalized review artifacts under $RESULTS_DIR"
