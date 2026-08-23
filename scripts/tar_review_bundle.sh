#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-server_results/results/round3e_electra_small}"
OUT="${2:-cartography_review_bundle_$(date +%Y%m%d_%H%M%S).tar.gz}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "Results directory not found: $RESULTS_ROOT" >&2
  exit 1
fi

mkdir -p "$TMP_DIR/results"
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='/audit/***' \
  --include='/metrics/***' \
  --include='/logs/***' \
  --include='/cartography/*.csv' \
  --include='/configs/***' \
  --include='/RUN_MANIFESTS/***' \
  --include='/CODE_AGENT_REPORT.md' \
  --exclude='*' \
  "$RESULTS_ROOT/" \
  "$TMP_DIR/results/$(basename "$RESULTS_ROOT")/"

tar -czf "$OUT" -C "$TMP_DIR" results

echo "Wrote $OUT"
du -sh "$OUT"
