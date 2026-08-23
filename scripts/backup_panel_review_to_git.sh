#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:?Usage: scripts/backup_panel_review_to_git.sh RESULTS_DIR [BRANCH]}"
BRANCH="${2:-$(git branch --show-current)}"
BACKUP_ROOT="${BACKUP_ROOT:-github_backups}"
PANEL_NAME="$(basename "$RESULTS_DIR")"
DEST="$BACKUP_ROOT/$PANEL_NAME"
INCLUDE_PREDICTIONS="${INCLUDE_PREDICTIONS:-0}"
COMMIT_MSG="${COMMIT_MSG:-Backup review artifacts for $PANEL_NAME}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

mkdir -p "$DEST"

rsync -a --delete --prune-empty-dirs \
  --include='*/' \
  --include='/audit/***' \
  --include='/metrics/***' \
  --include='/figures/***' \
  --include='/logs/***' \
  --include='/cartography/*.csv' \
  --include='/cartography/*/*.png' \
  --include='/configs/***' \
  --include='/runs/*/all_results.json' \
  --include='/runs/*/train_metrics.json' \
  --include='/runs/*/train_results.json' \
  --include='/runs/*/run_manifest.json' \
  --include='/runs/*/trainer_state.json' \
  --include='/evals/*/eval_metrics.json' \
  --include='/evals/*/run_manifest.json' \
  --exclude='*' \
  "$RESULTS_DIR/" "$DEST/"

if [[ "$INCLUDE_PREDICTIONS" == "1" ]]; then
  mkdir -p "$DEST/predictions"
  rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='*.jsonl' \
    --exclude='*' \
    "$RESULTS_DIR/predictions/" "$DEST/predictions/" 2>/dev/null || true
  git lfs track "$DEST/**/*.jsonl" || true
fi

cat > "$DEST/BACKUP_README.md" <<EOF2
# $PANEL_NAME review backup

Source results directory: \`$RESULTS_DIR\`
Created UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Branch: \`$BRANCH\`
Predictions included: \`$INCLUDE_PREDICTIONS\`

This backup intentionally excludes model weights, optimizer states, full checkpoints,
and by default raw prediction JSONL files. It keeps the artifacts needed for paper
review: configs, logs, aggregate metrics, audit outputs, figures, cartography CSVs,
run manifests, and train/eval metric JSON files.
EOF2

# Keep checkpoint/model artifacts out even if rsync rules are changed later.
find "$DEST" -type f \( -name '*.bin' -o -name '*.pt' -o -name '*.safetensors' \) -delete
find "$DEST" -type d -name 'checkpoint-*' -prune -exec rm -rf {} +

git add -f "$DEST"
if [[ "$INCLUDE_PREDICTIONS" == "1" && -f .gitattributes ]]; then
  git add .gitattributes
fi

if git diff --cached --quiet; then
  echo "No backup changes to commit for $PANEL_NAME"
else
  git commit -m "$COMMIT_MSG"
fi

git push -u origin "$BRANCH"

echo "Backed up review artifacts to $DEST on branch $BRANCH"
