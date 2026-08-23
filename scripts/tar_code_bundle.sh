#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
OUT="${2:-dataset_artifacts_code_bundle_$(date +%Y%m%d_%H%M%S).tar.gz}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
DEST="$TMP_DIR/dataset-artifacts"
mkdir -p "$DEST"

rsync -a \
  --exclude='/.git' \
  --exclude='/.venv' \
  --exclude='/.direnv' \
  --exclude='/data' \
  --exclude='/results' \
  --exclude='/server_results' \
  --exclude='/wandb' \
  --exclude='/outputs' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='*.pyc' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  "$REPO_ROOT/" \
  "$DEST/"

SNAPSHOT_DIR="$DEST/GIT_SNAPSHOT"
mkdir -p "$SNAPSHOT_DIR"
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD > "$SNAPSHOT_DIR/branch.txt" || true
  git -C "$REPO_ROOT" rev-parse HEAD > "$SNAPSHOT_DIR/commit.txt" || true
  git -C "$REPO_ROOT" status --short > "$SNAPSHOT_DIR/status_short.txt" || true
  git -C "$REPO_ROOT" remote -v > "$SNAPSHOT_DIR/remotes.txt" || true
  git -C "$REPO_ROOT" diff --binary > "$SNAPSHOT_DIR/uncommitted.diff" || true
  git -C "$REPO_ROOT" diff --cached --binary > "$SNAPSHOT_DIR/staged.diff" || true
  git -C "$REPO_ROOT" ls-files > "$SNAPSHOT_DIR/tracked_files.txt" || true
fi

tar -czf "$OUT" -C "$TMP_DIR" dataset-artifacts

echo "Wrote $OUT"
du -sh "$OUT"
