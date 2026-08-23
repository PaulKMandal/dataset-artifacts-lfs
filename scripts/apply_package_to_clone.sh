#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="${1:?Usage: scripts/apply_package_to_clone.sh /path/to/dataset-artifacts-repro}"
TARGET_DIR="${2:-$(pwd)}"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  echo "Target directory is not a git clone: $TARGET_DIR" >&2
  exit 2
fi

rsync -a \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='results/' \
  --exclude='server_results/' \
  "$PKG_DIR/" "$TARGET_DIR/"

cat <<'EOF'
Copied package files into the target clone.
Next:
  git status
  nix develop .#default
  uv sync --extra cpu --group dev
  uv run pytest
  git add .
  git commit -m 'Add reproducible Nix uv setup and fast scalar dynamics'
EOF
