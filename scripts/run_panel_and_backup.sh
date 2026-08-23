#!/usr/bin/env bash
set -euo pipefail

# RunPod/Nix runtime-library shim.
# Do not add /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH wholesale.
# Symlink only required runtime libs such as libz into a narrow shim dir.
if [ -z "${DATASET_ARTIFACTS_RUNTIME_LIBS_READY:-}" ]; then
  if [ -d /workspace ]; then
    _runtime_lib_dir="/workspace/runtime-libs"
  else
    _runtime_lib_dir="$PWD/.runtime-libs"
  fi

  mkdir -p "$_runtime_lib_dir" 2>/dev/null || true

  for _f in \
    /lib/x86_64-linux-gnu/libz.so* \
    /usr/lib/x86_64-linux-gnu/libz.so*
  do
    if [ -e "$_f" ]; then
      ln -sf "$_f" "$_runtime_lib_dir/$(basename "$_f")" 2>/dev/null || true
    fi
  done

  if command -v gcc >/dev/null 2>&1; then
    _cxxlib="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
  else
    _cxxlib=""
  fi

  _driver_lib="${DATASET_ARTIFACTS_DRIVER_LIB_DIR:-/workspace/cuda-driver-lib}"

  for _lib in "$_runtime_lib_dir" "$_cxxlib" "$_driver_lib" /usr/local/nvidia/lib64; do
    if [ -n "$_lib" ] && [ -d "$_lib" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$_lib:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done

  export DATASET_ARTIFACTS_RUNTIME_LIBS_READY=1
fi


CONFIG="${1:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
RUN_NUMBER="${2:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
MODEL_SLUG="${3:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
BASE_BRANCH="${BASE_BRANCH:-working}"
BRANCH="run${RUN_NUMBER}_${MODEL_SLUG}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree has uncommitted changes. Commit/stash before launching automatic branch backup." >&2
  git status --short
  exit 1
fi

git fetch origin --prune

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
elif git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git switch --track "origin/$BRANCH"
else
  if git show-ref --verify --quiet "refs/remotes/origin/$BASE_BRANCH"; then
    git switch -c "$BRANCH" "origin/$BASE_BRANCH"
  else
    git switch -c "$BRANCH"
  fi
  git push -u origin "$BRANCH"
fi

RESULTS_DIR="$(python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('$CONFIG').read_text())
print(cfg['panel']['results_dir'])
PY
)"

echo "Running panel config: $CONFIG"
echo "Branch: $BRANCH"
echo "Results: $RESULTS_DIR"

scripts/run_full_panel.sh "$CONFIG"
scripts/finalize_panel_artifacts.sh "$RESULTS_DIR"

COMMIT_MSG="Add Run $RUN_NUMBER $MODEL_SLUG review artifacts" \
  scripts/backup_panel_review_to_git.sh "$RESULTS_DIR" "$BRANCH"
