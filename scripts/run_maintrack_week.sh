#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
export UV_NO_SYNC=1
export MAINTRACK_GPU_IDS="${MAINTRACK_GPU_IDS:-0,1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

results_dir="$(
  while read -r key value _; do
    if [ "$key" = "results_dir:" ]; then
      printf '%s\n' "$value"
      break
    fi
  done < "$CONFIG"
)"
if [ -z "$results_dir" ]; then
  echo "Could not read suite.results_dir from $CONFIG" >&2
  exit 1
fi
mkdir -p "$results_dir/status"

exit_file="$results_dir/status/week_runner_exit_code.txt"
started_file="$results_dir/status/week_runner_started_at.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$started_file"

record_exit() {
  exit_code=$?
  printf '%s\n' "$exit_code" > "$exit_file"
  exit "$exit_code"
}
trap record_exit EXIT

uv sync --frozen --extra cuda --group dev
scripts/preflight_maintrack.sh "$CONFIG"
uv run --no-sync python scripts/run_maintrack_suite.py \
  --config "$CONFIG" \
  --stage all \
  --gpu-ids "$MAINTRACK_GPU_IDS"
