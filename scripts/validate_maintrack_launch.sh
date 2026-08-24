#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

case "$CONFIG" in
  /*) config_path="$CONFIG" ;;
  *) config_path="$REPO_ROOT/$CONFIG" ;;
esac

export MAINTRACK_GPU_IDS="${MAINTRACK_GPU_IDS:-0,1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
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
  done < "$config_path"
)"
if [ -z "$results_dir" ]; then
  echo "ERROR: Could not read suite.results_dir from $config_path" >&2
  exit 1
fi
case "$results_dir" in
  /*) ;;
  *) results_dir="$REPO_ROOT/$results_dir" ;;
esac
mkdir -p "$results_dir/status"

echo "=== foreground launch validation: safe disk reclamation ==="
if [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
  "$REPO_ROOT/.venv/bin/python3" scripts/reclaim_maintrack_space.py --config "$config_path"
fi

echo "=== foreground launch validation: dependency/runtime preflight ==="
uv sync --frozen --extra cuda --group dev
scripts/recover_adversarial_squad_sources.sh "$REPO_ROOT/data/sources/squad_adversarial"
scripts/preflight_maintrack.sh "$config_path"

echo "=== foreground launch validation: complete data materialization ==="
uv run --no-sync python scripts/run_maintrack_suite.py \
  --config "$config_path" \
  --stage data \
  --gpu-ids "$MAINTRACK_GPU_IDS"

echo "=== foreground launch validation: end-to-end GPU smoke matrix ==="
uv run --no-sync python scripts/run_maintrack_suite.py \
  --config "$config_path" \
  --stage smoke \
  --gpu-ids "$MAINTRACK_GPU_IDS"

for marker in DATA_SUCCESS.json SMOKE_SUCCESS.json; do
  if [ ! -s "$results_dir/status/$marker" ]; then
    echo "ERROR: Launch validation did not produce $results_dir/status/$marker" >&2
    exit 1
  fi
done

completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_head="$(git rev-parse HEAD)"
printf '{"completed_at_utc":"%s","git_head":"%s"}\n' "$completed_at" "$git_head" \
  > "$results_dir/status/LAUNCH_GATE_SUCCESS.json"

echo "Foreground launch validation passed: all data and the end-to-end smoke matrix are complete."
