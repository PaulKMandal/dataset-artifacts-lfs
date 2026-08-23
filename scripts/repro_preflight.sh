#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/panel.minimum.yaml}"
LIMIT_RUNS="${REPRO_PREFLIGHT_LIMIT_RUNS:-2}"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export UV_NO_SYNC=1

uv run --no-sync pytest -q
uv run --no-sync python -m py_compile \
  run.py helpers.py dynamics.py compare_adversarial.py scripts/*.py tests/*.py
uv run --no-sync python scripts/run_experiment_panel.py \
  --config "$CONFIG" \
  --dry-run \
  --limit-runs "$LIMIT_RUNS"
