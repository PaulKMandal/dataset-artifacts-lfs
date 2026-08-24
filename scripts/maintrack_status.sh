#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
shift || true
exec uv run --no-sync python scripts/maintrack_status.py --config "$CONFIG" "$@"
