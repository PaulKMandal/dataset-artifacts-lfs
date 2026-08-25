#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
GPU_IDS="${MAINTRACK_GPU_IDS:-0,1}"

for command_name in nvidia-smi tmux uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 1
  fi
done

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
if [ "${#gpu_array[@]}" -ne 2 ]; then
  echo "MAINTRACK_GPU_IDS must name exactly two GPUs; got: $GPU_IDS" >&2
  exit 1
fi

for gpu_id in "${gpu_array[@]}"; do
  gpu_name="$(nvidia-smi --id="$gpu_id" --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "$gpu_name" != *V100* ]]; then
    echo "GPU $gpu_id is not a V100: $gpu_name" >&2
    exit 1
  fi
done

uv run --no-sync python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.device_count() >= 2, f"Expected at least 2 visible GPUs, found {torch.cuda.device_count()}"
for index in range(2):
    name = torch.cuda.get_device_name(index)
    assert "V100" in name, f"CUDA device {index} is not a V100: {name}"
print("CUDA preflight passed:", [torch.cuda.get_device_name(i) for i in range(2)])
PY

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uv run --no-sync pytest -q
uv run --no-sync ruff check \
  run.py helpers.py dynamics.py dataset_cache.py \
  scripts/run_maintrack_suite.py \
  scripts/reclaim_maintrack_space.py \
  scripts/aggregate_maintrack_metrics.py \
  scripts/audit_maintrack_resume.py \
  scripts/materialize_qa_data.py \
  scripts/materialize_sentence_classification.py \
  scripts/select_qa_subsets.py \
  scripts/analyze_map_stability.py \
  scripts/analyze_subset_coverage.py \
  scripts/combine_cartography_maps.py \
  scripts/hierarchical_bootstrap.py \
  scripts/maintrack_status.py \
  tests
uv run --no-sync python -m py_compile run.py helpers.py dynamics.py dataset_cache.py compare_adversarial.py scripts/*.py tests/*.py
uv run --no-sync python scripts/run_maintrack_suite.py \
  --config "$CONFIG" \
  --stage all \
  --dry-run \
  --print-job-counts

echo "Main-track preflight passed."
