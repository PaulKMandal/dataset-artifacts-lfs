#!/usr/bin/env bash
set -euo pipefail

export UV_NO_SYNC=1

# PyTorch CUDA wheels install support libraries under site-packages/nvidia/*/lib.
# Add them at script runtime because they may not exist when the Nix shell first starts.
for p in "$PWD"/.venv/lib/python3.11/site-packages/nvidia/*/lib; do
  if [ -d "$p" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done

# Host NVIDIA driver shim, created by the Nix shell hook.
if [ -d "$PWD/.nix-driver-libs" ]; then
  export LD_LIBRARY_PATH="$PWD/.nix-driver-libs:${LD_LIBRARY_PATH:-}"
fi

OUT_DIR="${1:-results/smoke_squad_fast_dynamics}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --no-sync python run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 1 \
  --save_only_final_model \
  --save_dynamics \
  --fp16 \
  --report_to none

uv run --no-sync python dynamics.py \
  --td_dir "$OUT_DIR" \
  --output_dir "$OUT_DIR/cartography"
