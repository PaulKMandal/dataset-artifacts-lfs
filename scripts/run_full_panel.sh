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


export UV_NO_SYNC=1

# PyTorch CUDA wheel libraries are under site-packages/nvidia/*/lib after uv sync.
for p in "$PWD"/.venv/lib/python3.11/site-packages/nvidia/*/lib; do
  if [ -d "$p" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done

# Host NVIDIA driver shim, created by the Nix shell hook.
if [ -d "$PWD/.nix-driver-libs" ]; then
  export LD_LIBRARY_PATH="$PWD/.nix-driver-libs:${LD_LIBRARY_PATH:-}"
fi

CONFIG="${1:-configs/panel.full.yaml}"
shift || true

uv run --no-sync python scripts/run_experiment_panel.py --config "$CONFIG" "$@"
