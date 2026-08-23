#!/usr/bin/env bash
set -euo pipefail

SERVER="${DATASET_ARTIFACTS_SERVER:?Set DATASET_ARTIFACTS_SERVER, e.g. user@gpu-host}"
REMOTE_DIR="${DATASET_ARTIFACTS_REMOTE_DIR:-~/dataset-artifacts}"
REMOTE_FLAKE_ATTR="${DATASET_ARTIFACTS_REMOTE_FLAKE_ATTR:-server}"
REMOTE_SYNC_EXTRA="${DATASET_ARTIFACTS_REMOTE_SYNC_EXTRA:-cuda}"
REMOTE_SYNC_GROUP="${DATASET_ARTIFACTS_REMOTE_SYNC_GROUP:-dev}"

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/remote_run.sh uv run python run.py --do_train ...

Required env:
  DATASET_ARTIFACTS_SERVER=user@gpu-host

Optional env:
  DATASET_ARTIFACTS_REMOTE_DIR=~/dataset-artifacts
  DATASET_ARTIFACTS_LOCAL_RESULTS_DIR=./server_results
  DATASET_ARTIFACTS_REMOTE_RESULTS_SUBDIR=results
  DATASET_ARTIFACTS_REMOTE_FLAKE_ATTR=server
  DATASET_ARTIFACTS_REMOTE_SYNC_EXTRA=cuda
EOF
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Push source but not local environments, caches, results, or model outputs.
rsync -az --delete --info=stats2 \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.direnv/' \
  --exclude='.cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='data/' \
  --exclude='results/' \
  --exclude='server_results/' \
  --exclude='output/' \
  --exclude='outputs/' \
  --exclude='checkpoints/' \
  ./ "$SERVER:$REMOTE_DIR/"

tmp_script="$(mktemp)"
trap 'rm -f "$tmp_script"' EXIT
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'echo "Remote host: $(hostname)"'
  echo 'echo "Remote pwd: $(pwd)"'
  printf 'uv sync --extra %q --group %q\n' "$REMOTE_SYNC_EXTRA" "$REMOTE_SYNC_GROUP"
  echo 'echo "Python environment:"'
  echo 'uv run python - <<'"'PY'"''
  echo 'import sys'
  echo 'print(sys.version)'
  echo 'try:'
  echo '    import torch'
  echo '    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
  echo 'except Exception as exc:'
  echo '    print("torch import failed", repr(exc))'
  echo 'PY'
  echo 'echo "Running user command:"'
  printf 'printf '\''%%q '\'''
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\nprintf '\''\\n'\''\n'
  printf 'exec'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$tmp_script"

scp "$tmp_script" "$SERVER:$REMOTE_DIR/.remote_command.sh" >/dev/null
remote_attr_q="$(printf '%q' "$REMOTE_FLAKE_ATTR")"
# Keep REMOTE_DIR unescaped so a leading ~ expands on the remote shell. Avoid spaces in REMOTE_DIR.
ssh "$SERVER" "cd $REMOTE_DIR && chmod +x .remote_command.sh && nix develop .#$remote_attr_q -c bash .remote_command.sh"

# Pull metrics/results back. Models are excluded unless PULL_MODELS=1 is set.
"$repo_root/scripts/pull_results.sh"
