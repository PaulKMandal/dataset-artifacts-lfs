#!/usr/bin/env bash
set -Eeuo pipefail

DIRECT_HOST="${DIRECT_HOST:?Set DIRECT_HOST, e.g. HOSTNAME_OR_IP}"
DIRECT_PORT="${DIRECT_PORT:?Set DIRECT_PORT, e.g. PORT}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/dataset-artifacts}"
DEST_DIR="${DEST_DIR:-$PWD/adversarial_repairs}"
LOCAL_REPO="${LOCAL_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
REMOTE_RUNNER="/workspace/run_adversarial_repair_once.sh"
REMOTE_TARBALL_PATH_FILE="/workspace/latest_adversarial_repair_tarball_path.txt"
REMOTE_TARBALL_SHA_FILE="/workspace/latest_adversarial_repair_tarball_sha256.txt"

SSH_OPTS=(
  -p "$DIRECT_PORT"
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
)

REMOTE="${REMOTE_USER}@${DIRECT_HOST}"

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { printf '\n[error] %s\n' "$*" >&2; exit 1; }

need_local_file() {
  local p="$1"
  [ -f "$LOCAL_REPO/$p" ] || fail "Missing local file: $LOCAL_REPO/$p"
}

need_local_file "scripts/repair_adversarial_metrics.py"
need_local_file "scripts/run_adversarial_repair_all.sh"
need_local_file "docs/adversarial_repair_runbook.md"

mkdir -p "$DEST_DIR"

log "Remote: $REMOTE"
log "Remote repo: $REMOTE_REPO"
log "Destination: $DEST_DIR"

log "Checking SSH connection"
ssh "${SSH_OPTS[@]}" "$REMOTE" 'echo "connected: $(hostname)"; pwd'

log "Installing remote system packages"
ssh "${SSH_OPTS[@]}" "$REMOTE" '
  set -Eeuo pipefail

  export DEBIAN_FRONTEND=noninteractive

  apt-get update
  apt-get install -y \
    sudo git git-lfs curl xz-utils ca-certificates \
    rsync tmux less nano coreutils procps findutils openssh-client
'

log "Checking remote repo exists; not cloning or pulling"
ssh "${SSH_OPTS[@]}" "$REMOTE" "
  set -Eeuo pipefail

  test -d '$REMOTE_REPO' || {
    echo 'Missing remote repo: $REMOTE_REPO' >&2
    exit 1
  }

  cd '$REMOTE_REPO'

  echo 'branch:' \$(git branch --show-current || true)
  echo 'commit:' \$(git rev-parse --short HEAD || true)
  echo 'results directories:'
  find results -maxdepth 1 -mindepth 1 -type d -print 2>/dev/null | sort || true

  grep -qxF '/data/' .git/info/exclude || echo '/data/' >> .git/info/exclude
  grep -qxF '/results/' .git/info/exclude || echo '/results/' >> .git/info/exclude
  grep -qxF '/.venv/' .git/info/exclude || echo '/.venv/' >> .git/info/exclude
"

log "Uploading repair scripts only"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_REPO/scripts' '$REMOTE_REPO/docs'"

rsync -aP --no-owner --no-group --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  -e "ssh -p $DIRECT_PORT -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "$LOCAL_REPO/scripts/repair_adversarial_metrics.py" \
  "$LOCAL_REPO/scripts/run_adversarial_repair_all.sh" \
  "$REMOTE:$REMOTE_REPO/scripts/"

rsync -aP --no-owner --no-group --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  -e "ssh -p $DIRECT_PORT -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "$LOCAL_REPO/docs/adversarial_repair_runbook.md" \
  "$REMOTE:$REMOTE_REPO/docs/"

ssh "${SSH_OPTS[@]}" "$REMOTE" "chmod +x '$REMOTE_REPO/scripts/repair_adversarial_metrics.py' '$REMOTE_REPO/scripts/run_adversarial_repair_all.sh'"

log "Writing remote repair runner"
ssh "${SSH_OPTS[@]}" "$REMOTE" "cat > '$REMOTE_RUNNER' <<'REMOTE_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_REPO='$REMOTE_REPO'
TARBALL_PATH_FILE='$REMOTE_TARBALL_PATH_FILE'
TARBALL_SHA_FILE='$REMOTE_TARBALL_SHA_FILE'

cd \"\$REMOTE_REPO\"

echo \"===== adversarial repair started \$(date -u +%Y-%m-%dT%H:%M:%SZ) =====\"
echo \"repo: \$REMOTE_REPO\"
echo \"branch: \$(git branch --show-current || true)\"
echo \"commit: \$(git rev-parse --short HEAD || true)\"

echo \"===== available results dirs =====\"
find results -maxdepth 1 -mindepth 1 -type d -print 2>/dev/null | sort || true

echo \"===== prediction JSONL count =====\"
find results -type f -path '*/predictions/*.jsonl' 2>/dev/null | wc -l || true

echo \"===== setting up runtime shims =====\"
mkdir -p /workspace/runtime-libs /workspace/cuda-driver-lib /workspace/.cache/huggingface /workspace/.cache/uv

find /workspace/runtime-libs -type l -delete || true
for f in /lib/x86_64-linux-gnu/libz.so* /usr/lib/x86_64-linux-gnu/libz.so*; do
  [ -e \"\$f\" ] && ln -sf \"\$f\" \"/workspace/runtime-libs/\$(basename \"\$f\")\"
done

find /workspace/cuda-driver-lib -type l -delete || true
for f in \\
  /usr/lib/x86_64-linux-gnu/libcuda.so* \\
  /usr/lib/x86_64-linux-gnu/libnvidia*.so* \\
  /usr/local/nvidia/lib64/libcuda.so* \\
  /usr/local/nvidia/lib64/libnvidia*.so*
do
  [ -e \"\$f\" ] && ln -sf \"\$f\" \"/workspace/cuda-driver-lib/\$(basename \"\$f\")\"
done

cat > /workspace/.env <<'ENV_EOF'
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export UV_CACHE_DIR=/workspace/.cache/uv
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ENV_EOF

echo \"===== ensuring Nix =====\"
unset LD_LIBRARY_PATH || true
mkdir -p /etc/nix
cat > /etc/nix/nix.conf <<'NIX_EOF'
build-users-group =
experimental-features = nix-command flakes
accept-flake-config = true
NIX_EOF

export PATH=\"/root/.nix-profile/bin:/nix/var/nix/profiles/per-user/root/profile/bin:/nix/var/nix/profiles/default/bin:\$PATH\"

if ! command -v nix >/dev/null 2>&1; then
  sh <(curl -L https://nixos.org/nix/install) --no-daemon
  export PATH=\"/root/.nix-profile/bin:/nix/var/nix/profiles/per-user/root/profile/bin:/nix/var/nix/profiles/default/bin:\$PATH\"
fi

nix --version

echo \"===== running repair inside Nix shell =====\"
unset LD_LIBRARY_PATH
unset DATASET_ARTIFACTS_GPU_IDS

env -u LD_LIBRARY_PATH nix develop .#server --command bash -lc '
  set -Eeuo pipefail

  cd \"'$REMOTE_REPO'\"
  . /workspace/.env

  unset LD_LIBRARY_PATH
  unset DATASET_ARTIFACTS_GPU_IDS

  uv sync --extra cuda --group dev
  uv pip install --python \"\$PWD/.venv/bin/python\" sentencepiece protobuf
  export PATH=\"\$PWD/.venv/bin:\$PATH\"

  CXXLIB=\"\$(dirname \"\$(gcc -print-file-name=libstdc++.so.6)\")\"
  export LD_LIBRARY_PATH=\"/workspace/runtime-libs:\$CXXLIB:/workspace/cuda-driver-lib:/usr/local/nvidia/lib64:\${LD_LIBRARY_PATH:-}\"

  python3 - <<PY
import numpy, pandas, yaml
print(\"numpy:\", numpy.__version__)
print(\"pandas:\", pandas.__version__)
print(\"PyYAML OK\")
PY

  scripts/run_adversarial_repair_all.sh
'

echo \"===== locating newest repair tarball =====\"
tarball=\"\$(find \"\$REMOTE_REPO\" -maxdepth 1 -type f -name 'cartography_adversarial_repair_*.tar.gz' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)\"

if [ -z \"\$tarball\" ]; then
  echo \"ERROR: no cartography_adversarial_repair_*.tar.gz found in \$REMOTE_REPO\" >&2
  exit 1
fi

test -s \"\$tarball\"

sha256sum \"\$tarball\" > \"\$TARBALL_SHA_FILE\"
printf '%s\n' \"\$tarball\" > \"\$TARBALL_PATH_FILE\"

echo \"===== repair tarball =====\"
ls -lh \"\$tarball\"
cat \"\$TARBALL_SHA_FILE\"

echo \"===== adversarial repair finished \$(date -u +%Y-%m-%dT%H:%M:%SZ) =====\"
REMOTE_EOF
chmod +x '$REMOTE_RUNNER'"

log "Running remote adversarial repair"
ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_RUNNER" | tee "$DEST_DIR/remote_adversarial_repair_$(date +%Y%m%d_%H%M%S).log"

log "Reading remote tarball path"
REMOTE_TARBALL="$(
  ssh "${SSH_OPTS[@]}" "$REMOTE" "cat '$REMOTE_TARBALL_PATH_FILE'"
)"
REMOTE_TARBALL="${REMOTE_TARBALL//$'\r'/}"

[ -n "$REMOTE_TARBALL" ] || fail "Remote tarball path was empty"

log "Remote tarball: $REMOTE_TARBALL"

log "Reading remote checksum"
REMOTE_SHA="$(
  ssh "${SSH_OPTS[@]}" "$REMOTE" "sha256sum '$REMOTE_TARBALL' | awk '{print \$1}'"
)"
REMOTE_SHA="${REMOTE_SHA//$'\r'/}"

log "Remote SHA256: $REMOTE_SHA"

log "Pulling tarball with rsync progress"
rsync -aP \
  -e "ssh -p $DIRECT_PORT -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "$REMOTE:$REMOTE_TARBALL" \
  "$DEST_DIR/"

LOCAL_TARBALL="$DEST_DIR/$(basename "$REMOTE_TARBALL")"

log "Verifying local checksum"
LOCAL_SHA="$(sha256sum "$LOCAL_TARBALL" | awk '{print $1}')"

echo "remote sha256: $REMOTE_SHA"
echo "local  sha256: $LOCAL_SHA"

if [ "$REMOTE_SHA" != "$LOCAL_SHA" ]; then
  fail "Checksum mismatch for $LOCAL_TARBALL"
fi

log "Verifying tarball integrity"
tar -tzf "$LOCAL_TARBALL" >/dev/null

ls -lh "$LOCAL_TARBALL"

log "Done"
echo
echo "Upload this file:"
echo "$LOCAL_TARBALL"
