#!/usr/bin/env bash
set -Eeuo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
BASE_DEST="${BASE_DEST:-$PWD/adversarial_repairs}"

SERVERS=(
  # Add hosts as "HOST:PORT" when running outside the anonymized submission package.
)

if (( ${#SERVERS[@]} == 0 )); then
  echo "No servers configured. Add HOST:PORT entries to SERVERS before running this helper." >&2
  exit 2
fi

test -x scripts/runpod_run_adversarial_repair_and_pull.sh || {
  echo "Missing or non-executable scripts/runpod_run_adversarial_repair_and_pull.sh" >&2
  exit 1
}

mkdir -p "$BASE_DEST"

for hp in "${SERVERS[@]}"; do
  host="${hp%%:*}"
  port="${hp##*:}"
  label="${host}_${port}"
  dest="$BASE_DEST/$label"

  echo
  echo "============================================================"
  echo "Running adversarial repair on $host:$port"
  echo "Destination: $dest"
  echo "============================================================"

  mkdir -p "$dest"

  DIRECT_HOST="$host" \
  DIRECT_PORT="$port" \
  REMOTE_USER=root \
  SSH_KEY="$SSH_KEY" \
  DEST_DIR="$dest" \
  scripts/runpod_run_adversarial_repair_and_pull.sh
done

echo
echo "All requested servers processed."
find "$BASE_DEST" -type f -name 'cartography_adversarial_repair_*.tar.gz' -print -exec du -sh {} \;
