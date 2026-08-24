#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DESTINATION="${1:-$REPO_ROOT/data/sources/squad_adversarial}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python environment is unavailable: $PYTHON_BIN" >&2
  echo "Run uv sync --frozen --extra cuda --group dev first." >&2
  exit 1
fi

mkdir -p "$DESTINATION"

search_roots=(
  /home/rhel/Projects/dataset-artifacts
  /home/rhel/Data/dataset-artifacts
  /home/rhel/.cache/huggingface
  /home/rhel/Data/dataset-artifacts-maintrack/data
  /home/rhel/Data/dataset-artifacts-maintrack/cache/huggingface
)

validate_jsonl() {
  local path=$1
  local expected_count=$2
  "$PYTHON_BIN" - "$path" "$expected_count" <<'PY'
import sys
from pathlib import Path

from scripts.materialize_qa_data import local_records, validate_record

path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
seen = set()
count = 0
for record in local_records(path):
    validate_record(record, path)
    identifier = record["id"]
    if identifier in seen:
        raise ValueError(f"duplicate id: {identifier}")
    seen.add(identifier)
    count += 1
if count != expected_count:
    raise ValueError(f"found {count} examples; expected {expected_count}")
print(f"validated {path}: {count} examples")
PY
}

copy_atomic() {
  local source=$1
  local destination=$2
  local temporary
  temporary="$(mktemp "$DESTINATION/.recovery.XXXXXXXX")"
  cp -- "$source" "$temporary"
  chmod 0644 "$temporary"
  mv -- "$temporary" "$destination"
}

recover_one() {
  local name=$1
  local expected_size=$2
  local expected_sha=$3
  local expected_count=$4
  local raw_destination="$DESTINATION/$name.json"
  local jsonl_destination="$DESTINATION/$name.jsonl"

  if [ -f "$raw_destination" ]; then
    actual_sha="$(sha256sum "$raw_destination" | cut -d' ' -f1)"
    if [ "$(stat -c %s "$raw_destination")" = "$expected_size" ] && \
       [ "$actual_sha" = "$expected_sha" ]; then
      echo "Using existing checksum-pinned source: $raw_destination"
      return 0
    fi
    echo "ERROR: Existing source does not match the official checksum: $raw_destination" >&2
    return 1
  fi
  if [ -f "$jsonl_destination" ]; then
    validate_jsonl "$jsonl_destination" "$expected_count"
    echo "Using existing validated materialization: $jsonl_destination"
    return 0
  fi

  for root in "${search_roots[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r -d '' candidate; do
      actual_sha="$(sha256sum "$candidate" | cut -d' ' -f1)"
      if [ "$actual_sha" = "$expected_sha" ]; then
        copy_atomic "$candidate" "$raw_destination"
        echo "Recovered official checksum-pinned $name from $candidate"
        return 0
      fi
    done < <(find "$root" -xdev -type f -size "${expected_size}c" -print0 2>/dev/null)
  done

  for root in "${search_roots[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r -d '' candidate; do
      if validate_jsonl "$candidate" "$expected_count" >/dev/null 2>&1; then
        copy_atomic "$candidate" "$jsonl_destination"
        validate_jsonl "$jsonl_destination" "$expected_count"
        echo "Recovered validated $name JSONL from $candidate"
        return 0
      fi
    done < <(find "$root" -xdev -type f -name "$name.jsonl" -print0 2>/dev/null)
  done

  echo "No prior local $name source was found; the foreground gate will try the official endpoint."
}

cd "$REPO_ROOT"
recover_one addsent \
  4073864 \
  40e3602aa5195cdacd03904a9c301ceb17ccf730cc32bd3ab998b66b4401e660 \
  3560
recover_one addonesent \
  1920649 \
  50420ac8d8b7547cd3715347c9a276802bc9466328ba0814adce4c20495e2889 \
  1787

echo "Adversarial SQuAD source recovery check complete: $DESTINATION"
