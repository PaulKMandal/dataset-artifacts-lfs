#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:?Set REPO_URL to the artifact repository URL}"
OUT_ROOT="${OUT_ROOT:-$PWD/analysis_results_pull}"
BUNDLE_DIR="${BUNDLE_DIR:-$PWD/bundles}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_PATH="${BUNDLE_PATH:-$BUNDLE_DIR/cartography_analysis_results_bundle_${STAMP}.tar.gz}"
PULL_LFS="${PULL_LFS:-0}"

RUN_LABELS=(
  run7_electra_small
  run5_6_bert_base
  run5_electra_base
  run5_roberta_base
  run6_deberta_v3_base
  run6_distilbert
  run6_albert_base
)

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
fail() { printf '[error] %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || fail "git not found"
command -v rsync >/dev/null || fail "rsync not found"
command -v tar >/dev/null || fail "tar not found"

mkdir -p "$BUNDLE_DIR"
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pick_branch() {
  local label="$1"
  local candidates=(
    "artifacts/$label"
    "$label"
  )

  local b
  for b in "${candidates[@]}"; do
    if git ls-remote --exit-code --heads "$REPO_URL" "$b" >/dev/null 2>&1; then
      printf '%s\n' "$b"
      return 0
    fi
  done

  return 1
}

copy_compact_results_from_clone() {
  local clone_dir="$1"
  local label="$2"
  local dest="$OUT_ROOT/$label"

  mkdir -p "$dest"

  # Preferred artifact-branch layout from finalize_and_backup_run_artifacts.sh.
  if [ -d "$clone_dir/github_backups/$label" ]; then
    rsync -a \
      --exclude='.git/' \
      --exclude='*.pt' \
      --exclude='*.bin' \
      --exclude='*.safetensors' \
      --exclude='checkpoint-*/' \
      "$clone_dir/github_backups/$label/" "$dest/"
    return 0
  fi

  # If the entire artifact branch root is the backup.
  if [ -d "$clone_dir/github_backups" ]; then
    rsync -a \
      --exclude='.git/' \
      --exclude='*.pt' \
      --exclude='*.bin' \
      --exclude='*.safetensors' \
      --exclude='checkpoint-*/' \
      "$clone_dir/github_backups/" "$dest/github_backups/"
    return 0
  fi

  # Fallback: copy compact outputs from results trees.
  if [ -d "$clone_dir/results" ]; then
    rsync -a \
      --include='*/' \
      --include='audit/***' \
      --include='metrics/***' \
      --include='figures/***' \
      --include='logs/***' \
      --include='configs/***' \
      --include='cartography/**/*.csv' \
      --include='cartography/**/*.json' \
      --include='cartography/**/*.png' \
      --include='runs/*/all_results.json' \
      --include='runs/*/train_metrics.json' \
      --include='runs/*/train_results.json' \
      --include='runs/*/trainer_state.json' \
      --include='runs/*/run_manifest.json' \
      --include='runs/*/config.json' \
      --include='evals/*/eval_metrics.json' \
      --include='evals/*/run_manifest.json' \
      --exclude='*' \
      "$clone_dir/results/" "$dest/results/"
    return 0
  fi

  # Last resort: copy the whole compact branch excluding obvious bulk.
  rsync -a \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='data/' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='checkpoint-*/' \
    "$clone_dir/" "$dest/"
}

pull_one() {
  local label="$1"
  local branch
  local safe_branch
  local clone_dir
  local dest
  local file_count

  if ! branch="$(pick_branch "$label")"; then
    warn "missing branch for $label; tried artifacts/$label and $label"
    return 0
  fi

  safe_branch="${branch//\//_}"
  clone_dir="$TMP/$safe_branch"
  dest="$OUT_ROOT/$label"

  log "pulling $label from $branch"

  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch "$branch" "$REPO_URL" "$clone_dir"

  if [ "$PULL_LFS" = "1" ] && command -v git-lfs >/dev/null 2>&1; then
    log "PULL_LFS=1, pulling LFS objects for $label"
    git -C "$clone_dir" lfs pull || warn "git lfs pull failed for $label; continuing"
  fi

  log "tree preview for $label"
  find "$clone_dir" -maxdepth 5 -mindepth 1 \
    -not -path "$clone_dir/.git*" \
    | sed "s#^$clone_dir#  #" \
    | head -120 || true

  copy_compact_results_from_clone "$clone_dir" "$label"

  file_count="$(find "$dest" -type f 2>/dev/null | wc -l | tr -d ' ')"

  {
    echo "label=$label"
    echo "branch=$branch"
    echo "repo_url=$REPO_URL"
    echo "pulled_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=$(git -C "$clone_dir" rev-parse HEAD)"
    echo "file_count=$file_count"
    echo
    echo "files:"
    find "$dest" -type f 2>/dev/null | sed "s#^$dest/##" | sort
  } > "$dest/PULL_MANIFEST.txt"

  if [ "$file_count" -eq 0 ]; then
    warn "$label copied zero files"
    return 0
  fi

  if ! find "$dest" -type f \( \
      -name 'main_table.csv' -o \
      -name 'seed_level_metrics.csv' -o \
      -name 'eval_metrics.json' -o \
      -name 'ARTIFACT_MANIFEST.txt' -o \
      -name 'table_audit.md' \
    \) | grep -q .; then
    warn "$label does not contain obvious metrics/audit markers"
  fi

  log "$label copied $file_count files"
}

for label in "${RUN_LABELS[@]}"; do
  pull_one "$label"
done

log "writing top-level manifest"

{
  echo "bundle_created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_url=$REPO_URL"
  echo "out_root=$OUT_ROOT"
  echo
  echo "expected_runs:"
  printf '  %s\n' "${RUN_LABELS[@]}"
  echo
  echo "present_runs:"
  find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort
  echo
  echo "included_files:"
  find "$OUT_ROOT" -type f | sed "s#^$OUT_ROOT/##" | sort
  echo
  echo "sizes:"
  du -ah "$OUT_ROOT" | sort -h
} > "$OUT_ROOT/ANALYSIS_BUNDLE_MANIFEST.txt"

missing=0
for label in "${RUN_LABELS[@]}"; do
  if [ ! -d "$OUT_ROOT/$label" ] || ! find "$OUT_ROOT/$label" -type f | grep -q .; then
    warn "expected run missing or empty in bundle: $label"
    missing=1
  fi
done

log "creating tarball: $BUNDLE_PATH"
tar -czf "$BUNDLE_PATH" -C "$OUT_ROOT" .

log "verifying bundle listing"
LISTING="$TMP/bundle_listing.txt"
tar -tzf "$BUNDLE_PATH" > "$LISTING"

for label in "${RUN_LABELS[@]}"; do
  if grep -q "$label" "$LISTING"; then
    echo "[ok] bundle contains $label"
  else
    echo "[missing] bundle lacks $label" >&2
  fi
done

du -sh "$BUNDLE_PATH"
echo
echo "Upload this file:"
echo "$BUNDLE_PATH"

if [ "$missing" -ne 0 ]; then
  echo
  warn "One or more expected runs were missing/empty. Upload is still possible, but analysis may be partial."
fi
