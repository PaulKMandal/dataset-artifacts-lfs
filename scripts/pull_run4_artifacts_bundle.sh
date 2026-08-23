#!/usr/bin/env bash
set -Eeuo pipefail

# Pull compact Run 4 artifact branches from GitHub and bundle them for review/upload.
# Run from anywhere inside the dataset-artifacts repo.
#
# Defaults:
#   REPO_URL:    current repo's origin URL
#   OUTPUT_DIR:  <repo>/run4_artifacts_pull
#   BUNDLE_DIR:  <repo>/bundles
#   PULL_LFS:    0  (set PULL_LFS=1 to run git lfs pull in cloned artifact branches)
#
# Output:
#   bundles/cartography_run4_artifacts_bundle_<timestamp>.tar.gz

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
fail() { printf '[error] %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || fail "git not found"
command -v rsync >/dev/null || fail "rsync not found"
command -v tar >/dev/null || fail "tar not found"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not inside a git repo"
cd "$REPO_ROOT"

REPO_URL="${REPO_URL:-$(git config --get remote.origin.url)}"
[ -n "$REPO_URL" ] || fail "could not determine REPO_URL; set REPO_URL=..."

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/run4_artifacts_pull}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/bundles}"
BUNDLE_PATH="${BUNDLE_PATH:-$BUNDLE_DIR/cartography_run4_artifacts_bundle_${STAMP}.tar.gz}"
PULL_LFS="${PULL_LFS:-0}"

LABELS=(
  run4_electra_small
  run4_bert_base
)

BRANCHES=(
  artifacts/run4_electra_small
  artifacts/run4_bert_base
)

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$BUNDLE_DIR"

log "repo root:   $REPO_ROOT"
log "repo url:    $REPO_URL"
log "output dir:  $OUTPUT_DIR"
log "bundle path: $BUNDLE_PATH"

clone_and_copy_branch() {
  local label="$1"
  local branch="$2"
  local clone_dir="$TMP/$label"
  local dest="$OUTPUT_DIR/$label"
  local file_count

  log "checking branch: $branch"
  git ls-remote --exit-code --heads "$REPO_URL" "$branch" >/dev/null 2>&1 \
    || fail "remote branch not found: $branch"

  log "cloning $branch"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch "$branch" "$REPO_URL" "$clone_dir"

  if [ "$PULL_LFS" = "1" ] && command -v git-lfs >/dev/null 2>&1; then
    log "PULL_LFS=1, pulling LFS objects for $branch"
    git -C "$clone_dir" lfs pull || warn "git lfs pull failed for $branch; continuing with available files"
  fi

  log "preview for $branch"
  find "$clone_dir" -maxdepth 5 -mindepth 1 \
    -not -path "$clone_dir/.git*" \
    | sed "s#^$clone_dir#  #" \
    | head -150 || true

  mkdir -p "$dest"

  rsync -a \
    --exclude='.git/' \
    --exclude='.git/**' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.direnv/' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='checkpoint-*/' \
    "$clone_dir/" "$dest/"

  file_count="$(find "$dest" -type f | wc -l | tr -d ' ')"
  [ "$file_count" -gt 0 ] || fail "$branch copied zero files"

  if ! find "$dest" -type f \( \
      -name 'main_table.csv' -o \
      -name 'seed_level_metrics.csv' -o \
      -name 'eval_metrics.json' -o \
      -name 'ARTIFACT_MANIFEST.txt' -o \
      -name 'table_audit.md' \
    \) | grep -q .; then
    warn "$label does not contain expected metrics/audit markers; it will still be bundled"
  fi

  {
    echo "label=$label"
    echo "branch=$branch"
    echo "repo_url=$REPO_URL"
    echo "pulled_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=$(git -C "$clone_dir" rev-parse HEAD)"
    echo "file_count=$file_count"
    echo
    echo "files:"
    find "$dest" -type f | sed "s#^$dest/##" | sort
  } > "$dest/PULL_MANIFEST.txt"

  log "$label: copied $file_count files"
}

for i in "${!LABELS[@]}"; do
  clone_and_copy_branch "${LABELS[$i]}" "${BRANCHES[$i]}"
done

log "writing bundle manifest"
{
  echo "bundle_created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_url=$REPO_URL"
  echo "repo_root=$REPO_ROOT"
  echo "output_dir=$OUTPUT_DIR"
  echo
  echo "source_branches:"
  for i in "${!LABELS[@]}"; do
    echo "  ${LABELS[$i]} ${BRANCHES[$i]}"
  done
  echo
  echo "included_files:"
  find "$OUTPUT_DIR" -type f | sed "s#^$OUTPUT_DIR/##" | sort
  echo
  echo "sizes:"
  du -ah "$OUTPUT_DIR" | sort -h
} > "$OUTPUT_DIR/RUN4_BUNDLE_MANIFEST.txt"

log "creating tarball"
tar -czf "$BUNDLE_PATH" -C "$OUTPUT_DIR" .

log "verifying bundle contains both runs"
tar -tzf "$BUNDLE_PATH" | grep -q 'run4_electra_small/' || fail "bundle is missing run4_electra_small"
tar -tzf "$BUNDLE_PATH" | grep -q 'run4_bert_base/' || fail "bundle is missing run4_bert_base"

log "bundle complete"
du -sh "$BUNDLE_PATH"
printf '\nUpload this file:\n%s\n' "$BUNDLE_PATH"
