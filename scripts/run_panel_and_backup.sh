#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
RUN_NUMBER="${2:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
MODEL_SLUG="${3:?Usage: scripts/run_panel_and_backup.sh CONFIG RUN_NUMBER MODEL_SLUG}"
BASE_BRANCH="${BASE_BRANCH:-working}"
BRANCH="run${RUN_NUMBER}_${MODEL_SLUG}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree has uncommitted changes. Commit/stash before launching automatic branch backup." >&2
  git status --short
  exit 1
fi

git fetch origin --prune

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
elif git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git switch --track "origin/$BRANCH"
else
  if git show-ref --verify --quiet "refs/remotes/origin/$BASE_BRANCH"; then
    git switch -c "$BRANCH" "origin/$BASE_BRANCH"
  else
    git switch -c "$BRANCH"
  fi
  git push -u origin "$BRANCH"
fi

RESULTS_DIR="$(python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('$CONFIG').read_text())
print(cfg['panel']['results_dir'])
PY
)"

echo "Running panel config: $CONFIG"
echo "Branch: $BRANCH"
echo "Results: $RESULTS_DIR"

scripts/run_full_panel.sh "$CONFIG"
scripts/finalize_panel_artifacts.sh "$RESULTS_DIR"

COMMIT_MSG="Add Run $RUN_NUMBER $MODEL_SLUG review artifacts" \
  scripts/backup_panel_review_to_git.sh "$RESULTS_DIR" "$BRANCH"
