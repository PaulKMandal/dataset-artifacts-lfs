#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for backing up an already-finished RunPod panel.
# Example:
#   scripts/backup_runpod_review_results.sh results/round3e_electra_small run4_electra_small

RESULTS_DIR="${1:?Usage: scripts/backup_runpod_review_results.sh RESULTS_DIR BRANCH}"
BRANCH="${2:?Usage: scripts/backup_runpod_review_results.sh RESULTS_DIR BRANCH}"

git fetch origin --prune
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
elif git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git switch --track "origin/$BRANCH"
else
  git switch -c "$BRANCH"
fi

scripts/finalize_panel_artifacts.sh "$RESULTS_DIR"
COMMIT_MSG="Backup review artifacts for $BRANCH" scripts/backup_panel_review_to_git.sh "$RESULTS_DIR" "$BRANCH"
