#!/usr/bin/env bash
set -Eeuo pipefail

# Generate compact paper artifacts from a completed panel and push them to a
# separate artifact branch without modifying the branch that ran the experiment.
#
# Usage:
#   scripts/finalize_and_backup_run_artifacts.sh results/round3e_electra_small run4_electra_small
#   scripts/finalize_and_backup_run_artifacts.sh results/round3e_bert_base run4_bert_base
#
# Optional env:
#   BACKUP_BRANCH=artifacts/run4_electra_small
#   MAX_BACKUP_FILE_MB=100
#   REMOTE=origin
#   BACKUP_ROOT=/workspace/artifact_worktrees

RESULTS_DIR="${1:?Usage: $0 RESULTS_DIR RUN_LABEL}"
RUN_LABEL="${2:?Usage: $0 RESULTS_DIR RUN_LABEL}"
REMOTE="${REMOTE:-origin}"
BACKUP_BRANCH="${BACKUP_BRANCH:-artifacts/${RUN_LABEL}}"
BACKUP_ROOT="${BACKUP_ROOT:-/workspace/artifact_worktrees}"
MAX_BACKUP_FILE_MB="${MAX_BACKUP_FILE_MB:-100}"
INCLUDE_PREDICTIONS="${INCLUDE_PREDICTIONS:-0}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

RESULTS_DIR="$(realpath "$RESULTS_DIR")"
MAX_BYTES=$((MAX_BACKUP_FILE_MB * 1024 * 1024))
SAFE_BRANCH_PATH="${BACKUP_BRANCH//\//_}"
WORKTREE_DIR="$BACKUP_ROOT/$SAFE_BRANCH_PATH"
DEST_REL="github_backups/$RUN_LABEL"
DEST_DIR="$WORKTREE_DIR/$DEST_REL"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_optional() {
  log "$*"
  "$@" || true
}

python_bin=".venv/bin/python3"
if [ ! -x "$python_bin" ]; then
  python_bin="$(command -v python3)"
fi

log "Finalizing artifacts for $RUN_LABEL"
log "RESULTS_DIR=$RESULTS_DIR"
log "BACKUP_BRANCH=$BACKUP_BRANCH"
log "MAX_BACKUP_FILE_MB=$MAX_BACKUP_FILE_MB"

test -d "$RESULTS_DIR" || {
  echo "Missing results dir: $RESULTS_DIR" >&2
  exit 1
}

mkdir -p "$RESULTS_DIR/audit" "$RESULTS_DIR/logs" "$RESULTS_DIR/figures" "$RESULTS_DIR/metrics"

{
  echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "run_label=$RUN_LABEL"
  echo "results_dir=$RESULTS_DIR"
  echo "branch=$(git branch --show-current || true)"
  echo "commit=$(git rev-parse HEAD || true)"
  echo
  echo "git_status_short:"
  git status --short || true
  echo
  echo "last_commit:"
  git --no-pager log -1 --oneline --decorate || true
} > "$RESULTS_DIR/logs/repo_snapshot.txt"

if [ -f scripts/audit_current_results.py ]; then
  run_optional "$python_bin" scripts/audit_current_results.py --results-dir "$RESULTS_DIR" --out-dir "$RESULTS_DIR/audit"
fi

if [ -f scripts/audit_evalsets.py ]; then
  run_optional "$python_bin" scripts/audit_evalsets.py --results-dir "$RESULTS_DIR"
fi

if [ -f scripts/aggregate_metrics.py ] && [ -d "$RESULTS_DIR/metrics/raw" ]; then
  agg_args=(--metrics-dir "$RESULTS_DIR/metrics/raw" --out-dir "$RESULTS_DIR/metrics")
  if [ -f "$RESULTS_DIR/audit/eval_split_metrics.csv" ]; then
    agg_args+=(--eval-split-metrics "$RESULTS_DIR/audit/eval_split_metrics.csv")
  fi
  if [ -f "$RESULTS_DIR/audit/paired_robustness_metrics.csv" ]; then
    agg_args+=(--paired-metrics "$RESULTS_DIR/audit/paired_robustness_metrics.csv")
  fi
  run_optional "$python_bin" scripts/aggregate_metrics.py "${agg_args[@]}"
fi

log "Generating compact metric figures"
"$python_bin" - "$RESULTS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    print(f"matplotlib unavailable; skipping figures: {exc}")
    raise SystemExit(0)

root = Path(sys.argv[1])
fig_dir = root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
main_path = root / "metrics" / "main_table.csv"
seed_path = root / "metrics" / "seed_level_metrics.csv"

if not main_path.exists() and not seed_path.exists():
    print("No main_table.csv or seed_level_metrics.csv found; skipping metric figures")
    raise SystemExit(0)

if main_path.exists():
    df = pd.read_csv(main_path)
else:
    seed = pd.read_csv(seed_path)
    group_cols = [c for c in ["train_subset", "subset_protocol", "subset_protocol_norm", "subset_fraction", "train_budget_type", "confidence_definition"] if c in seed.columns]
    metric_cols = [c for c in seed.columns if c.startswith("f1") or c.startswith("exact_match")]
    df = seed.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()

rows = []
for _, row in df.iterrows():
    subset = row.get("train_subset", row.get("subset", "unknown"))
    protocol = row.get("subset_protocol_norm", row.get("subset_protocol", ""))
    frac = row.get("subset_fraction", "")
    budget = row.get("train_budget_type", "")
    conf = row.get("confidence_definition", "")
    label_bits = [str(x) for x in [subset, protocol, frac, budget, conf] if str(x) not in {"", "nan", "None"}]
    condition = " | ".join(label_bits)
    for col in df.columns:
        if col.startswith("f1_mean__"):
            rows.append({"condition": condition, "evalset": col.replace("f1_mean__", ""), "metric": "f1", "value": row[col]})
        elif col.startswith("exact_match_mean__"):
            rows.append({"condition": condition, "evalset": col.replace("exact_match_mean__", ""), "metric": "exact_match", "value": row[col]})
        elif col == "f1_mean" and "evalset" in df.columns:
            rows.append({"condition": condition, "evalset": row["evalset"], "metric": "f1", "value": row[col]})
        elif col == "exact_match_mean" and "evalset" in df.columns:
            rows.append({"condition": condition, "evalset": row["evalset"], "metric": "exact_match", "value": row[col]})

plot_df = pd.DataFrame(rows)
plot_df.to_csv(fig_dir / "panel_plot_rows.csv", index=False)
if plot_df.empty:
    print("No plottable rows found; wrote empty panel_plot_rows.csv")
    raise SystemExit(0)

for evalset in sorted(plot_df["evalset"].dropna().unique()):
    sub = plot_df[(plot_df["evalset"] == evalset) & (plot_df["metric"] == "f1")].dropna(subset=["value"])
    if sub.empty:
        continue
    sub = sub.sort_values("value", ascending=False).head(25)
    plt.figure(figsize=(12, max(5, 0.28 * len(sub))))
    plt.barh(sub["condition"], sub["value"])
    plt.xlabel("F1")
    plt.ylabel("condition")
    plt.title(f"{evalset}: F1 by condition")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / f"f1_by_condition_{evalset}.png", dpi=180)
    plt.close()

adv = plot_df[(plot_df["metric"] == "f1") & (plot_df["evalset"].isin(["addsent", "addonesent"]))].dropna(subset=["value"])
if not adv.empty:
    pivot = adv.pivot_table(index="condition", columns="evalset", values="value", aggfunc="mean")
    pivot["mean_adv_f1"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_adv_f1", ascending=False).head(25)
    plt.figure(figsize=(12, max(5, 0.28 * len(pivot))))
    plt.barh(pivot.index, pivot["mean_adv_f1"])
    plt.xlabel("Mean adversarial F1")
    plt.ylabel("condition")
    plt.title("Adversarial F1 by condition")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "adversarial_f1_by_condition.png", dpi=180)
    plt.close()

random_path = root / "metrics" / "random_subset_distribution.csv"
if random_path.exists():
    rdf = pd.read_csv(random_path)
    value_cols = [c for c in rdf.columns if c.startswith("f1") or c.endswith("f1")]
    for col in value_cols[:6]:
        vals = pd.to_numeric(rdf[col], errors="coerce").dropna()
        if vals.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=20)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"Random subset distribution: {col}")
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in col)
        plt.savefig(fig_dir / f"random_distribution_{safe}.png", dpi=180)
        plt.close()

print(f"Wrote figures to {fig_dir}")
PY

log "Preparing backup worktree"
mkdir -p "$BACKUP_ROOT"
if git worktree list --porcelain | grep -q "worktree $WORKTREE_DIR"; then
  git worktree remove --force "$WORKTREE_DIR" || true
fi
rm -rf "$WORKTREE_DIR"
git worktree add -B "$BACKUP_BRANCH" "$WORKTREE_DIR" HEAD

mkdir -p "$DEST_DIR"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

log "Copying compact artifacts"
rsync -a --prune-empty-dirs \
  --include='/audit/***' \
  --include='/metrics/***' \
  --include='/figures/***' \
  --include='/logs/***' \
  --include='/configs/***' \
  --include='/cartography/' \
  --include='/cartography/*.csv' \
  --include='/cartography/*.json' \
  --include='/cartography/*/' \
  --include='/cartography/*/*.csv' \
  --include='/cartography/*/*.json' \
  --include='/cartography/*/*.png' \
  --include='/runs/' \
  --include='/runs/*/' \
  --include='/runs/*/all_results.json' \
  --include='/runs/*/train_metrics.json' \
  --include='/runs/*/train_results.json' \
  --include='/runs/*/trainer_state.json' \
  --include='/runs/*/run_manifest.json' \
  --include='/evals/' \
  --include='/evals/*/' \
  --include='/evals/*/eval_metrics.json' \
  --include='/evals/*/run_manifest.json' \
  --exclude='*' \
  "$RESULTS_DIR/" "$DEST_DIR/"

if [ "$INCLUDE_PREDICTIONS" = "1" ]; then
  mkdir -p "$DEST_DIR/predictions"
  find "$RESULTS_DIR/predictions" -type f \
    \( -name '*.jsonl' -o -name '*.jsonl.gz' \) \
    -size -"${MAX_BACKUP_FILE_MB}"M -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do cp "$f" "$DEST_DIR/predictions/"; done
fi

log "Removing oversized backup files"
find "$DEST_DIR" -type f -size +"${MAX_BACKUP_FILE_MB}"M -print -delete > "$DEST_DIR/OVERSIZED_FILES_SKIPPED.txt" || true

{
  echo "run_label=$RUN_LABEL"
  echo "source_results_dir=$RESULTS_DIR"
  echo "source_branch=$(git branch --show-current || true)"
  echo "source_commit=$(git rev-parse HEAD || true)"
  echo "backup_branch=$BACKUP_BRANCH"
  echo "created_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "include_predictions=$INCLUDE_PREDICTIONS"
  echo "max_backup_file_mb=$MAX_BACKUP_FILE_MB"
  echo
  echo "file_count=$(find "$DEST_DIR" -type f | wc -l)"
  echo "size=$(du -sh "$DEST_DIR" | awk '{print $1}')"
  echo
  echo "files:"
  find "$DEST_DIR" -type f | sed "s#^$DEST_DIR/##" | sort
} > "$DEST_DIR/ARTIFACT_MANIFEST.txt"

log "Committing backup artifacts"
cd "$WORKTREE_DIR"
git lfs install --skip-smudge || true
git add -f "$DEST_REL"
if git diff --cached --quiet; then
  log "No artifact changes to commit"
else
  git commit -m "Back up ${RUN_LABEL} paper artifacts"
fi

git push -u "$REMOTE" "$BACKUP_BRANCH"
git lfs push "$REMOTE" "$BACKUP_BRANCH" || true

log "Backup complete: branch $BACKUP_BRANCH, path $DEST_REL"
