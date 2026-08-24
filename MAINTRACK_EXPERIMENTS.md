# Main-track two-V100 experiment suite

`configs/maintrack.full.yaml` is the frozen experiment design for the main-track
revision. It expands to 219 distinct training run IDs and 1,025 evaluation jobs,
plus an eight-run smoke gate. The historical `configs/panel.*.yaml` workflow is
still available and is not modified by this suite.

## Scientific matrix

| Risk or reviewer question | Implemented experiment | Key control |
|---|---|---|
| Is the result specific to span extraction? | Matched SQuAD span QA and two-label question/sentence classification | One classification training row per SQuAD source `idx`; balanced labels; identical source text and examples |
| Does capacity change the conclusion? | ELECTRA small and base for the full core matrix; ELECTRA large pilot | Same model family and tokenizer lineage |
| Is a map an accident of one source seed? | Source maps from seeds 13, 21, and 42 | Pairwise Spearman/Kendall, region agreement, and ambiguous-set Jaccard |
| Is subset training merely cheaper? | Same-epoch and same-optimizer-step runs | Full-run optimizer steps are read from `trainer_state.json`; explicit fallback is frozen in the config |
| Is one random subset a weak baseline? | Five random draws with a mixed seed/draw design | Persistent subset files and draw IDs |
| Is the ambiguous set losing topical or structural coverage? | Coverage-constrained ambiguous selection | Quotas over question type, answer length, and normalized answer position; lexical/title/length coverage analysis |
| Is the cartography definition arbitrary? | QA joint confidence, record weighting, endpoint confidence, negative span loss, and answer-window-only maps; classification epoch/record maps | Same source checkpoint and fixed selectors |
| Can a cheaper map select data for a larger model? | Small-to-base/large proxy maps | Self-map comparison at the same target architecture |
| Are maps task-specific? | Cross-task proxy maps, within-task consensus maps, and cross-task consensus maps | Percentile-rank alignment across maps |
| Does the effect persist across data budgets? | 20%, 33.3%, 50%, 75%, and full-data conditions | Easy/ambiguous/hard/random curves |
| Does the finding extend beyond one adversarial set? | SQuAD dev, AddSent, AddOneSent, AdversarialQA, and MRQA NewsQA/TriviaQA/SearchQA | Frozen JSONL materializations with checksums |
| Are uncertainty claims seed-only? | Hierarchical bootstrap over subset draw, training seed, and evaluation examples | Per-example QA EM/F1 and classification correctness are retained |

QA exact match and F1 are saved together. Classification accuracy is scaled to
percentage points before task-interaction analyses.

## Launch

Run from the repository root on a host with exactly two available V100 GPUs:

```bash
nix develop .#server
scripts/launch_maintrack_tmux.sh configs/maintrack.full.yaml
```

The launcher creates the `cartography-maintrack` tmux session, enables a raw
`pipe-pane` log before starting the runner, and then executes:

1. frozen CUDA dependency synchronization;
2. two-V100, PyTorch/CUDA, test, lint, compile, and dry-expansion preflight;
3. data materialization and checksum validation;
4. a smoke gate using the real max lengths and per-model batch profiles;
5. source, core, capacity, budget, ablation, and analysis stages.

The smoke gate trains every configured model/task pair for two optimizer steps,
derives every cartography definition, materializes every selector schema, checks
every extended OOD dataset with the small model, and trains a selected subset.
The full matrix does not begin unless the gate completes.

Override the session or GPU IDs only explicitly:

```bash
MAINTRACK_TMUX_SESSION=cartography-maintrack \
MAINTRACK_GPU_IDS=0,1 \
scripts/launch_maintrack_tmux.sh configs/maintrack.full.yaml
```

## Monitoring

```bash
tmux attach -t cartography-maintrack

tail -F "$(cat results/maintrack_week/status/tmux_log_path.txt)"

scripts/maintrack_status.sh configs/maintrack.full.yaml --watch 10
```

The tmux log contains everything written to the experiment pane. The structured
status view shows active GPU assignments, counts by stage/status, recent
failures, and live utilization. Additional audit trails are under:

```text
results/maintrack_week/logs/commands.log
results/maintrack_week/status/events.jsonl
results/maintrack_week/status/status.json
results/maintrack_week/status/week_runner_exit_code.txt
results/maintrack_week/status/job_manifest.csv
```

## Failure and resume semantics

Failure handling is deliberately strict:

- shell entry points use `set -euo pipefail`;
- child commands use checked exit codes;
- the first GPU-worker failure stops further dequeueing;
- the other GPU may finish only the atomic train/eval unit already running;
- there are no automatic retries, fallback batch sizes, exception suppression,
  or “continue on error” paths;
- the failing traceback is saved in structured status and the weekly runner
  writes a nonzero exit code.

After inspecting and fixing a failure, rerun the same launcher manually. A unit
is skipped only when its required artifacts and hash-aware success marker match
the current config and input bytes. Partial or stale outputs are not accepted as
complete.

For an intentional single-stage resume after the normal preflight has already
passed:

```bash
uv run --no-sync python scripts/run_maintrack_suite.py \
  --config configs/maintrack.full.yaml \
  --stage core \
  --gpu-ids 0,1
```

## Primary result locations

```text
results/maintrack_week/
  cartography/                 # source, proxy, and consensus maps
  runs/                        # trained checkpoints and hashed train markers
  evals/                       # raw evaluator outputs
  predictions/                 # per-example predictions used by bootstrap
  metrics/raw/                 # normalized run/eval records
  metrics/maintrack_summary.csv
  metrics/task_formulation_interactions.csv
  analysis/map_stability.csv
  analysis/coverage__*.csv
  analysis/hierarchical_bootstrap.json
  status/
  logs/
```

The copied config, environment manifest, Git identity/commit, dataset checksums,
train specifications, and success-marker hashes provide the provenance needed
to audit or reconstruct any reported cell.
