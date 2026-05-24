# Adversarial repair runbook

This runbook explains how to repair AddSent/AddOneSent robustness metrics without retraining.

## Why this repair exists

`data/qa/addsent.jsonl` and `data/qa/addonesent.jsonl` contain mixed original and adversarial rows. All-row AddSent/AddOneSent EM/F1 can be retained as legacy comparability numbers, but the primary robustness tables must use adversarial-only and paired metrics.

Expected row counts are:

| Evalset | All rows | Original rows | Adversarial rows |
|---|---:|---:|---:|
| AddSent | 3,560 | 1,000 | 2,560 |
| AddOneSent | 1,787 | 1,000 | 787 |

The repair classifies rows whose `id` contains `-high-conf-` as adversarial. The base ID is the prefix before `-high-conf-`.

## Running the repair

From a workspace that still has row-level predictions:

```bash
scripts/run_adversarial_repair_all.sh
```

If predictions are missing but checkpoints exist, run eval-only regeneration:

```bash
REGENERATE_MISSING=1 scripts/run_adversarial_repair_all.sh
```

This reruns `run.py --do_eval` only. It does not train.

## Outputs

The repair package contains:

- `metrics/split_count_audit.csv`
- `metrics/prediction_count_audit.csv`
- `metrics/split_schema_audit.md`
- `metrics/seed_level_metrics_by_split.csv`
- `metrics/main_table_adversarial_only.csv`
- `metrics/main_table_original_only.csv`
- `metrics/main_table_all_row_legacy.csv`
- `metrics/paired_robustness_table.csv`
- `metrics/seed_matched_delta_by_split.csv`
- `metrics/random_draw_distribution_by_split.csv`
- `metrics/selected_condition_win_counts_by_split.csv`
- `metrics/missing_or_failed_runs.csv`
- `predictions/MANIFEST.csv`
- compressed row-level prediction evidence under `predictions/`

## Blockers

Stop and inspect `metrics/missing_or_failed_runs.csv` if predictions cannot be joined to dataset rows or if both predictions and checkpoints are absent.
