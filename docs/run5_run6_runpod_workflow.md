# Run 5 / Run 6 RunPod workflow

Run 4 is the current ELECTRA-small + BERT-base pair. The next experiments are:

- Run 5: `run5_electra_base` and `run5_roberta_base`.
- Run 6: `run6_distilbert` and `run6_deberta_v3_base`.

Each model should run on its own branch named `runX_model`, for example
`run5_electra_base`. The branch helper creates/pushes that branch before the run
and pushes review artifacts after completion.

## Run a panel with automatic review backup

From the configured RunPod/Nix shell:

```bash
scripts/run_panel_and_backup.sh configs/panel.run5_electra_base.yaml 5 electra_base
scripts/run_panel_and_backup.sh configs/panel.run5_roberta_base.yaml 5 roberta_base
scripts/run_panel_and_backup.sh configs/panel.run6_distilbert.yaml 6 distilbert
scripts/run_panel_and_backup.sh configs/panel.run6_deberta_v3.yaml 6 deberta_v3_base
```

The helper backs up a compact review subset under `github_backups/<panel>/` and
pushes it to the run branch. By default it excludes raw predictions and model
weights so Git/LFS backup stays fast.

To include prediction JSONL files in the backup, set:

```bash
INCLUDE_PREDICTIONS=1 scripts/run_panel_and_backup.sh configs/panel.run5_electra_base.yaml 5 electra_base
```

## Back up an already-finished panel

```bash
scripts/backup_runpod_review_results.sh results/round3e_electra_small run4_electra_small
scripts/backup_runpod_review_results.sh results/round3e_bert_base run4_bert_base
```

## Regenerate figures/tables only

```bash
scripts/finalize_panel_artifacts.sh results/round3e_electra_small
scripts/finalize_panel_artifacts.sh results/round3e_bert_base
```

The existing cartography step already produces cartography plots for the source
seed-42 run. The finalization helper adds aggregate metric figures from
`main_table.csv` and `seed_level_metrics.csv`.
