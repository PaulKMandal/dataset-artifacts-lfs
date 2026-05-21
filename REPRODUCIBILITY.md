# Reproducibility plan

## Environment

- Nix flake provides the system shell: Python 3.11, uv, git, rsync, OpenSSH, and common build libraries.
- uv manages Python dependencies.
- Use `uv sync --extra cpu --group dev` locally.
- Use `uv sync --extra cuda --group dev` on the V100 server.

## Expected commands

Local checks:

```bash
nix develop .#default
uv sync --extra cpu --group dev
uv run pytest
uv run python -m py_compile run.py helpers.py dynamics.py compare_adversarial.py
```

Server smoke test:

```bash
nix develop .#server
uv sync --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
  --do_train --do_eval --task qa --dataset squad \
  --output_dir results/smoke_squad_fast_dynamics \
  --overwrite_output_dir \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 1 \
  --save_only_final_model \
  --save_dynamics \
  --fp16 \
  --report_to none
```

Cartography after training:

```bash
uv run python dynamics.py \
  --td_dir results/smoke_squad_fast_dynamics \
  --output_dir results/smoke_squad_fast_dynamics/cartography
```

## Artifacts to retain

Keep these:

- `eval_metrics.json`
- `eval_predictions.jsonl`
- `training_dynamics.jsonl` or `training_dynamics.rank*.jsonl`
- `cartography/cartography_scores.csv`
- `cartography/categorized_examples.json`
- `cartography/*.png`
- `trainer_state.json`
- command logs and environment logs

Do not copy back full model weights by default unless `PULL_MODELS=1` is set.

## Pass/fail checks

- `--max_length` must affect QA tokenized sequence length.
- `training_dynamics.jsonl` must contain scalar fields, not full `start_prob`/`end_prob` arrays.
- One V100 smoke run should produce non-empty dynamics and eval metrics.
- Cartography script should produce a non-empty `cartography_scores.csv`.

## Full-panel reproduction command

On the GPU server:

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.full.yaml
```

The panel writes normalized metrics to `results/panel_electra_small/metrics/raw/` and regenerates all CSV/JSON tables through `scripts/aggregate_metrics.py`. The full panel is resumable; delete a run directory and its corresponding `metrics/raw/*.json` file to force a rerun.

## Evalset audit

`results/panel_electra_small/logs/table_audit.md` is generated from metrics files and checks the observed SQuAD dev/AddSent/AddOneSent example counts and hashes. Treat a `CHECK_*` status as a reporting blocker until the dataset path and table label are resolved.

## Round-2 reporting artifacts

Before reporting adversarial robustness, run:

```bash
scripts/run_round2_audit.sh results/panel_electra_small
```

Required audit outputs:

- `audit/eval_split_metrics.csv`: all-row, original-only, and adversarial-only EM/F1 from prediction JSONL files.
- `audit/paired_robustness_metrics.csv`: paired original/adversarial metrics for AddSent/AddOneSent base questions.
- `audit/subset_purity_overlap.csv`: ranked-subset purity and overlap checks plus region-pure diagnostics.
- `audit/windowing_audit.csv`: QA overflow-feature counts and non-gold-window availability.
- `logs/table_audit.md`: manuscript table provenance by run, dataset path, count, hash, and split metric.

Paper tables should prioritize adversarial-only and paired AddSent/AddOneSent metrics. The blended all-row metric is acceptable only when explicitly labeled as a legacy mixed-file number.
