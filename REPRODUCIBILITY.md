# Reproducing the cartography and adversarial SQuAD experiments

This repository contains the code, configuration, and driver scripts for the EMNLP submission experiments. It does **not** include model weights, Hugging Face datasets, or generated result artifacts. The full run materializes the datasets, trains the source cartography model, creates cartography-based subsets, trains the experiment panel, evaluates each model on SQuAD dev/AddSent/AddOneSent, and writes aggregate tables from saved metric files.

> For the expanded main-track two-V100 revision suite, use
> [`MAINTRACK_EXPERIMENTS.md`](MAINTRACK_EXPERIMENTS.md). The sections below
> preserve the earlier ELECTRA-small panel instructions.

## 1. Requirements

Recommended environment:

- Linux host with Nix flakes enabled.
- Python 3.11 supplied by the Nix shell.
- `uv` for Python dependency management.
- NVIDIA GPU for experiment reproduction. The CUDA dependency set uses PyTorch CUDA 12.1 wheels; the NVIDIA driver must be compatible with CUDA 12.1 runtime libraries.
- Internet access to download the Hugging Face model and datasets, unless local mirrors are configured as described below.

The project has two Nix shells:

- `.#default`: CPU/local checks.
- `.#server`: GPU experiment shell.

## 2. Unpack and enter the repository

```bash
tar -xzf dataset-cartography-squad-anonymized-normalized-repro.tgz
cd dataset-cartography-squad
```

## 3. Local preflight checks

Run these on a CPU machine before launching GPU jobs:

```bash
nix develop .#default
uv sync --frozen --extra cpu --group dev

# Keeps externally installed pytest plugins from affecting this project.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q
uv run --no-sync python -m py_compile \
  run.py helpers.py dynamics.py compare_adversarial.py scripts/*.py tests/*.py

# Verifies that the panel driver expands the configured runs and commands.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python \
  scripts/run_experiment_panel.py \
  --config configs/panel.minimum.yaml \
  --dry-run \
  --limit-runs 2
```

A successful dry run prints commands for dataset materialization, source-model training, cartography generation, subset selection, train/eval runs, aggregation, and validation. It does not train models or download datasets.

The same checks are wrapped in:

```bash
scripts/repro_preflight.sh
```

## 4. GPU environment setup

On the GPU host:

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev

uv run --no-sync python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda runtime', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
PY
```

If `torch.cuda.is_available()` is `False`, fix the GPU driver/library setup before running the panel.

## 5. One-GPU smoke run

This run downloads SQuAD and `google/electra-small-discriminator`, trains on a small slice, writes scalar training dynamics, evaluates on a small validation slice, and converts the dynamics to cartography scores.

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 scripts/run_qa_smoke.sh results/smoke_squad_fast_dynamics
```

Expected smoke outputs:

```text
results/smoke_squad_fast_dynamics/
  eval_metrics.json
  eval_predictions.jsonl
  run_manifest.json
  train_metrics.json
  trainer_state.json
  training_dynamics.jsonl
  cartography/cartography_scores.csv
  cartography/categorized_examples.json
```

The dynamics file should contain scalar fields such as `confidence`, `joint_confidence`, `correctness`, `pred_start`, and `pred_end`. It should not contain full `start_prob` or `end_prob` arrays.

## 6. Dataset materialization

By default, the panel materializes SQuAD through Hugging Face and obtains the
official adversarial SQuAD source files through checksum-pinned local storage:

- SQuAD v1.1 train and validation via `datasets.load_dataset("squad")`.
- AddSent: 4,073,864-byte source with SHA-256
  `40e3602aa5195cdacd03904a9c301ceb17ccf730cc32bd3ab998b66b4401e660`.
- AddOneSent: 1,920,649-byte source with SHA-256
  `50420ac8d8b7547cd3715347c9a276802bc9466328ba0814adce4c20495e2889`.
- MRQA NewsQA, TriviaQA-web, and SearchQA are read directly from the pinned
  official validation archives under `data/sources/mrqa/`. The converter uses
  the release's `detected_answers.char_spans` rather than the lossy answer
  aliases emitted by the `tau/mrqa` adapter.

The source files live under `data/sources/squad_adversarial/`. The main-track
foreground gate first searches the prior server checkout/cache for verified
copies and only then attempts the official CodaLab endpoints. A source transfer,
count mismatch, duplicate ID, or invalid answer span fails before the batch is
started. Materialized output replacement is atomic.

The official TriviaQA-web validation archive contains one malformed row,
`355adac432e64303a0d035784b5078c2`, whose detected answer span points to the
structural `[DOC]` marker. That row is explicitly excluded and recorded in the
dataset manifest. The pinned materializations contain 4,212 NewsQA rows, 7,784
TriviaQA rows, and 16,980 SearchQA rows.

The materialized files and checksums are written under `data/qa/`:

```text
data/qa/squad_train.jsonl
data/qa/squad_dev.jsonl
data/qa/addsent.jsonl
data/qa/addonesent.jsonl
data/qa/dataset_manifest.json
```

To use local frozen dataset copies instead of downloading, edit the `data` section in the panel config and set these paths to local SQuAD-v1-style JSON files:

```yaml
data:
  squad_train_json: /path/to/train-v1.1.json
  squad_dev_json: /path/to/dev-v1.1.json
  addsent_json: /path/to/addsent.json
  addonesent_json: /path/to/addonesent.json
```

The materializer accepts either SQuAD JSON or already flattened JSONL files. It
records both source and materialized hashes in `dataset_manifest.json` for the
adversarial and MRQA sets.

## 7. Reproduce the minimum panel

The minimum panel is intended as a smaller end-to-end check. It uses ELECTRA-small, SQuAD training data, the joint-confidence cartography definition plus the endpoint-confidence ablation, and evaluates every trained model on SQuAD dev, AddSent, and AddOneSent.

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.minimum.yaml
```

For multiple GPUs, use independent workers:

```bash
DATASET_ARTIFACTS_GPU_IDS=0,1,2,3 \
scripts/run_full_panel.sh configs/panel.minimum.yaml --parallel-workers 4
```

The panel is resumable. Existing completed train/eval units are skipped. To force a rerun of one unit, delete its run directory and the corresponding file under `metrics/raw/`. To force the whole panel to rerun, remove the configured `results_dir` and generated subset/data files, or pass `--no-resume` after the config path.

## 8. Reproduce the full panel

The full experiment panel is defined in `configs/panel.full.yaml`. Important settings are recorded directly in that file, including:

- model: `google/electra-small-discriminator`
- max sequence length: `384`
- epochs: `3`
- learning rate: `5e-5`
- train batch size: `32`
- eval batch size: `64`
- cartography source seed: `42`
- primary confidence definition: `joint_confidence`
- evalsets: SQuAD dev, AddSent, AddOneSent

Run:

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.full.yaml
```

Multi-GPU version:

```bash
DATASET_ARTIFACTS_GPU_IDS=0,1,2,3 \
scripts/run_full_panel.sh configs/panel.full.yaml --parallel-workers 4
```

## 9. Main outputs

For the full panel, the configured result root is `results/panel_electra_small/`. For the minimum panel, it is `results/panel_electra_small_minimum/`.

Expected result structure:

```text
results/panel_electra_small/
  configs/                         # config snapshots for each run
  logs/command_log.txt             # every command executed by the panel
  logs/environment.txt             # Python, uv, Nix, CUDA, and relevant env vars
  logs/git_commit.txt              # git commit/status if available
  logs/table_audit.md              # evalset label/count/hash audit
  runs/                            # trained model output dirs
  evals/                           # temporary eval output dirs
  predictions/*.jsonl              # per-example predictions copied from evals
  metrics/raw/*.json               # one normalized metric record per run/evalset
  metrics/main_table.csv           # aggregate table
  metrics/seed_level_metrics.csv   # seed-level table
  metrics/random_subset_distribution.csv
  metrics/confidence_definition_ablation.csv
  metrics/mechanism_features.csv   # optional mechanism analysis output
  metrics/mechanism_univariate.csv # optional mechanism analysis output
  cartography/joint/cartography_scores.csv
  cartography/endpoint/cartography_scores.csv
  cartography/neg_loss/cartography_scores.csv
  cartography/cartography_scores.csv
  cartography/subset_assignments.csv
```

After the panel completes, these validation commands should pass:

```bash
uv run --no-sync python scripts/audit_evalsets.py \
  --results-dir results/panel_electra_small
uv run --no-sync python scripts/validate_results_tree.py \
  --results-dir results/panel_electra_small
```

For the minimum panel, replace `results/panel_electra_small` with `results/panel_electra_small_minimum`.

## 10. Rebuild tables from saved metrics

If all `metrics/raw/*.json` files already exist, aggregate tables can be rebuilt without retraining:

```bash
uv run --no-sync python scripts/aggregate_metrics.py \
  --metrics-dir results/panel_electra_small/metrics/raw \
  --out-dir results/panel_electra_small/metrics
```

## 11. Changing model or local cache paths

To use a local model mirror, edit the panel config:

```yaml
model:
  name_or_path: /path/to/local/electra-small-discriminator
```

Hugging Face cache locations can also be controlled with standard environment variables, for example:

```bash
export HF_HOME=/path/to/hf-cache
export TRANSFORMERS_CACHE=/path/to/hf-cache/transformers
export HF_DATASETS_CACHE=/path/to/hf-cache/datasets
```

## 12. Notes on determinism

All train/eval seeds used by the panel are recorded in the YAML config and copied into `results/*/configs/`. GPU training may still have small nondeterministic variation from CUDA kernels and library versions. For reporting, use the aggregate tables regenerated from the saved `metrics/raw/*.json` files from the final run.
