import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_maintrack_suite as suite

ROOT = Path(__file__).resolve().parents[1]


def config():
    return suite.load_config(ROOT / "configs" / "maintrack.full.yaml")


def test_full_design_expands_unique_expected_jobs():
    cfg = config()
    expected_counts = {
        "sources": 12,
        "core": 104,
        "capacity": 40,
        "budget": 60,
        "ablation": 15,
    }
    for stage, expected in expected_counts.items():
        specs = suite.stage_specs(cfg, stage)
        assert len(specs) == expected
        assert len({spec.run_id for spec in specs}) == expected


def test_smoke_matrix_covers_every_task_and_model():
    cfg = config()
    specs = suite.configured_smoke_specs(cfg)
    expected_models = set(cfg["design"]["core_models"] + cfg["design"]["pilot_models"])
    assert {(spec.task, spec.model_key) for spec in specs} == {
        (task, model) for task in cfg["design"]["tasks"] for model in expected_models
    }
    assert all(spec.max_steps == 2 for spec in specs)
    proxy_specs = [spec for spec in specs if spec.model_key == cfg["design"]["proxy_source_model"]]
    for spec in proxy_specs:
        assert (
            suite.smoke_evalsets(cfg, spec) == cfg["tasks"][spec.task]["eval_profiles"]["extended"]
        )


def test_equal_update_design_includes_ambiguous_for_every_core_cell():
    cfg = config()
    specs = suite.build_core_specs(cfg)
    for model_key in cfg["design"]["core_models"]:
        for task in cfg["design"]["tasks"]:
            matching = [
                spec
                for spec in specs
                if spec.model_key == model_key
                and spec.task == task
                and spec.family == "equal_updates"
                and spec.train_subset == "ambiguous"
            ]
            assert {spec.seed for spec in matching} == {13, 21, 42}
            assert all(spec.max_steps is not None for spec in matching)


def test_classification_command_is_two_label_matched_pair():
    cfg = config()
    spec = suite.source_spec(cfg, "classification", "electra_base", 42)
    args = suite.train_args(cfg, spec)
    assert args[args.index("--task") + 1] == "classification"
    assert args[args.index("--num_labels") + 1] == "2"
    assert args[args.index("--text_a_column") + 1] == "question"
    assert args[args.index("--text_b_column") + 1] == "candidate_sentence"
    assert args[args.index("--gradient_accumulation_steps") + 1] == "2"


def test_training_completion_hash_ignores_eval_stage_metadata():
    cfg = config()
    original = suite.source_spec(cfg, "qa", "electra_small", 42)
    reused = replace(original, stage="core", eval_profile="core")
    assert suite.train_hash(cfg, original) == suite.train_hash(cfg, reused)


def test_parallel_scheduler_stops_dequeuing_after_first_failure(monkeypatch, tmp_path):
    started = []
    second_started = threading.Event()

    def fake_run(config, spec, **kwargs):
        started.append(spec.run_id)
        if spec.run_id == "job0":
            assert second_started.wait(timeout=2)
            raise RuntimeError("intentional")
        second_started.set()
        time.sleep(0.05)

    monkeypatch.setattr(suite, "run_train_and_evals", fake_run)
    base = suite.TrainSpec(
        run_id="job0",
        stage="test",
        family="test",
        task="qa",
        model_key="m",
        model_name="m",
        train_subset="full",
        subset_fraction=1.0,
        subset_draw_id=None,
        seed=1,
        train_budget_type="same_epochs",
        train_data="train.jsonl",
        output_dir="out",
        selection_map="full",
        eval_profile="core",
    )
    specs = [replace(base, run_id=f"job{index}") for index in range(6)]

    with pytest.raises(RuntimeError, match="intentional"):
        suite.run_parallel_specs(
            {},
            specs,
            tracker=None,
            log_path=tmp_path / "commands.log",
            dry_run=False,
            gpu_ids=["0", "1"],
            evaluate=False,
        )

    assert set(started) == {"job0", "job1"}


def test_training_hash_changes_when_input_bytes_change(tmp_path):
    cfg = config()
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    spec = replace(suite.source_spec(cfg, "qa", "electra_small", 42), train_data=str(train_data))
    before = suite.train_hash(cfg, spec)
    train_data.write_text('{"idx":0,"changed":true}\n', encoding="utf-8")
    after = suite.train_hash(cfg, spec)
    assert before != after


def test_training_completion_rejects_changed_artifact_size(tmp_path):
    cfg = config()
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
        output_dir=str(tmp_path / "run"),
    )
    output = Path(spec.output_dir)
    output.mkdir()
    for name in [
        "maintrack_train_spec.json",
        "config.json",
        "train_metrics.json",
        "trainer_state.json",
        "training_args.bin",
        "run_manifest.json",
        "tokenizer_config.json",
        "model.safetensors",
        "tokenizer.json",
        "training_dynamics.jsonl",
    ]:
        (output / name).write_text(name, encoding="utf-8")
    artifacts = suite.train_artifact_paths(spec)
    assert artifacts is not None
    suite.write_success_marker(
        suite.train_success_path(spec),
        {
            "train_hash": suite.train_hash(cfg, spec),
            "artifact_sizes": {path.name: path.stat().st_size for path in artifacts},
        },
    )
    assert suite.train_complete(cfg, spec)
    (output / "config.json").write_text("changed-size", encoding="utf-8")
    assert not suite.train_complete(cfg, spec)


def test_data_marker_is_invalidated_by_changed_dataset(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text('{"idx":0}\n', encoding="utf-8")
    evaluation.write_text('{"id":"q0"}\n', encoding="utf-8")
    cfg = {
        "data": {"source": "fixture"},
        "tasks": {"qa": {"train_data": str(train), "evalsets": {"dev": str(evaluation)}}},
    }
    marker = tmp_path / "DATA_SUCCESS.json"
    marker.write_text(
        json.dumps(
            {
                "data_hash": suite.data_completion_hash(cfg),
                "files": {
                    str(train): suite.sha256_file(train),
                    str(evaluation): suite.sha256_file(evaluation),
                },
            }
        ),
        encoding="utf-8",
    )
    assert suite.data_marker_is_valid(cfg, marker)
    evaluation.write_text('{"id":"q0","changed":true}\n', encoding="utf-8")
    assert not suite.data_marker_is_valid(cfg, marker)


def test_default_bootstrap_contrasts_cover_both_task_formulations(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    contrasts = json.loads(suite.write_default_contrasts(cfg).read_text(encoding="utf-8"))
    assert len(contrasts) == 42
    assert {item["metric"] for item in contrasts} == {"exact_match", "f1", "correct"}
    assert {item["common"]["task"] for item in contrasts} == {"qa", "classification"}
