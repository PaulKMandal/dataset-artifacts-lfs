import gzip
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import reclaim_maintrack_space as reclaimer
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


def _write_complete_eval(cfg, spec, evalset, eval_path):
    results = Path(cfg["suite"]["results_dir"])
    eval_id = f"{spec.run_id}__{evalset}"
    predictions = results / "predictions" / f"{eval_id}.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text('{"id":"q0"}\n', encoding="utf-8")
    metrics = results / "metrics" / "raw" / f"{eval_id}.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    suite.write_success_marker(
        metrics,
        {
            "eval_hash": suite.eval_hash(cfg, spec, evalset, eval_path),
            "predictions_hash": suite.sha256_file(predictions),
        },
    )


def test_pruned_training_remains_complete_for_same_config(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q0"}\n', encoding="utf-8")
    cfg["tasks"]["qa"]["evalsets"] = {"dev": str(eval_path)}
    cfg["tasks"]["qa"]["eval_profiles"]["core"] = ["dev"]
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
        output_dir=str(Path(cfg["suite"]["results_dir"]) / "runs" / "test"),
        eval_profile="core",
    )
    output = Path(spec.output_dir)
    output.mkdir(parents=True)
    (output / "maintrack_train_spec.json").write_text(
        json.dumps(suite.asdict(spec)), encoding="utf-8"
    )
    for name in [
        "config.json", "train_metrics.json", "trainer_state.json", "training_args.bin",
        "run_manifest.json", "tokenizer_config.json", "tokenizer.json", "training_dynamics.jsonl",
        "model.safetensors",
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
    _write_complete_eval(cfg, spec, "dev", str(eval_path))
    assert suite.prune_model_weights(cfg, spec) > 0
    assert not (output / "model.safetensors").exists()
    assert suite.train_complete(cfg, spec)

    changed = json.loads(json.dumps(cfg))
    changed["suite"]["name"] = "changed-suite"
    assert not suite.train_complete(changed, spec)


def test_reclaim_removes_partial_failed_weight(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    run_dir = Path(cfg["suite"]["results_dir"]) / "runs" / "failed"
    run_dir.mkdir(parents=True)
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        run_id="failed",
        output_dir=str(run_dir),
    )
    (run_dir / "maintrack_train_spec.json").write_text(
        json.dumps(suite.asdict(spec)), encoding="utf-8"
    )
    (run_dir / "model.safetensors").write_bytes(b"x" * 4096)
    report = suite.reclaim_completed_run_space(cfg)
    assert report["partial_weight_files"] == 1
    assert report["partial_weight_bytes"] == 4096
    assert not (run_dir / "model.safetensors").exists()


def test_reclaim_cli_imports_when_executed_directly():
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "reclaim_maintrack_space.py"), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def _legacy_train_hash(cfg, spec):
    code = suite.code_hashes(*suite.TRAIN_CODE_PATHS)
    assert code["run.py"] == suite.ARROW_CACHE_RECOVERY_RUN_PY_SHA256
    code["run.py"] = suite.PRE_ARROW_CACHE_RUN_PY_SHA256
    return suite.train_hash_with_code(cfg, spec, code)


def test_pre_arrow_training_marker_is_resume_compatible(tmp_path):
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
    legacy_hash = _legacy_train_hash(cfg, spec)
    suite.write_success_marker(
        suite.train_success_path(spec),
        {
            "train_hash": legacy_hash,
            "artifact_sizes": {path.name: path.stat().st_size for path in artifacts},
        },
    )
    assert legacy_hash != suite.train_hash(cfg, spec)
    assert legacy_hash in suite.compatible_train_hashes(cfg, spec)
    assert suite.train_complete(cfg, spec)


def test_pre_arrow_eval_marker_is_resume_compatible(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q0"}\n', encoding="utf-8")
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
        output_dir=str(tmp_path / "run"),
        eval_profile="core",
    )
    cfg["tasks"]["qa"]["evalsets"] = {"dev": str(eval_path)}
    cfg["tasks"]["qa"]["eval_profiles"]["core"] = ["dev"]
    results = Path(cfg["suite"]["results_dir"])
    predictions = results / "predictions" / f"{spec.run_id}__dev.jsonl"
    predictions.parent.mkdir(parents=True)
    predictions.write_text('{"id":"q0"}\n', encoding="utf-8")
    metrics = results / "metrics" / "raw" / f"{spec.run_id}__dev.json"
    metrics.parent.mkdir(parents=True)
    legacy_eval_hash = suite.eval_hash_for_train_hash(
        cfg, spec, "dev", str(eval_path), _legacy_train_hash(cfg, spec)
    )
    suite.write_success_marker(
        metrics,
        {
            "eval_hash": legacy_eval_hash,
            "predictions_hash": suite.sha256_file(predictions),
        },
    )
    assert legacy_eval_hash != suite.eval_hash(cfg, spec, "dev", str(eval_path))
    assert suite.eval_complete(cfg, spec, "dev", str(eval_path))


def test_future_run_py_change_does_not_inherit_cache_recovery_compatibility():
    future = "sha256:" + "f" * 64
    assert suite.compatible_run_py_hashes(future) == {future}


def test_pruner_refuses_success_marker_with_modified_artifact(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q0"}\n', encoding="utf-8")
    cfg["tasks"]["qa"]["evalsets"] = {"dev": str(eval_path)}
    cfg["tasks"]["qa"]["eval_profiles"]["core"] = ["dev"]
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
        output_dir=str(Path(cfg["suite"]["results_dir"]) / "runs" / "test"),
        eval_profile="core",
    )
    output = Path(spec.output_dir)
    output.mkdir(parents=True)
    for name in [
        "maintrack_train_spec.json",
        "config.json",
        "train_metrics.json",
        "trainer_state.json",
        "training_args.bin",
        "run_manifest.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "training_dynamics.jsonl",
        "model.safetensors",
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
    _write_complete_eval(cfg, spec, "dev", str(eval_path))
    (output / "trainer_state.json").write_text("modified-after-success", encoding="utf-8")
    assert suite.prune_model_weights(cfg, spec) == 0
    assert (output / "model.safetensors").exists()


def test_pre_arrow_cartography_hash_is_resume_compatible(tmp_path):
    cfg = config()
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
    )
    definition_name, definition = suite.map_definitions_for_source(cfg, spec)[0]
    assert definition_name
    legacy_map_hash = suite.cartography_completion_hash_for_train_hash(
        cfg, spec, definition, _legacy_train_hash(cfg, spec)
    )
    assert legacy_map_hash != suite.cartography_completion_hash(cfg, spec, definition)
    assert legacy_map_hash in suite.compatible_cartography_completion_hashes(
        cfg, spec, definition
    )


def test_pre_arrow_smoke_hash_is_resume_compatible(monkeypatch, tmp_path):
    cfg = config()
    data = tmp_path / "data.jsonl"
    data.write_text('{"idx":0}\n', encoding="utf-8")

    def fake_required_data_paths(_config):
        return [data]

    monkeypatch.setattr(suite, "required_data_paths", fake_required_data_paths)
    legacy_code = suite.code_hashes(*suite.LEGACY_SMOKE_CODE_PATHS)
    assert legacy_code["run.py"] == suite.ARROW_CACHE_RECOVERY_RUN_PY_SHA256
    legacy_code["run.py"] = suite.PRE_ARROW_CACHE_RUN_PY_SHA256
    legacy_code["scripts/run_maintrack_suite.py"] = suite.PRE_RESUME_COMPAT_SUITE_SHA256
    legacy_hash = suite.legacy_smoke_completion_hash_with_code(cfg, legacy_code)
    assert legacy_hash != suite.smoke_completion_hash(cfg)
    assert legacy_hash in suite.compatible_smoke_completion_hashes(cfg)


def test_run_cmd_uses_ephemeral_hf_datasets_cache(monkeypatch, tmp_path):
    scratch_root = tmp_path / "scratch"
    captured = {}

    def fake_run(args, check, env):
        captured["args"] = args
        captured["cache"] = env.get("DATASET_ARTIFACTS_MAP_CACHE_DIR")
        cache = Path(captured["cache"])
        assert cache.parent == scratch_root
        (cache / "sentinel").write_text("scratch", encoding="utf-8")

    monkeypatch.setenv("MAINTRACK_HF_SCRATCH_ROOT", str(scratch_root))
    monkeypatch.setenv("MAINTRACK_HF_SCRATCH_MIN_FREE_GIB", "0")
    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    suite.run_cmd(
        [sys.executable, "run.py", "--help"],
        log_path=tmp_path / "commands.log",
        dry_run=False,
    )

    assert captured["cache"] is not None
    assert not Path(captured["cache"]).exists()
    assert list(scratch_root.iterdir()) == []


def test_completed_predictions_can_be_losslessly_gzipped(tmp_path):
    results = tmp_path / "results"
    predictions = results / "predictions" / "run__dev.jsonl"
    metrics = results / "metrics" / "raw" / "run__dev.json"
    predictions.parent.mkdir(parents=True)
    metrics.parent.mkdir(parents=True)
    original = ('{"id":"q0","context":"' + "repeat-me-" * 200 + '"}\n') * 20
    predictions.write_text(original, encoding="utf-8")
    suite.write_success_marker(
        metrics,
        {
            "eval_hash": "unchanged-scientific-hash",
            "predictions_path": str(predictions),
            "predictions_hash": suite.sha256_file(predictions),
        },
    )

    report = suite.compress_completed_predictions(results)

    compressed = results / "predictions" / "run__dev.jsonl.gz"
    assert report["compressed_prediction_files"] == 1
    assert report["prediction_bytes_freed"] > 0
    assert not predictions.exists()
    assert compressed.exists()
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        assert handle.read() == original
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["eval_hash"] == "unchanged-scientific-hash"
    assert payload["predictions_path"] == str(compressed)
    assert payload["predictions_hash"] == suite.sha256_file(compressed)
    assert payload["predictions_compression"] == "gzip"


def test_eval_complete_accepts_compressed_canonical_predictions(tmp_path):
    cfg = config()
    cfg["suite"]["results_dir"] = str(tmp_path / "results")
    train_data = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_data.write_text('{"idx":0}\n', encoding="utf-8")
    eval_path.write_text('{"id":"q0"}\n', encoding="utf-8")
    cfg["tasks"]["qa"]["evalsets"] = {"dev": str(eval_path)}
    cfg["tasks"]["qa"]["eval_profiles"]["core"] = ["dev"]
    spec = replace(
        suite.source_spec(cfg, "qa", "electra_small", 42),
        train_data=str(train_data),
        output_dir=str(tmp_path / "run"),
        eval_profile="core",
    )
    results = Path(cfg["suite"]["results_dir"])
    eval_id = f"{spec.run_id}__dev"
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"id":"q0","exact_match":1,"f1":1}\n', encoding="utf-8")
    compressed = results / "predictions" / f"{eval_id}.jsonl.gz"
    suite.gzip_copy_atomic(raw, compressed)
    metrics = results / "metrics" / "raw" / f"{eval_id}.json"
    metrics.parent.mkdir(parents=True)
    suite.write_success_marker(
        metrics,
        {
            "eval_hash": suite.eval_hash(cfg, spec, "dev", str(eval_path)),
            "predictions_path": str(compressed),
            "predictions_hash": suite.sha256_file(compressed),
        },
    )

    assert suite.eval_complete(cfg, spec, "dev", str(eval_path))


def test_reclaimer_removes_only_derived_transform_caches(monkeypatch, tmp_path):
    datasets_cache = tmp_path / "datasets"
    json_cache = datasets_cache / "json" / "generated"
    mrqa_cache = datasets_cache / "tau___mrqa" / "newsqa"
    json_cache.mkdir(parents=True)
    mrqa_cache.mkdir(parents=True)
    derived = json_cache / "cache-deadbeef.arrow"
    base_json = json_cache / "json-train.arrow"
    base_mrqa = mrqa_cache / "mrqa-train.arrow"
    derived.write_bytes(b"x" * 4096)
    base_json.write_bytes(b"z" * 2048)
    base_mrqa.write_bytes(b"y" * 1024)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))

    removed_files, removed_bytes = reclaimer.reclaim_persistent_transform_cache()

    assert removed_files == 1
    assert removed_bytes == 4096
    assert not derived.exists()
    assert base_json.exists()
    assert base_mrqa.exists()
