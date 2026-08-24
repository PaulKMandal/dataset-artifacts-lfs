#!/usr/bin/env python3
"""Run the staged two-GPU main-track experiment suite.

The historical panel runner remains unchanged. This runner adds matched
span-vs-sentence-classification experiments, within-family capacity controls,
source-map stability, equal-update controls, proxy/consensus maps, coverage
constraints, broader OOD evaluation, and a fail-fast resumable scheduler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PRINT_LOCK = threading.Lock()
HASH_LOCK = threading.Lock()
FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainSpec:
    run_id: str
    stage: str
    family: str
    task: str
    model_key: str
    model_name: str
    train_subset: str
    subset_fraction: float
    subset_draw_id: int | None
    seed: int
    train_budget_type: str
    train_data: str
    output_dir: str
    selection_map: str
    eval_profile: str
    max_steps: int | None = None
    save_dynamics: bool = False
    smoke: bool = False


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with HASH_LOCK:
        if key not in FILE_HASH_CACHE:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            FILE_HASH_CACHE[key] = "sha256:" + digest.hexdigest()
        return FILE_HASH_CACHE[key]


def sha256_if_exists(path: str | Path) -> str | None:
    resolved = Path(path)
    return sha256_file(resolved) if resolved.exists() else None


def code_hashes(*relative_paths: str) -> dict[str, str | None]:
    return {
        relative_path: sha256_if_exists(REPO_ROOT / relative_path)
        for relative_path in relative_paths
    }


def fraction_label(fraction: float) -> str:
    return f"{fraction:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"suite", "models", "tasks", "training", "cartography", "design", "data"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing top-level sections: {sorted(missing)}")
    return config


def config_hash(config: dict[str, Any]) -> str:
    return canonical_hash(config)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"FAILED: {exc}"


def write_environment(results_dir: Path, config: dict[str, Any]) -> None:
    logs = results_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = [
        f"created_at_utc={now_utc()}",
        f"config_hash={config_hash(config)}",
        f"python={sys.version}",
        f"executable={sys.executable}",
        "",
        "git_head=" + git_value(["git", "rev-parse", "HEAD"]),
        "git_status:",
        git_value(["git", "status", "--short"]),
        "",
        "nvidia_smi:",
        git_value(["nvidia-smi", "-L"]),
    ]
    for key in sorted(os.environ):
        if key.startswith(("CUDA", "HF_", "TRANSFORMERS", "DATASET_ARTIFACTS", "MAINTRACK")):
            sensitive = any(
                marker in key.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "KEY")
            )
            value = "<redacted>" if sensitive else os.environ[key]
            environment.append(f"{key}={value}")
    (logs / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")


class StatusTracker:
    """Atomic human-readable state plus append-only events."""

    def __init__(self, results_dir: Path, config: dict[str, Any], dry_run: bool = False):
        self.dry_run = dry_run
        self.root = results_dir / "status"
        self.state_path = self.root / "status.json"
        self.events_path = self.root / "events.jsonl"
        self.lock = threading.Lock()
        self.config_hash = config_hash(config)
        self.state = {
            "suite": config["suite"]["name"],
            "config_hash": self.config_hash,
            "updated_at_utc": now_utc(),
            "jobs": {},
        }
        if self.state_path.exists() and not dry_run:
            prior = json.loads(self.state_path.read_text(encoding="utf-8"))
            if prior.get("config_hash") == self.config_hash:
                self.state = prior
        if not dry_run:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_state()

    def _write_state(self) -> None:
        self.state["updated_at_utc"] = now_utc()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.state_path)

    def register(self, job_id: str, **metadata: Any) -> None:
        if self.dry_run:
            return
        with self.lock:
            existing = self.state["jobs"].get(job_id, {})
            status = existing.get("status", "pending")
            self.state["jobs"][job_id] = {**metadata, **existing, "status": status}
            self._write_state()

    def transition(self, job_id: str, status: str, **details: Any) -> None:
        if self.dry_run:
            return
        event = {
            "at_utc": now_utc(),
            "config_hash": self.config_hash,
            "job_id": job_id,
            "status": status,
            **details,
        }
        with self.lock:
            current = self.state["jobs"].setdefault(job_id, {})
            current.update(details)
            current["status"] = status
            current["updated_at_utc"] = event["at_utc"]
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            self._write_state()


def run_cmd(
    args: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    gpu_id: str | None = None,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    prefix = ""
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} "
    display = prefix + shlex.join(args)
    with PRINT_LOCK:
        print(f"$ {display}", flush=True)
        if not dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{now_utc()}] $ {display}\n")
    if dry_run:
        return
    started = time.monotonic()
    try:
        subprocess.run(args, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        with PRINT_LOCK, log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{now_utc()}] FAILED exit={exc.returncode} elapsed={time.monotonic() - started:.1f}s\n"
            )
        raise
    with PRINT_LOCK, log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] completed elapsed={time.monotonic() - started:.1f}s\n")


def task_config(config: dict[str, Any], task: str) -> dict[str, Any]:
    if task not in config["tasks"]:
        raise ValueError(f"Unknown task: {task}")
    return config["tasks"][task]


def model_profile(config: dict[str, Any], model_key: str, task: str) -> dict[str, Any]:
    model = config["models"][model_key]
    profile = dict(config["training"]["defaults"])
    profile.update(model.get("defaults", {}))
    profile.update(model["tasks"].get(task, {}))
    return profile


def primary_definition(config: dict[str, Any], task: str) -> str:
    return config["cartography"]["primary_by_task"][task]


def source_run_id(task: str, model_key: str, seed: int) -> str:
    return f"source__{task}__{model_key}__full__seed{seed}"


def source_spec(config: dict[str, Any], task: str, model_key: str, seed: int) -> TrainSpec:
    results_dir = Path(config["suite"]["results_dir"])
    run_id = source_run_id(task, model_key, seed)
    return TrainSpec(
        run_id=run_id,
        stage="sources",
        family="full_baseline",
        task=task,
        model_key=model_key,
        model_name=config["models"][model_key]["name_or_path"],
        train_subset="full",
        subset_fraction=1.0,
        subset_draw_id=None,
        seed=seed,
        train_budget_type="same_epochs",
        train_data=config["tasks"][task]["train_data"],
        output_dir=str(results_dir / "runs" / run_id),
        selection_map="full",
        eval_profile="extended",
        save_dynamics=True,
    )


def all_source_specs(config: dict[str, Any]) -> list[TrainSpec]:
    return core_source_specs(config) + pilot_source_specs(config)


def core_source_specs(config: dict[str, Any]) -> list[TrainSpec]:
    design = config["design"]
    specs = []
    for model_key in design["core_models"]:
        for task in design["tasks"]:
            for seed in design["source_seeds"]:
                specs.append(source_spec(config, task, model_key, int(seed)))
    return specs


def pilot_source_specs(config: dict[str, Any]) -> list[TrainSpec]:
    design = config["design"]
    specs = []
    for model_key in design.get("pilot_models", []):
        for task in design["tasks"]:
            specs.append(source_spec(config, task, model_key, int(design["source_seed"])))
    return specs


def train_hash(config: dict[str, Any], spec: TrainSpec) -> str:
    payload = {
        "task": spec.task,
        "model_key": spec.model_key,
        "model_name": spec.model_name,
        "train_subset": spec.train_subset,
        "subset_fraction": spec.subset_fraction,
        "subset_draw_id": spec.subset_draw_id,
        "seed": spec.seed,
        "train_budget_type": spec.train_budget_type,
        "train_data": spec.train_data,
        "train_data_sha256": sha256_if_exists(spec.train_data),
        "max_steps": spec.max_steps,
        "save_dynamics": spec.save_dynamics,
        "smoke": spec.smoke,
        "task_config": task_config(config, spec.task),
        "profile": model_profile(config, spec.model_key, spec.task),
        "code": code_hashes("run.py", "helpers.py", "qa_metrics.py", "pyproject.toml", "uv.lock"),
    }
    return canonical_hash(payload)


def train_success_path(spec: TrainSpec) -> Path:
    return Path(spec.output_dir) / "_TRAIN_SUCCESS.json"


def train_artifact_paths(spec: TrainSpec) -> list[Path] | None:
    output_dir = Path(spec.output_dir)
    required = [
        output_dir / "maintrack_train_spec.json",
        output_dir / "config.json",
        output_dir / "train_metrics.json",
        output_dir / "trainer_state.json",
        output_dir / "training_args.bin",
        output_dir / "run_manifest.json",
        output_dir / "tokenizer_config.json",
    ]
    model_candidates = [output_dir / "model.safetensors", output_dir / "pytorch_model.bin"]
    tokenizer_candidates = [output_dir / "tokenizer.json", output_dir / "vocab.txt"]
    model_path = next((path for path in model_candidates if path.exists()), None)
    tokenizer_path = next((path for path in tokenizer_candidates if path.exists()), None)
    if model_path is None or tokenizer_path is None or not all(path.exists() for path in required):
        return None
    required.extend([model_path, tokenizer_path])
    if spec.save_dynamics:
        dynamics_paths = sorted(output_dir.glob("training_dynamics*.jsonl"))
        if not dynamics_paths:
            return None
        required.extend(dynamics_paths)
    return required


def train_complete(config: dict[str, Any], spec: TrainSpec) -> bool:
    marker = train_success_path(spec)
    artifacts = train_artifact_paths(spec)
    if not marker.exists() or artifacts is None:
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected_sizes = payload.get("artifact_sizes", {})
    return payload.get("train_hash") == train_hash(config, spec) and all(
        expected_sizes.get(path.name) == path.stat().st_size for path in artifacts
    )


def train_args(config: dict[str, Any], spec: TrainSpec) -> list[str]:
    profile = model_profile(config, spec.model_key, spec.task)
    task = task_config(config, spec.task)
    max_length = int(profile["max_length"])
    batch_size = int(profile["train_batch_size"])
    args = [
        sys.executable,
        "run.py",
        "--do_train",
        "--task",
        task["runner_task"],
        "--dataset",
        spec.train_data,
        "--model",
        spec.model_name,
        "--output_dir",
        spec.output_dir,
        "--overwrite_output_dir",
        "--max_length",
        str(max_length),
        "--per_device_train_batch_size",
        str(batch_size),
        "--per_device_eval_batch_size",
        str(profile["eval_batch_size"]),
        "--gradient_accumulation_steps",
        str(profile.get("gradient_accumulation_steps", 1)),
        "--learning_rate",
        str(profile["learning_rate"]),
        "--warmup_ratio",
        str(profile.get("warmup_ratio", 0.0)),
        "--weight_decay",
        str(profile.get("weight_decay", 0.0)),
        "--num_train_epochs",
        str(1 if spec.smoke else profile["num_train_epochs"]),
        "--save_only_final_model",
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
        "--dataloader_num_workers",
        str(profile.get("dataloader_num_workers", 2)),
    ]
    if task["runner_task"] in {"classification", "nli"}:
        args.extend(
            [
                "--num_labels",
                str(task["num_labels"]),
                "--text_a_column",
                task["text_a_column"],
                "--text_b_column",
                task["text_b_column"],
                "--label_column",
                task.get("label_column", "label"),
            ]
        )
    max_steps = 2 if spec.smoke else spec.max_steps
    if max_steps is not None:
        args.extend(["--max_steps", str(max_steps)])
    if spec.save_dynamics:
        args.append("--save_dynamics")
    if profile.get("fp16", True):
        args.append("--fp16")
    if profile.get("gradient_checkpointing", False):
        args.append("--gradient_checkpointing")
    return args


def write_success_marker(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_train(
    config: dict[str, Any],
    spec: TrainSpec,
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_id: str | None,
) -> None:
    job_id = f"train:{spec.run_id}"
    tracker.register(job_id, kind="train", stage=spec.stage, run_id=spec.run_id, gpu_id=None)
    if train_complete(config, spec):
        tracker.transition(job_id, "complete", resumed=True)
        with PRINT_LOCK:
            print(f"[skip train] {spec.run_id}", flush=True)
        return
    output_dir = Path(spec.output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "maintrack_train_spec.json").write_text(
            json.dumps(asdict(spec), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    tracker.transition(job_id, "running", gpu_id=gpu_id, started_at_utc=now_utc())
    try:
        run_cmd(train_args(config, spec), log_path=log_path, dry_run=dry_run, gpu_id=gpu_id)
        if dry_run:
            return
        artifacts = train_artifact_paths(spec)
        if artifacts is None:
            raise RuntimeError(f"Training finished without required artifacts for {spec.run_id}")
        write_success_marker(
            train_success_path(spec),
            {
                "run_id": spec.run_id,
                "train_hash": train_hash(config, spec),
                "artifact_sizes": {path.name: path.stat().st_size for path in artifacts},
                "completed_at_utc": now_utc(),
            },
        )
        tracker.transition(job_id, "complete", gpu_id=gpu_id, completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(
            job_id,
            "failed",
            gpu_id=gpu_id,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


def evalsets_for(config: dict[str, Any], spec: TrainSpec) -> list[tuple[str, str]]:
    task = task_config(config, spec.task)
    names = task["eval_profiles"][spec.eval_profile]
    return [(name, task["evalsets"][name]) for name in names]


def eval_hash(config: dict[str, Any], spec: TrainSpec, evalset: str, path: str) -> str:
    return canonical_hash(
        {
            "train_hash": train_hash(config, spec),
            "evalset": evalset,
            "dataset": path,
            "dataset_sha256": sha256_if_exists(path),
            "profile": model_profile(config, spec.model_key, spec.task),
        }
    )


def eval_args(config: dict[str, Any], spec: TrainSpec, eval_path: str, eval_out: Path) -> list[str]:
    profile = model_profile(config, spec.model_key, spec.task)
    task = task_config(config, spec.task)
    max_length = int(profile["max_length"])
    args = [
        sys.executable,
        "run.py",
        "--do_eval",
        "--task",
        task["runner_task"],
        "--dataset",
        eval_path,
        "--model",
        spec.output_dir,
        "--output_dir",
        str(eval_out),
        "--overwrite_output_dir",
        "--max_length",
        str(max_length),
        "--per_device_eval_batch_size",
        str(profile["eval_batch_size"]),
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
        "--dataloader_num_workers",
        str(profile.get("dataloader_num_workers", 2)),
    ]
    if task["runner_task"] in {"classification", "nli"}:
        args.extend(
            [
                "--num_labels",
                str(task["num_labels"]),
                "--text_a_column",
                task["text_a_column"],
                "--text_b_column",
                task["text_b_column"],
                "--label_column",
                task.get("label_column", "label"),
            ]
        )
    if profile.get("fp16_eval", profile.get("fp16", True)):
        args.append("--fp16")
    return args


def read_jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def normalize_eval_metrics(
    config: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    eval_out: Path,
    predictions_dest: Path,
) -> dict[str, Any]:
    raw = json.loads((eval_out / "eval_metrics.json").read_text(encoding="utf-8"))
    row = {
        "run_id": f"{spec.run_id}__{evalset}",
        "train_run_id": spec.run_id,
        "stage": spec.stage,
        "family": spec.family,
        "task": spec.task,
        "model_key": spec.model_key,
        "model": spec.model_name,
        "train_subset": spec.train_subset,
        "subset_fraction": spec.subset_fraction,
        "subset_draw_id": spec.subset_draw_id,
        "seed": spec.seed,
        "train_budget_type": spec.train_budget_type,
        "max_steps": spec.max_steps,
        "selection_map": spec.selection_map,
        "eval_profile": spec.eval_profile,
        "evalset": evalset,
        "num_train_examples": read_jsonl_count(Path(spec.train_data)),
        "num_eval_examples": read_jsonl_count(Path(eval_path)),
        "train_dataset_path": spec.train_data,
        "train_dataset_hash": sha256_file(Path(spec.train_data)),
        "dataset_path": eval_path,
        "dataset_hash": sha256_file(Path(eval_path)),
        "predictions_path": str(predictions_dest),
        "predictions_hash": sha256_file(predictions_dest),
        "eval_hash": eval_hash(config, spec, evalset, eval_path),
        "created_at_utc": now_utc(),
    }
    if spec.task == "qa":
        row["exact_match"] = raw.get("eval_exact_match")
        row["f1"] = raw.get("eval_f1")
        if row["exact_match"] is None or row["f1"] is None:
            raise ValueError(f"Missing QA metrics for {spec.run_id} on {evalset}")
    else:
        accuracy = raw.get("eval_accuracy")
        if accuracy is None:
            raise ValueError(f"Missing classification accuracy for {spec.run_id} on {evalset}")
        row["accuracy"] = 100.0 * float(accuracy)
    return row


def run_eval(
    config: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_id: str | None,
) -> None:
    results_dir = Path(config["suite"]["results_dir"])
    eval_id = f"{spec.run_id}__{evalset}"
    job_id = f"eval:{eval_id}"
    eval_out = results_dir / "evals" / eval_id
    metrics_out = results_dir / "metrics" / "raw" / f"{eval_id}.json"
    predictions_dest = results_dir / "predictions" / f"{eval_id}.jsonl"
    tracker.register(job_id, kind="eval", stage=spec.stage, run_id=spec.run_id, evalset=evalset)
    if metrics_out.exists() and predictions_dest.exists():
        prior = json.loads(metrics_out.read_text(encoding="utf-8"))
        if prior.get("eval_hash") == eval_hash(config, spec, evalset, eval_path) and prior.get(
            "predictions_hash"
        ) == sha256_file(predictions_dest):
            tracker.transition(job_id, "complete", resumed=True)
            with PRINT_LOCK:
                print(f"[skip eval] {eval_id}", flush=True)
            return
    tracker.transition(job_id, "running", gpu_id=gpu_id, started_at_utc=now_utc())
    try:
        if not dry_run:
            eval_out.mkdir(parents=True, exist_ok=True)
        run_cmd(
            eval_args(config, spec, eval_path, eval_out),
            log_path=log_path,
            dry_run=dry_run,
            gpu_id=gpu_id,
        )
        if dry_run:
            return
        raw_predictions = eval_out / "eval_predictions.jsonl"
        if not (eval_out / "eval_metrics.json").exists() or not raw_predictions.exists():
            raise RuntimeError(f"Evaluation finished without metrics/predictions for {eval_id}")
        predictions_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_predictions, predictions_dest)
        normalized = normalize_eval_metrics(
            config, spec, evalset, eval_path, eval_out, predictions_dest
        )
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        write_success_marker(metrics_out, normalized)
        tracker.transition(job_id, "complete", gpu_id=gpu_id, completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(
            job_id,
            "failed",
            gpu_id=gpu_id,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


def run_train_and_evals(
    config: dict[str, Any],
    spec: TrainSpec,
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_id: str | None,
    evaluate: bool = True,
) -> None:
    run_train(config, spec, tracker=tracker, log_path=log_path, dry_run=dry_run, gpu_id=gpu_id)
    if evaluate:
        for evalset, eval_path in evalsets_for(config, spec):
            run_eval(
                config,
                spec,
                evalset,
                eval_path,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
                gpu_id=gpu_id,
            )


def run_parallel_specs(
    config: dict[str, Any],
    specs: list[TrainSpec],
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_ids: list[str],
    evaluate: bool,
) -> None:
    if dry_run:
        for spec in specs:
            run_train_and_evals(
                config,
                spec,
                tracker=tracker,
                log_path=log_path,
                dry_run=True,
                gpu_id=gpu_ids[0] if gpu_ids else None,
                evaluate=evaluate,
            )
        return
    if not gpu_ids:
        raise ValueError("Non-dry experiment stages require at least one GPU id")

    work: queue.Queue[TrainSpec] = queue.Queue()
    for spec in specs:
        work.put(spec)
    stop = threading.Event()
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while not stop.is_set():
            try:
                spec = work.get_nowait()
            except queue.Empty:
                return
            try:
                if stop.is_set():
                    return
                with PRINT_LOCK:
                    print(f"[gpu {gpu_id}] {spec.stage}: {spec.run_id}", flush=True)
                run_train_and_evals(
                    config,
                    spec,
                    tracker=tracker,
                    log_path=log_path,
                    dry_run=False,
                    gpu_id=gpu_id,
                    evaluate=evaluate,
                )
            except BaseException as exc:
                with failure_lock:
                    if not failures:
                        failures.append(exc)
                stop.set()
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=False) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


def required_data_paths(config: dict[str, Any]) -> list[Path]:
    paths = []
    for task in config["tasks"].values():
        paths.append(Path(task["train_data"]))
        paths.extend(Path(path) for path in task["evalsets"].values())
    return paths


def data_completion_hash(config: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "data": config["data"],
            "tasks": config["tasks"],
            "code": code_hashes(
                "scripts/materialize_qa_data.py",
                "scripts/materialize_sentence_classification.py",
                "qa_metrics.py",
            ),
        }
    )


def data_marker_is_valid(config: dict[str, Any], marker: Path) -> bool:
    if not marker.exists() or not all(path.exists() for path in required_data_paths(config)):
        return False
    prior = json.loads(marker.read_text(encoding="utf-8"))
    if prior.get("data_hash") != data_completion_hash(config):
        return False
    expected_files = prior.get("files", {})
    return all(
        expected_files.get(str(path)) == sha256_file(path) for path in required_data_paths(config)
    )


def ensure_data(
    config: dict[str, Any], *, tracker: StatusTracker, log_path: Path, dry_run: bool
) -> None:
    job_id = "setup:data"
    tracker.register(job_id, kind="setup", stage="data")
    data_marker = Path(config["suite"]["results_dir"]) / "status" / "DATA_SUCCESS.json"
    if data_marker_is_valid(config, data_marker):
        tracker.transition(job_id, "complete", resumed=True)
        print("[skip data] validated data marker exists", flush=True)
        return
    tracker.transition(job_id, "running", started_at_utc=now_utc())
    data = config["data"]
    qa_args = [
        sys.executable,
        "scripts/materialize_qa_data.py",
        "--out-dir",
        data["qa_dir"],
        "--include-ood",
        "--mrqa-configs",
        *data["mrqa_configs"],
    ]
    for key, flag in [
        ("squad_train_json", "--squad-train-json"),
        ("squad_dev_json", "--squad-dev-json"),
        ("addsent_json", "--addsent-json"),
        ("addonesent_json", "--addonesent-json"),
        ("adversarialqa_json", "--adversarialqa-json"),
    ]:
        if data.get(key):
            qa_args.extend([flag, str(data[key])])
    try:
        run_cmd(qa_args, log_path=log_path, dry_run=dry_run)
        classification_args = [
            sys.executable,
            "scripts/materialize_sentence_classification.py",
            "--train-qa",
            config["tasks"]["qa"]["train_data"],
            "--out-dir",
            data["classification_dir"],
            "--seed",
            str(data.get("classification_seed", 1729)),
        ]
        for name, path in config["tasks"]["qa"]["evalsets"].items():
            classification_args.extend(["--evalset", f"{name}={path}"])
        run_cmd(classification_args, log_path=log_path, dry_run=dry_run)
        if not dry_run:
            missing = [str(path) for path in required_data_paths(config) if not path.exists()]
            if missing:
                raise RuntimeError(f"Data materialization left missing paths: {missing}")
            write_success_marker(
                data_marker,
                {
                    "data_hash": data_completion_hash(config),
                    "completed_at_utc": now_utc(),
                    "files": {str(path): sha256_file(path) for path in required_data_paths(config)},
                },
            )
        tracker.transition(job_id, "complete", completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(job_id, "failed", error=repr(exc), traceback=traceback.format_exc())
        raise


def map_id(task: str, model_key: str, seed: int, definition: str) -> str:
    return f"{task}__{model_key}__seed{seed}__{definition}"


def map_csv(config: dict[str, Any], identifier: str) -> Path:
    return (
        Path(config["suite"]["results_dir"]) / "cartography" / identifier / "cartography_scores.csv"
    )


def map_definitions_for_source(
    config: dict[str, Any], spec: TrainSpec
) -> list[tuple[str, dict[str, Any]]]:
    definitions = config["cartography"]["definitions_by_task"][spec.task]
    primary = primary_definition(config, spec.task)
    if spec.seed != int(config["design"]["source_seed"]):
        return [(primary, definitions[primary])]
    return list(definitions.items())


def cartography_completion_hash(
    config: dict[str, Any], spec: TrainSpec, definition: dict[str, Any]
) -> str:
    return canonical_hash(
        {
            "source_train_hash": train_hash(config, spec),
            "definition": definition,
            "code": code_hashes("dynamics.py"),
        }
    )


def cartography_complete(
    config: dict[str, Any],
    spec: TrainSpec,
    definition_name: str,
    definition: dict[str, Any],
) -> bool:
    identifier = map_id(spec.task, spec.model_key, spec.seed, definition_name)
    csv_path = map_csv(config, identifier)
    marker = csv_path.parent / "_MAP_SUCCESS.json"
    if not csv_path.exists() or not marker.exists():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return payload.get("map_hash") == cartography_completion_hash(
        config, spec, definition
    ) and payload.get("scores_sha256") == sha256_file(csv_path)


def build_map(
    config: dict[str, Any],
    spec: TrainSpec,
    definition_name: str,
    definition: dict[str, Any],
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
) -> Path:
    identifier = map_id(spec.task, spec.model_key, spec.seed, definition_name)
    job_id = f"map:{identifier}"
    output_dir = map_csv(config, identifier).parent
    tracker.register(job_id, kind="map", stage="maps", map_id=identifier)
    if cartography_complete(config, spec, definition_name, definition):
        tracker.transition(job_id, "complete", resumed=True)
        print(f"[skip map] {identifier}", flush=True)
        return output_dir / "cartography_scores.csv"
    tracker.transition(job_id, "running", started_at_utc=now_utc())
    args = [
        sys.executable,
        "dynamics.py",
        "--td_dir",
        spec.output_dir,
        "--output_dir",
        str(output_dir),
        "--confidence_field",
        definition["field"],
        "--aggregation",
        definition["aggregation"],
        "--seed",
        str(spec.seed),
        "--limit_scatter_samples",
        str(config["cartography"].get("limit_scatter_samples", 20000)),
    ]
    try:
        run_cmd(args, log_path=log_path, dry_run=dry_run)
        if not dry_run and not (output_dir / "cartography_scores.csv").exists():
            raise RuntimeError(f"Missing cartography CSV for {identifier}")
        if not dry_run:
            write_success_marker(
                output_dir / "_MAP_SUCCESS.json",
                {
                    "map_id": identifier,
                    "map_hash": cartography_completion_hash(config, spec, definition),
                    "scores_sha256": sha256_file(output_dir / "cartography_scores.csv"),
                    "completed_at_utc": now_utc(),
                },
            )
        tracker.transition(job_id, "complete", completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(job_id, "failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    return output_dir / "cartography_scores.csv"


def subset_root(config: dict[str, Any], selection_map: str, target_task: str) -> Path:
    return Path(config["data"]["subsets_dir"]) / selection_map / f"for_{target_task}"


def subset_file(
    config: dict[str, Any],
    selection_map: str,
    target_task: str,
    subset: str,
    fraction: float,
    draw_id: int | None = None,
) -> str:
    root = subset_root(config, selection_map, target_task)
    label = fraction_label(fraction)
    if subset == "random":
        if draw_id is None:
            raise ValueError("Random subsets require a draw id")
        return str(root / f"random_frac{label}_draw{draw_id:02d}.jsonl")
    return str(root / f"{subset}_frac{label}.jsonl")


def subset_outputs_exist(manifest: Path) -> bool:
    if not manifest.exists():
        return False
    with manifest.open(encoding="utf-8", newline="") as handle:
        paths = [Path(row["path"]) for row in csv.DictReader(handle)]
    return bool(paths) and all(path.exists() for path in paths)


def planned_subset_files(
    config: dict[str, Any], selection_map: str, target_task: str
) -> list[Path]:
    paths = set()
    for stage in ["core", "capacity", "budget", "ablation"]:
        for spec in stage_specs(config, stage):
            if spec.task == target_task and spec.selection_map == selection_map:
                paths.add(Path(spec.train_data))
    return sorted(paths)


def build_subsets(
    config: dict[str, Any],
    selection_map: str,
    scores_path: Path,
    target_task: str,
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
) -> None:
    root = subset_root(config, selection_map, target_task)
    planned_files = planned_subset_files(config, selection_map, target_task)
    if not planned_files:
        print(f"[skip unused subsets] {selection_map} for {target_task}", flush=True)
        return
    job_id = f"subsets:{selection_map}:for_{target_task}"
    manifest = root / "subset_manifest.csv"
    assignments = root / "subset_assignments.csv"
    marker = root / "_SUBSETS_SUCCESS.json"
    subset_hash = None
    planned_hashes = None
    if all(path.exists() for path in planned_files):
        planned_hashes = {str(path): sha256_file(path) for path in planned_files}
    train_path = Path(config["tasks"][target_task]["train_data"])
    if scores_path.exists() and train_path.exists():
        subset_hash = canonical_hash(
            {
                "scores_sha256": sha256_file(scores_path),
                "train_sha256": sha256_file(train_path),
                "fractions": config["design"]["subset_fractions"],
                "random_draws": config["design"]["random_draws"],
                "random_seed_base": config["design"]["random_seed_base"],
                "coverage_constrained": True,
                "code": code_hashes("scripts/select_qa_subsets.py"),
                "planned_files": [str(path) for path in planned_files],
            }
        )
    tracker.register(
        job_id, kind="subsets", stage="maps", map_id=selection_map, target_task=target_task
    )
    if marker.exists() and manifest.exists() and assignments.exists() and subset_hash is not None:
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if (
            prior.get("subset_hash") == subset_hash
            and prior.get("manifest_sha256") == sha256_file(manifest)
            and prior.get("assignments_sha256") == sha256_file(assignments)
            and planned_hashes is not None
            and prior.get("planned_file_hashes") == planned_hashes
            and subset_outputs_exist(manifest)
        ):
            tracker.transition(job_id, "complete", resumed=True)
            print(f"[skip subsets] {selection_map} for {target_task}", flush=True)
            return
    tracker.transition(job_id, "running", started_at_utc=now_utc())
    args = [
        sys.executable,
        "scripts/select_qa_subsets.py",
        "--train-data",
        config["tasks"][target_task]["train_data"],
        "--cartography-scores",
        str(scores_path),
        "--out-dir",
        str(root),
        "--assignments-out",
        str(assignments),
        "--confidence-definition",
        selection_map,
        "--random-draws",
        str(config["design"]["random_draws"]),
        "--random-seed-base",
        str(config["design"]["random_seed_base"]),
        "--fractions",
        *[str(value) for value in config["design"]["subset_fractions"]],
        "--coverage-constrained",
    ]
    try:
        run_cmd(args, log_path=log_path, dry_run=dry_run)
        if not dry_run and not (
            manifest.exists() and assignments.exists() and subset_outputs_exist(manifest)
        ):
            raise RuntimeError(
                f"Subset materialization incomplete for {selection_map}/{target_task}"
            )
        if not dry_run:
            assert subset_hash is not None
            missing_planned = [str(path) for path in planned_files if not path.exists()]
            if missing_planned:
                raise RuntimeError(
                    f"Subset materialization left missing planned files: {missing_planned}"
                )
            planned_hashes = {str(path): sha256_file(path) for path in planned_files}
            write_success_marker(
                marker,
                {
                    "selection_map": selection_map,
                    "target_task": target_task,
                    "subset_hash": subset_hash,
                    "manifest_sha256": sha256_file(manifest),
                    "assignments_sha256": sha256_file(assignments),
                    "planned_file_hashes": planned_hashes,
                    "completed_at_utc": now_utc(),
                },
            )
        tracker.transition(job_id, "complete", completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(job_id, "failed", error=repr(exc), traceback=traceback.format_exc())
        raise


def build_source_maps_and_subsets(
    config: dict[str, Any],
    sources: list[TrainSpec],
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
) -> None:
    source_seed = int(config["design"]["source_seed"])
    for spec in sources:
        for definition_name, definition in map_definitions_for_source(config, spec):
            path = build_map(
                config,
                spec,
                definition_name,
                definition,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
            )
            if spec.seed != source_seed:
                continue
            identifier = map_id(spec.task, spec.model_key, spec.seed, definition_name)
            targets = [spec.task]
            if definition_name == primary_definition(config, spec.task):
                targets = list(config["design"]["tasks"])
            for target_task in targets:
                build_subsets(
                    config,
                    identifier,
                    path,
                    target_task,
                    tracker=tracker,
                    log_path=log_path,
                    dry_run=dry_run,
                )


def consensus_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return config["cartography"].get("consensus_maps", [])


def consensus_completion_hash(config: dict[str, Any], item: dict[str, Any]) -> str | None:
    source_seed = int(config["design"]["source_seed"])
    inputs = []
    for task, model_key in item["inputs"]:
        identifier = map_id(task, model_key, source_seed, primary_definition(config, task))
        path = map_csv(config, identifier)
        if not path.exists():
            return None
        inputs.append({"map_id": identifier, "sha256": sha256_file(path)})
    return canonical_hash(
        {
            "spec": item,
            "inputs": inputs,
            "code": code_hashes("scripts/combine_cartography_maps.py", "dynamics.py"),
        }
    )


def build_consensus_maps(
    config: dict[str, Any], *, tracker: StatusTracker, log_path: Path, dry_run: bool
) -> None:
    source_seed = int(config["design"]["source_seed"])
    for item in consensus_specs(config):
        identifier = item["id"]
        output = map_csv(config, identifier)
        marker = output.parent / "_CONSENSUS_SUCCESS.json"
        expected_hash = consensus_completion_hash(config, item)
        job_id = f"consensus:{identifier}"
        tracker.register(job_id, kind="consensus", stage="maps", map_id=identifier)
        complete = False
        if output.exists() and marker.exists() and expected_hash is not None:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            complete = prior.get("consensus_hash") == expected_hash and prior.get(
                "scores_sha256"
            ) == sha256_file(output)
        if complete:
            tracker.transition(job_id, "complete", resumed=True)
        else:
            args = [sys.executable, "scripts/combine_cartography_maps.py"]
            for task, model_key in item["inputs"]:
                source_map = map_id(
                    task,
                    model_key,
                    source_seed,
                    primary_definition(config, task),
                )
                args.extend(["--map", f"{source_map}={map_csv(config, source_map)}"])
            args.extend(["--out", str(output)])
            tracker.transition(job_id, "running", started_at_utc=now_utc())
            try:
                run_cmd(args, log_path=log_path, dry_run=dry_run)
                if not dry_run and not output.exists():
                    raise RuntimeError(f"Consensus map missing: {identifier}")
                if not dry_run:
                    expected_hash = consensus_completion_hash(config, item)
                    assert expected_hash is not None
                    write_success_marker(
                        marker,
                        {
                            "map_id": identifier,
                            "consensus_hash": expected_hash,
                            "scores_sha256": sha256_file(output),
                            "completed_at_utc": now_utc(),
                        },
                    )
                tracker.transition(job_id, "complete", completed_at_utc=now_utc())
            except Exception as exc:
                tracker.transition(
                    job_id, "failed", error=repr(exc), traceback=traceback.format_exc()
                )
                raise
        for target_task in item["target_tasks"]:
            build_subsets(
                config,
                identifier,
                output,
                target_task,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
            )


def full_steps(config: dict[str, Any], task: str, model_key: str) -> int:
    source = source_spec(config, task, model_key, int(config["design"]["source_seed"]))
    state_path = Path(source.output_dir) / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("global_step", 0)) > 0:
            return int(state["global_step"])
    return int(model_profile(config, model_key, task)["same_steps_fallback"])


def make_run_id(
    family: str,
    task: str,
    model_key: str,
    selection_map: str,
    subset: str,
    fraction: float,
    seed: int,
    budget: str,
    draw_id: int | None,
) -> str:
    map_tag = sanitize(selection_map)
    if len(map_tag) > 42:
        map_tag = map_tag[:30] + "-" + canonical_hash(selection_map)[:10]
    parts = [family, task, model_key, f"map-{map_tag}", subset, f"frac{fraction_label(fraction)}"]
    if draw_id is not None:
        parts.append(f"draw{draw_id:02d}")
    parts.extend([f"seed{seed}", budget])
    return "__".join(parts)


def selected_spec(
    config: dict[str, Any],
    *,
    stage: str,
    family: str,
    task: str,
    model_key: str,
    selection_map: str,
    subset: str,
    fraction: float,
    seed: int,
    budget: str = "same_epochs",
    draw_id: int | None = None,
    eval_profile: str = "core",
) -> TrainSpec:
    run_id = make_run_id(
        family, task, model_key, selection_map, subset, fraction, seed, budget, draw_id
    )
    max_steps = full_steps(config, task, model_key) if budget == "same_steps" else None
    return TrainSpec(
        run_id=run_id,
        stage=stage,
        family=family,
        task=task,
        model_key=model_key,
        model_name=config["models"][model_key]["name_or_path"],
        train_subset=subset,
        subset_fraction=fraction,
        subset_draw_id=draw_id,
        seed=seed,
        train_budget_type=budget,
        train_data=subset_file(config, selection_map, task, subset, fraction, draw_id=draw_id),
        output_dir=str(Path(config["suite"]["results_dir"]) / "runs" / run_id),
        selection_map=selection_map,
        eval_profile=eval_profile,
        max_steps=max_steps,
    )


def append_unique(specs: dict[str, TrainSpec], spec: TrainSpec) -> None:
    prior = specs.get(spec.run_id)
    if prior is not None and prior != spec:
        raise ValueError(f"Run-id collision with different specs: {spec.run_id}")
    specs[spec.run_id] = spec


def mixed_random_design(config: dict[str, Any]) -> list[tuple[int, int]]:
    seeds = [int(seed) for seed in config["design"]["training_seeds"]]
    draws = int(config["design"]["random_draws"])
    pairs = [(0, seed) for seed in seeds]
    pairs.extend((draw, int(config["design"]["source_seed"])) for draw in range(1, draws))
    return pairs


def compact_random_design(config: dict[str, Any]) -> list[tuple[int, int]]:
    seeds = [int(seed) for seed in config["design"]["training_seeds"]]
    draws = min(3, int(config["design"]["random_draws"]))
    pairs = [(0, seed) for seed in seeds]
    pairs.extend((draw, int(config["design"]["source_seed"])) for draw in range(1, draws))
    return pairs


def self_map(config: dict[str, Any], task: str, model_key: str) -> str:
    return map_id(
        task,
        model_key,
        int(config["design"]["source_seed"]),
        primary_definition(config, task),
    )


def build_core_specs(config: dict[str, Any]) -> list[TrainSpec]:
    specs: dict[str, TrainSpec] = {}
    seeds = [int(seed) for seed in config["design"]["training_seeds"]]
    fraction = float(config["design"]["primary_fraction"])
    for model_key in config["design"]["core_models"]:
        for task in config["design"]["tasks"]:
            selection_map = self_map(config, task, model_key)
            for seed in config["design"]["source_seeds"]:
                append_unique(
                    specs,
                    replace(source_spec(config, task, model_key, int(seed)), stage="core"),
                )
            for seed in seeds:
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="primary",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="ambiguous",
                        fraction=fraction,
                        seed=seed,
                        eval_profile="extended",
                    ),
                )
            for draw_id, seed in mixed_random_design(config):
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="primary",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="random",
                        fraction=fraction,
                        seed=seed,
                        draw_id=draw_id,
                        eval_profile="extended",
                    ),
                )
            for seed in seeds:
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="equal_updates",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="ambiguous",
                        fraction=fraction,
                        seed=seed,
                        budget="same_steps",
                    ),
                )
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="coverage_constrained",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="ambiguous_coverage",
                        fraction=fraction,
                        seed=seed,
                    ),
                )
            for draw_id, seed in compact_random_design(config):
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="equal_updates",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="random",
                        fraction=fraction,
                        seed=seed,
                        budget="same_steps",
                        draw_id=draw_id,
                    ),
                )
            for subset in ["easy", "hard"]:
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="core",
                        family="equal_updates",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset=subset,
                        fraction=fraction,
                        seed=int(config["design"]["source_seed"]),
                        budget="same_steps",
                    ),
                )
    return list(specs.values())


def consensus_for_task(config: dict[str, Any], task: str) -> str:
    for item in consensus_specs(config):
        if item.get("role") == "within_task" and item.get("task") == task:
            return item["id"]
    raise ValueError(f"No within-task consensus map configured for {task}")


def cross_task_consensus(config: dict[str, Any]) -> str:
    for item in consensus_specs(config):
        if item.get("role") == "cross_task":
            return item["id"]
    raise ValueError("No cross-task consensus map is configured")


def other_task(config: dict[str, Any], task: str) -> str:
    others = [candidate for candidate in config["design"]["tasks"] if candidate != task]
    if len(others) != 1:
        raise ValueError("Cross-task proxy design currently requires exactly two tasks")
    return others[0]


def build_capacity_specs(config: dict[str, Any]) -> list[TrainSpec]:
    specs: dict[str, TrainSpec] = {}
    source_seed = int(config["design"]["source_seed"])
    seeds = [int(seed) for seed in config["design"]["training_seeds"]]
    fraction = float(config["design"]["primary_fraction"])
    small_model = config["design"]["proxy_source_model"]

    for model_key in config["design"]["core_models"]:
        if model_key == small_model:
            continue
        for task in config["design"]["tasks"]:
            proxy_maps = {
                "small_proxy": self_map(config, task, small_model),
                "cross_task_proxy": self_map(config, other_task(config, task), model_key),
                "consensus_proxy": consensus_for_task(config, task),
                "cross_consensus_proxy": cross_task_consensus(config),
            }
            for family, selection_map in proxy_maps.items():
                for seed in seeds:
                    append_unique(
                        specs,
                        selected_spec(
                            config,
                            stage="capacity",
                            family=family,
                            task=task,
                            model_key=model_key,
                            selection_map=selection_map,
                            subset="ambiguous",
                            fraction=fraction,
                            seed=seed,
                            eval_profile="extended",
                        ),
                    )

    for model_key in config["design"].get("pilot_models", []):
        for task in config["design"]["tasks"]:
            append_unique(
                specs,
                replace(source_spec(config, task, model_key, source_seed), stage="capacity"),
            )
            maps = {
                "self": self_map(config, task, model_key),
                "small_proxy": self_map(config, task, small_model),
                "consensus_proxy": consensus_for_task(config, task),
                "cross_consensus_proxy": cross_task_consensus(config),
            }
            for family, selection_map in maps.items():
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="capacity",
                        family=family,
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset="ambiguous",
                        fraction=fraction,
                        seed=source_seed,
                        eval_profile="extended",
                    ),
                )
            for draw_id in range(int(config["design"].get("pilot_random_draws", 3))):
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="capacity",
                        family="large_pilot",
                        task=task,
                        model_key=model_key,
                        selection_map=self_map(config, task, model_key),
                        subset="random",
                        fraction=fraction,
                        seed=source_seed,
                        draw_id=draw_id,
                        eval_profile="extended",
                    ),
                )
    return list(specs.values())


def build_budget_specs(config: dict[str, Any]) -> list[TrainSpec]:
    specs: dict[str, TrainSpec] = {}
    source_seed = int(config["design"]["source_seed"])
    for item in config["design"].get("budget_curves", []):
        model_key = item["model"]
        task = item["task"]
        selection_map = self_map(config, task, model_key)
        for fraction in item["fractions"]:
            fraction = float(fraction)
            for subset in item["subsets"]:
                draws = range(int(item.get("random_draws", 1))) if subset == "random" else [None]
                for draw_id in draws:
                    append_unique(
                        specs,
                        selected_spec(
                            config,
                            stage="budget",
                            family="budget_curve",
                            task=task,
                            model_key=model_key,
                            selection_map=selection_map,
                            subset=subset,
                            fraction=fraction,
                            seed=source_seed,
                            draw_id=draw_id,
                        ),
                    )
    return list(specs.values())


def build_ablation_specs(config: dict[str, Any]) -> list[TrainSpec]:
    specs: dict[str, TrainSpec] = {}
    source_seed = int(config["design"]["source_seed"])
    fraction = float(config["design"]["primary_fraction"])
    for item in config["design"].get("cartography_ablations", []):
        task = item["task"]
        model_key = item["model"]
        for definition in item["definitions"]:
            selection_map = map_id(task, model_key, source_seed, definition)
            for subset in item["subsets"]:
                append_unique(
                    specs,
                    selected_spec(
                        config,
                        stage="ablation",
                        family="cartography_ablation",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset=subset,
                        fraction=fraction,
                        seed=source_seed,
                    ),
                )
    return list(specs.values())


def stage_specs(config: dict[str, Any], stage: str) -> list[TrainSpec]:
    builders = {
        "sources": core_source_specs,
        "core": build_core_specs,
        "capacity": build_capacity_specs,
        "budget": build_budget_specs,
        "ablation": build_ablation_specs,
    }
    if stage not in builders:
        return []
    return builders[stage](config)


def write_job_manifest(config: dict[str, Any]) -> Path:
    results_dir = Path(config["suite"]["results_dir"])
    output = results_dir / "status" / "job_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    specs = []
    for stage in ["sources", "core", "capacity", "budget", "ablation"]:
        specs.extend(stage_specs(config, stage))
    unique = {spec.run_id: spec for spec in specs}
    rows = [asdict(spec) for spec in unique.values()]
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return output


def write_smoke_jsonl(source: Path, destination: Path, limit: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        source.open("r", encoding="utf-8") as input_handle,
        destination.open("w", encoding="utf-8") as output_handle,
    ):
        for line in input_handle:
            if not line.strip():
                continue
            output_handle.write(line)
            written += 1
            if written >= limit:
                break
    if written < min(limit, 2):
        raise ValueError(f"Smoke source {source} has only {written} usable rows")


def smoke_marker(config: dict[str, Any]) -> Path:
    return Path(config["suite"]["results_dir"]) / "status" / "SMOKE_SUCCESS.json"


def configured_smoke_specs(config: dict[str, Any]) -> list[TrainSpec]:
    smoke_root = Path(config["suite"]["results_dir"]) / "smoke"
    specs = []
    model_keys = [
        *config["design"]["core_models"],
        *config["design"].get("pilot_models", []),
    ]
    for task in config["design"]["tasks"]:
        for model_key in model_keys:
            run_id = f"smoke__{task}__{model_key}"
            specs.append(
                TrainSpec(
                    run_id=run_id,
                    stage="smoke",
                    family="smoke",
                    task=task,
                    model_key=model_key,
                    model_name=config["models"][model_key]["name_or_path"],
                    train_subset="smoke",
                    subset_fraction=1.0,
                    subset_draw_id=None,
                    seed=int(config["design"]["source_seed"]),
                    train_budget_type="smoke",
                    train_data=str(smoke_root / "data" / task / "train.jsonl"),
                    output_dir=str(smoke_root / "runs" / run_id),
                    selection_map="smoke",
                    eval_profile="core",
                    max_steps=2,
                    save_dynamics=True,
                    smoke=True,
                )
            )
    return specs


def smoke_evalsets(config: dict[str, Any], spec: TrainSpec) -> list[str]:
    if spec.model_key == config["design"]["proxy_source_model"]:
        return list(config["tasks"][spec.task]["eval_profiles"]["extended"])
    return [config["tasks"][spec.task]["eval_profiles"]["core"][0]]


def smoke_subset_dir(config: dict[str, Any], spec: TrainSpec, definition_name: str) -> Path:
    return (
        Path(config["suite"]["results_dir"])
        / "smoke"
        / "subsets"
        / spec.task
        / spec.model_key
        / definition_name
    )


def smoke_selected_spec(config: dict[str, Any], spec: TrainSpec) -> TrainSpec:
    selected_id = f"smoke_selected__{spec.task}__{spec.model_key}"
    primary_root = smoke_subset_dir(config, spec, primary_definition(config, spec.task))
    return replace(
        spec,
        run_id=selected_id,
        family="smoke_selected",
        train_subset="ambiguous_coverage",
        subset_fraction=0.5,
        train_data=str(primary_root / "ambiguous_coverage_frac0p5.jsonl"),
        output_dir=str(Path(config["suite"]["results_dir"]) / "smoke" / "runs" / selected_id),
        save_dynamics=False,
    )


def smoke_completion_hash(config: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "config_hash": config_hash(config),
            "data": {str(path): sha256_file(path) for path in required_data_paths(config)},
            "code": code_hashes(
                "run.py",
                "helpers.py",
                "dynamics.py",
                "scripts/select_qa_subsets.py",
                "scripts/run_maintrack_suite.py",
            ),
        }
    )


def smoke_auxiliary_paths(config: dict[str, Any], specs: list[TrainSpec]) -> list[Path]:
    smoke_root = Path(config["suite"]["results_dir"]) / "smoke"
    paths = []
    for spec in specs:
        for definition_name in config["cartography"]["definitions_by_task"][spec.task]:
            subset_dir = smoke_subset_dir(config, spec, definition_name)
            paths.extend(
                [
                    subset_dir / "subset_manifest.csv",
                    subset_dir / "subset_assignments.csv",
                ]
            )
        for evalset in smoke_evalsets(config, spec):
            eval_dir = smoke_root / "evals" / spec.run_id / evalset
            paths.extend([eval_dir / "eval_metrics.json", eval_dir / "eval_predictions.jsonl"])
    return paths


def smoke_complete(config: dict[str, Any], marker: Path) -> bool:
    if not marker.exists() or not all(path.exists() for path in required_data_paths(config)):
        return False
    specs = configured_smoke_specs(config)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("smoke_hash") != smoke_completion_hash(config):
        return False
    for spec in specs:
        if not train_complete(config, spec):
            return False
        for definition_name, definition in config["cartography"]["definitions_by_task"][
            spec.task
        ].items():
            if not cartography_complete(
                config, spec, f"smoke_{definition_name}", definition
            ) or not subset_outputs_exist(
                smoke_subset_dir(config, spec, definition_name) / "subset_manifest.csv"
            ):
                return False
        if spec.model_key == config["design"]["proxy_source_model"] and not train_complete(
            config, smoke_selected_spec(config, spec)
        ):
            return False
    expected_hashes = payload.get("auxiliary_hashes", {})
    auxiliary_paths = smoke_auxiliary_paths(config, specs)
    return all(
        path.exists() and expected_hashes.get(str(path)) == sha256_file(path)
        for path in auxiliary_paths
    )


def run_smoke(
    config: dict[str, Any],
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_ids: list[str],
) -> None:
    marker = smoke_marker(config)
    if not dry_run and smoke_complete(config, marker):
        print("[skip smoke] validated smoke marker exists", flush=True)
        return
    smoke_root = Path(config["suite"]["results_dir"]) / "smoke"
    specs = configured_smoke_specs(config)
    for task in config["design"]["tasks"]:
        train_path = smoke_root / "data" / task / "train.jsonl"
        if not dry_run:
            write_smoke_jsonl(
                Path(config["tasks"][task]["train_data"]),
                train_path,
                int(config["suite"]["smoke_train_examples"]),
            )
            for evalset in config["tasks"][task]["eval_profiles"]["extended"]:
                write_smoke_jsonl(
                    Path(config["tasks"][task]["evalsets"][evalset]),
                    smoke_root / "data" / task / "eval" / f"{evalset}.jsonl",
                    int(config["suite"]["smoke_eval_examples"]),
                )

    for index, spec in enumerate(specs):
        gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
        run_train(
            config,
            spec,
            tracker=tracker,
            log_path=log_path,
            dry_run=dry_run,
            gpu_id=gpu_id,
        )
        primary_scores = None
        primary_subset_root = None
        for definition_name, definition in config["cartography"]["definitions_by_task"][
            spec.task
        ].items():
            scores = build_map(
                config,
                spec,
                f"smoke_{definition_name}",
                definition,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
            )
            smoke_subset_root = smoke_subset_dir(config, spec, definition_name)
            args = [
                sys.executable,
                "scripts/select_qa_subsets.py",
                "--train-data",
                spec.train_data,
                "--cartography-scores",
                str(scores),
                "--out-dir",
                str(smoke_subset_root),
                "--fractions",
                "0.5",
                "--random-draws",
                "1",
                "--coverage-constrained",
            ]
            run_cmd(args, log_path=log_path, dry_run=dry_run)
            if not dry_run and not (smoke_subset_root / "subset_manifest.csv").exists():
                raise RuntimeError(f"Smoke subset selection failed for {spec.run_id}")
            if definition_name == primary_definition(config, spec.task):
                primary_scores = scores
                primary_subset_root = smoke_subset_root

        for evalset in smoke_evalsets(config, spec):
            eval_out = smoke_root / "evals" / spec.run_id / evalset
            run_cmd(
                eval_args(
                    config,
                    spec,
                    str(smoke_root / "data" / spec.task / "eval" / f"{evalset}.jsonl"),
                    eval_out,
                ),
                log_path=log_path,
                dry_run=dry_run,
                gpu_id=gpu_id,
            )
            if not dry_run and not (eval_out / "eval_predictions.jsonl").exists():
                raise RuntimeError(f"Smoke evaluation failed for {spec.run_id}/{evalset}")

        if spec.model_key == config["design"]["proxy_source_model"]:
            assert primary_scores is not None and primary_subset_root is not None
            selected_spec = smoke_selected_spec(config, spec)
            run_train(
                config,
                selected_spec,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
                gpu_id=gpu_id,
            )
    if not dry_run:
        auxiliary_paths = smoke_auxiliary_paths(config, specs)
        write_success_marker(
            marker,
            {
                "config_hash": config_hash(config),
                "smoke_hash": smoke_completion_hash(config),
                "git_head": git_value(["git", "rev-parse", "HEAD"]),
                "completed_at_utc": now_utc(),
                "models": [spec.model_key for spec in specs],
                "tasks": [spec.task for spec in specs],
                "auxiliary_hashes": {str(path): sha256_file(path) for path in auxiliary_paths},
            },
        )


def run_map_stability(
    config: dict[str, Any], *, tracker: StatusTracker, log_path: Path, dry_run: bool
) -> None:
    results_dir = Path(config["suite"]["results_dir"])
    output = results_dir / "analysis" / "map_stability.csv"
    job_id = "analysis:map_stability"
    tracker.register(job_id, kind="analysis", stage="analysis")
    args = [sys.executable, "scripts/analyze_map_stability.py"]
    for model_key in config["design"]["core_models"]:
        for task in config["design"]["tasks"]:
            for seed in config["design"]["source_seeds"]:
                identifier = map_id(task, model_key, int(seed), primary_definition(config, task))
                args.extend(["--map", f"{identifier}={map_csv(config, identifier)}"])
    args.extend(
        [
            "--fractions",
            *[str(value) for value in config["design"]["subset_fractions"]],
            "--out",
            str(output),
        ]
    )
    tracker.transition(job_id, "running", started_at_utc=now_utc())
    try:
        run_cmd(args, log_path=log_path, dry_run=dry_run)
        tracker.transition(job_id, "complete", completed_at_utc=now_utc())
    except Exception as exc:
        tracker.transition(job_id, "failed", error=repr(exc), traceback=traceback.format_exc())
        raise


def run_coverage_analyses(
    config: dict[str, Any], *, tracker: StatusTracker, log_path: Path, dry_run: bool
) -> None:
    results_dir = Path(config["suite"]["results_dir"])
    for model_key in config["design"]["core_models"]:
        for task in config["design"]["tasks"]:
            identifier = self_map(config, task, model_key)
            assignments = subset_root(config, identifier, task) / "subset_assignments.csv"
            output = results_dir / "analysis" / f"coverage__{identifier}.csv"
            job_id = f"analysis:coverage:{identifier}"
            tracker.register(job_id, kind="analysis", stage="analysis")
            tracker.transition(job_id, "running", started_at_utc=now_utc())
            try:
                run_cmd(
                    [
                        sys.executable,
                        "scripts/analyze_subset_coverage.py",
                        "--assignments",
                        str(assignments),
                        "--train-data",
                        config["tasks"][task]["train_data"],
                        "--out",
                        str(output),
                    ],
                    log_path=log_path,
                    dry_run=dry_run,
                )
                tracker.transition(job_id, "complete", completed_at_utc=now_utc())
            except Exception as exc:
                tracker.transition(
                    job_id, "failed", error=repr(exc), traceback=traceback.format_exc()
                )
                raise


def write_default_contrasts(config: dict[str, Any]) -> Path:
    results_dir = Path(config["suite"]["results_dir"])
    output = results_dir / "analysis" / "bootstrap_contrasts.json"
    contrasts = []
    fraction = float(config["design"]["primary_fraction"])
    for model_key in config["design"]["core_models"]:
        for evalset in config["tasks"]["qa"]["eval_profiles"]["extended"]:
            common = {
                "task": "qa",
                "model_key": model_key,
                "family": "primary",
                "subset_fraction": fraction,
                "evalset": evalset,
                "train_budget_type": "same_epochs",
            }
            for metric in ["exact_match", "f1"]:
                contrasts.append(
                    {
                        "name": f"{model_key}_{evalset}_{metric}_ambiguous_minus_random",
                        "metric": metric,
                        "common": common,
                        "a": {"train_subset": "ambiguous"},
                        "b": {"train_subset": "random"},
                    }
                )
        for evalset in config["tasks"]["classification"]["eval_profiles"]["extended"]:
            contrasts.append(
                {
                    "name": f"{model_key}_{evalset}_accuracy_ambiguous_minus_random",
                    "metric": "correct",
                    "common": {
                        "task": "classification",
                        "model_key": model_key,
                        "family": "primary",
                        "subset_fraction": fraction,
                        "evalset": evalset,
                        "train_budget_type": "same_epochs",
                    },
                    "a": {"train_subset": "ambiguous"},
                    "b": {"train_subset": "random"},
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contrasts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_final_analysis(
    config: dict[str, Any], *, tracker: StatusTracker, log_path: Path, dry_run: bool
) -> None:
    results_dir = Path(config["suite"]["results_dir"])
    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_maintrack_metrics.py",
            "--metrics-dir",
            str(results_dir / "metrics" / "raw"),
            "--out-dir",
            str(results_dir / "metrics"),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )
    run_map_stability(config, tracker=tracker, log_path=log_path, dry_run=dry_run)
    run_coverage_analyses(config, tracker=tracker, log_path=log_path, dry_run=dry_run)
    contrasts = write_default_contrasts(config)
    run_cmd(
        [
            sys.executable,
            "scripts/hierarchical_bootstrap.py",
            "--metrics-dir",
            str(results_dir / "metrics" / "raw"),
            "--contrasts",
            str(contrasts),
            "--out",
            str(results_dir / "analysis" / "hierarchical_bootstrap.json"),
            "--n",
            str(config["design"].get("bootstrap_samples", 10000)),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )


def parse_gpu_ids(raw: str | None) -> list[str]:
    if raw is None:
        raw = os.environ.get("MAINTRACK_GPU_IDS", "0,1")
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_stage(
    stage: str,
    config: dict[str, Any],
    *,
    tracker: StatusTracker,
    log_path: Path,
    dry_run: bool,
    gpu_ids: list[str],
) -> None:
    if stage == "data":
        ensure_data(config, tracker=tracker, log_path=log_path, dry_run=dry_run)
        return
    if stage == "smoke":
        run_smoke(
            config,
            tracker=tracker,
            log_path=log_path,
            dry_run=dry_run,
            gpu_ids=gpu_ids,
        )
        return
    if stage == "sources":
        sources = core_source_specs(config)
        run_parallel_specs(
            config,
            sources,
            tracker=tracker,
            log_path=log_path,
            dry_run=dry_run,
            gpu_ids=gpu_ids,
            evaluate=False,
        )
        build_source_maps_and_subsets(
            config,
            sources,
            tracker=tracker,
            log_path=log_path,
            dry_run=dry_run,
        )
        build_consensus_maps(config, tracker=tracker, log_path=log_path, dry_run=dry_run)
        if not dry_run:
            # Equal-update specs can now use the observed full-data optimizer
            # step counts rather than the frozen pre-source fallback.
            write_job_manifest(config)
        return
    if stage in {"core", "capacity", "budget", "ablation"}:
        if stage == "capacity":
            pilots = pilot_source_specs(config)
            run_parallel_specs(
                config,
                pilots,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
                gpu_ids=gpu_ids,
                evaluate=False,
            )
            build_source_maps_and_subsets(
                config,
                pilots,
                tracker=tracker,
                log_path=log_path,
                dry_run=dry_run,
            )
        run_parallel_specs(
            config,
            stage_specs(config, stage),
            tracker=tracker,
            log_path=log_path,
            dry_run=dry_run,
            gpu_ids=gpu_ids,
            evaluate=True,
        )
        return
    if stage == "analysis":
        run_final_analysis(config, tracker=tracker, log_path=log_path, dry_run=dry_run)
        return
    raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maintrack.full.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "data",
            "smoke",
            "sources",
            "core",
            "capacity",
            "budget",
            "ablation",
            "analysis",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-job-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    results_dir = Path(config["suite"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, results_dir / "configs" / config_path.name)
    tracker = StatusTracker(results_dir, config, dry_run=args.dry_run)
    log_path = results_dir / "logs" / "commands.log"
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    manifest = write_job_manifest(config)
    if args.print_job_counts:
        for stage in ["sources", "core", "capacity", "budget", "ablation"]:
            print(f"{stage}: {len(stage_specs(config, stage))}")
        print(f"job manifest: {manifest}")
        if args.stage != "all" or args.dry_run:
            return
    if not args.dry_run:
        write_environment(results_dir, config)

    stages = (
        ["data", "smoke", "sources", "core", "capacity", "budget", "ablation", "analysis"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"\n=== main-track stage: {stage} ===", flush=True)
        run_stage(
            stage,
            config,
            tracker=tracker,
            log_path=log_path,
            dry_run=args.dry_run,
            gpu_ids=gpu_ids,
        )
    print(f"Main-track suite complete: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
