#!/usr/bin/env python3
"""Run the ELECTRA-small cartography/adversarial QA experiment panel.

This is designed to be launched on the GPU server from inside the Nix+uv shell:

    uv run --no-sync python scripts/run_experiment_panel.py --config configs/panel.full.yaml

The script is resumable. It skips train/eval units whose expected metrics already
exist, writes command/environment logs, saves raw predictions, emits normalized
metrics JSON files, and finally regenerates aggregate tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass

class TrainSpec:
    run_id: str
    model_short: str
    model_name: str
    train_subset: str
    subset_fraction: float
    subset_draw_id: int | None
    seed: int
    train_budget_type: str
    train_data: str
    output_dir: str
    confidence_definition: str
    num_train_epochs: float
    max_steps: int | None
    save_dynamics: bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/panel.full.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None, help="Debugging aid; do not use for final tables.")
    return parser.parse_args()

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fraction_label(frac: float) -> str:
    text = f"{frac:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()

def read_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def run_cmd(args: list[str], *, log_path: Path, dry_run: bool = False, cwd: Path | None = None) -> None:
    cwd = cwd or Path.cwd()
    display = shlex.join(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] cwd={cwd}\n$ {display}\n")
    print(f"$ {display}")
    if dry_run:
        return
    start = time.time()
    subprocess.run(args, cwd=str(cwd), check=True)
    elapsed = time.time() - start
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[elapsed_seconds] {elapsed:.1f}\n")

def capture_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # noqa: BLE001 - environment logging should not fail the panel
        return f"FAILED {shlex.join(args)}: {exc}"

def environment_lines() -> list[str]:
    lines = [
        f"created_at_utc={now_utc()}",
        f"python={sys.executable}",
        capture_cmd([sys.executable, "--version"]),
        "",
        "uv:",
        capture_cmd(["uv", "--version"]),
        "",
        "nix:",
        capture_cmd(["nix", "--version"]),
        "",
        "nvidia-smi:",
        capture_cmd(["nvidia-smi", "-L"]),
        "",
        "torch:",
        capture_cmd([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"]),
        "",
        "environment:",
    ]
    for key in sorted(os.environ):
        if key.startswith(("CUDA", "DATASET_ARTIFACTS", "HF_", "TRANSFORMERS", "UV_")):
            lines.append(f"{key}={os.environ[key]}")
    return lines

def write_environment_logs(results_dir: Path) -> None:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "git_commit.txt").write_text(
        "\n".join([capture_cmd(["git", "rev-parse", "HEAD"]), "", "git status --short:", capture_cmd(["git", "status", "--short"]), ""]),
        encoding="utf-8",
    )
    (logs_dir / "environment.txt").write_text("\n".join(environment_lines()) + "\n", encoding="utf-8")

def model_file_exists(out_dir: Path) -> bool:
    has_config = (out_dir / "config.json").exists()
    has_weights = (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()
    return has_config and has_weights

def make_run_id(
    model_short: str,
    subset: str,
    frac: float,
    seed: int,
    budget: str,
    *,
    draw_id: int | None = None,
    confidence_definition: str | None = None,
    primary_confidence_definition: str | None = None,
) -> str:
    parts = [model_short, subset, f"frac{fraction_label(frac)}"]
    if draw_id is not None:
        parts.append(f"draw{draw_id:02d}")
    if confidence_definition and confidence_definition != primary_confidence_definition:
        parts.append(confidence_definition)
    parts.extend([f"seed{seed}", budget])
    return "__".join(parts)

def base_train_args(cfg: dict[str, Any], spec: TrainSpec) -> list[str]:
    training = cfg["training"]
    model = cfg["model"]
    args = [
        sys.executable,
        "run.py",
        "--do_train",
        "--task",
        "qa",
        "--dataset",
        spec.train_data,
        "--model",
        spec.model_name,
        "--output_dir",
        spec.output_dir,
        "--overwrite_output_dir",
        "--max_length",
        str(model["max_seq_length"]),
        "--per_device_train_batch_size",
        str(training["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training["per_device_eval_batch_size"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--warmup_ratio",
        str(training.get("warmup_ratio", 0.0)),
        "--weight_decay",
        str(training.get("weight_decay", 0.0)),
        "--num_train_epochs",
        str(spec.num_train_epochs),
        "--save_only_final_model",
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
    ]
    if training.get("fp16", False):
        args.append("--fp16")
    if spec.max_steps is not None and spec.max_steps > 0:
        args.extend(["--max_steps", str(spec.max_steps)])
    if spec.save_dynamics:
        args.append("--save_dynamics")
    return args

def train_is_done(out_dir: Path, spec: TrainSpec) -> bool:
    done = (out_dir / "train_metrics.json").exists() and model_file_exists(out_dir)
    if spec.save_dynamics:
        done = done and (
            (out_dir / "training_dynamics.jsonl").exists()
            or any(out_dir.glob("training_dynamics.rank*.jsonl"))
        )
    return done

def write_train_config_copy(cfg: dict[str, Any], spec: TrainSpec) -> None:
    config_copy = Path(cfg["panel"]["results_dir"]) / "configs" / f"{spec.run_id}.yaml"
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train_spec": asdict(spec), "panel_config": cfg}
    config_copy.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def train_model(cfg: dict[str, Any], spec: TrainSpec, *, log_path: Path, dry_run: bool, resume: bool) -> None:
    out_dir = Path(spec.output_dir)
    if resume and train_is_done(out_dir, spec):
        print(f"[skip train] {spec.run_id}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_train_config_copy(cfg, spec)
    run_cmd(base_train_args(cfg, spec), log_path=log_path, dry_run=dry_run)

def eval_args(cfg: dict[str, Any], spec: TrainSpec, eval_path: str, eval_out: Path) -> list[str]:
    model = cfg["model"]
    training = cfg["training"]
    args = [
        sys.executable,
        "run.py",
        "--do_eval",
        "--task",
        "qa",
        "--dataset",
        eval_path,
        "--model",
        spec.output_dir,
        "--output_dir",
        str(eval_out),
        "--overwrite_output_dir",
        "--max_length",
        str(model["max_seq_length"]),
        "--per_device_eval_batch_size",
        str(training["per_device_eval_batch_size"]),
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
    ]
    if training.get("fp16_eval", training.get("fp16", False)):
        args.append("--fp16")
    return args



def subset_protocol(subset: str) -> str:
    if subset == "full":
        return "full"
    if subset == "random":
        return "uniform_random"
    if subset.endswith("_region_pure"):
        return "region_pure_equal_count"
    if subset.endswith("_ranked") or subset in {"easy", "ambiguous", "hard"}:
        return "ranked_equal_size"
    return "custom"


def check_eval_outputs(eval_out: Path) -> tuple[Path, Path]:
    metrics_path = eval_out / "eval_metrics.json"
    predictions_path = eval_out / "eval_predictions.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    return metrics_path, predictions_path

def normalized_eval_metrics(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    metrics_path: Path,
    predictions_dest: Path,
) -> dict[str, Any]:
    results_dir = Path(cfg["panel"]["results_dir"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    eval_data_path = Path(eval_path)
    train_data_path = Path(spec.train_data)
    return {
        "run_id": f"{spec.run_id}__{evalset}",
        "train_run_id": spec.run_id,
        "model": spec.model_name,
        "model_short": spec.model_short,
        "train_subset": spec.train_subset,
        "subset_protocol": subset_protocol(spec.train_subset),
        "subset_size": read_jsonl_count(train_data_path),
        "subset_fraction": spec.subset_fraction,
        "subset_draw_id": spec.subset_draw_id,
        "seed": spec.seed,
        "train_budget_type": spec.train_budget_type,
        "num_train_epochs": spec.num_train_epochs,
        "max_steps": spec.max_steps,
        "evalset": evalset,
        "num_eval_examples": read_jsonl_count(eval_data_path),
        "exact_match": metrics.get("eval_exact_match"),
        "f1": metrics.get("eval_f1"),
        "eval_script": "run.py/custom_squad_postprocess",
        "dataset_path": str(eval_data_path),
        "dataset_hash": sha256_file(eval_data_path),
        "train_dataset_path": str(train_data_path),
        "train_dataset_hash": sha256_file(train_data_path),
        "predictions_path": str(predictions_dest),
        "config_path": str(results_dir / "configs" / f"{spec.run_id}.yaml"),
        "confidence_definition": spec.confidence_definition,
        "created_at_utc": now_utc(),
    }

def write_eval_metrics(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    eval_out: Path,
    raw_metrics_path: Path,
) -> None:
    metrics_path, predictions_path = check_eval_outputs(eval_out)
    predictions_dest = Path(cfg["panel"]["results_dir"]) / "predictions" / f"{spec.run_id}__{evalset}.jsonl"
    predictions_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(predictions_path, predictions_dest)
    normalized = normalized_eval_metrics(cfg, spec, evalset, eval_path, metrics_path, predictions_dest)
    raw_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    raw_metrics_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")

def eval_model(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    *,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    eval_run_id = f"{spec.run_id}__{evalset}"
    eval_out = results_dir / "evals" / eval_run_id
    raw_metrics_path = results_dir / "metrics" / "raw" / f"{eval_run_id}.json"
    if resume and raw_metrics_path.exists():
        print(f"[skip eval] {eval_run_id}")
        return
    eval_out.mkdir(parents=True, exist_ok=True)
    run_cmd(eval_args(cfg, spec, eval_path, eval_out), log_path=log_path, dry_run=dry_run)
    if not dry_run:
        write_eval_metrics(cfg, spec, evalset, eval_path, eval_out, raw_metrics_path)

def ensure_data(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    data_cfg = cfg["data"]
    required = [Path(v) for v in data_cfg["evalsets"].values()] + [Path(data_cfg["squad_train"])]
    if all(path.exists() for path in required):
        print("[data] required JSONL files already exist")
        return
    args = [sys.executable, "scripts/materialize_qa_data.py", "--out-dir", str(data_cfg["data_dir"])]
    for key, cli in [
        ("squad_train_json", "--squad-train-json"),
        ("squad_dev_json", "--squad-dev-json"),
        ("addsent_json", "--addsent-json"),
        ("addonesent_json", "--addonesent-json"),
    ]:
        if data_cfg.get(key):
            args.extend([cli, str(data_cfg[key])])
    run_cmd(args, log_path=log_path, dry_run=dry_run)

def full_seed42_spec(cfg: dict[str, Any]) -> TrainSpec:
    panel = cfg["panel"]
    model = cfg["model"]
    training = cfg["training"]
    seed = int(cfg["cartography"].get("source_seed", 42))
    run_id = make_run_id(
        model["short_name"],
        "full",
        1.0,
        seed,
        "same_epochs",
        confidence_definition=cfg["cartography"]["primary"],
        primary_confidence_definition=cfg["cartography"]["primary"],
    )
    return TrainSpec(
        run_id=run_id,
        model_short=model["short_name"],
        model_name=model["name_or_path"],
        train_subset="full",
        subset_fraction=1.0,
        subset_draw_id=None,
        seed=seed,
        train_budget_type="same_epochs",
        train_data=cfg["data"]["squad_train"],
        output_dir=str(Path(panel["results_dir"]) / "runs" / run_id),
        confidence_definition=cfg["cartography"]["primary"],
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=None,
        save_dynamics=True,
    )

def cartography_args(cfg: dict[str, Any], source_spec: TrainSpec, out_dir: Path, definition: dict[str, Any]) -> list[str]:
    args = [
        sys.executable,
        "dynamics.py",
        "--td_dir",
        source_spec.output_dir,
        "--output_dir",
        str(out_dir),
        "--confidence_field",
        definition["field"],
        "--seed",
        str(source_spec.seed),
    ]
    if definition.get("window_policy"):
        args.extend(["--qa_window_policy", str(definition["window_policy"])])
    if cfg["cartography"].get("limit_scatter_samples"):
        args.extend(["--limit_scatter_samples", str(cfg["cartography"]["limit_scatter_samples"])])
    return args

def run_cartography_scores(
    cfg: dict[str, Any],
    source_spec: TrainSpec,
    name: str,
    definition: dict[str, Any],
    *,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> Path:
    out_dir = Path(cfg["panel"]["results_dir"]) / "cartography" / name
    csv_path = out_dir / "cartography_scores.csv"
    if resume and csv_path.exists():
        print(f"[skip cartography] {name}")
    else:
        run_cmd(cartography_args(cfg, source_spec, out_dir, definition), log_path=log_path, dry_run=dry_run)
    return csv_path

def subset_selection_args(
    cfg: dict[str, Any],
    csv_path: Path,
    subsets_out: Path,
    assignments_out: Path,
    name: str,
    definition: dict[str, Any],
) -> list[str]:
    fractions = sorted({float(x) for x in cfg["cartography"]["subset_fractions"]})
    region_fractions = sorted({float(x) for x in cfg["cartography"].get("region_pure_fractions", [0.198])})
    return [
        sys.executable,
        "scripts/select_qa_subsets.py",
        "--train-data",
        cfg["data"]["squad_train"],
        "--cartography-scores",
        str(csv_path),
        "--out-dir",
        str(subsets_out),
        "--assignments-out",
        str(assignments_out),
        "--confidence-definition",
        definition.get("label", name),
        "--random-draws",
        str(cfg["cartography"].get("random_draws", 10)),
        "--random-seed-base",
        str(cfg["cartography"].get("random_seed_base", 7300)),
        "--fractions",
        *[str(x) for x in fractions],
        "--region-pure-fractions",
        *[str(x) for x in region_fractions],
    ]

def run_subset_selection(
    cfg: dict[str, Any],
    csv_path: Path,
    name: str,
    definition: dict[str, Any],
    *,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> None:
    cart_dir = Path(cfg["panel"]["results_dir"]) / "cartography"
    subsets_out = Path(cfg["data"]["subsets_dir"]) / name
    assignments_out = cart_dir / ("subset_assignments.csv" if name == cfg["cartography"]["primary"] else f"subset_assignments_{name}.csv")
    manifest_path = subsets_out / "subset_manifest.csv"
    if resume and manifest_path.exists() and assignments_out.exists():
        print(f"[skip subset selection] {name}")
        return
    args = subset_selection_args(cfg, csv_path, subsets_out, assignments_out, name, definition)
    run_cmd(args, log_path=log_path, dry_run=dry_run)

def publish_primary_cartography(cfg: dict[str, Any], *, dry_run: bool) -> None:
    primary = cfg["cartography"]["primary"]
    primary_csv = Path(cfg["panel"]["results_dir"]) / "cartography" / primary / "cartography_scores.csv"
    top_csv = Path(cfg["panel"]["results_dir"]) / "cartography" / "cartography_scores.csv"
    if primary_csv.exists() and not dry_run:
        shutil.copy2(primary_csv, top_csv)

def run_cartography(cfg: dict[str, Any], source_spec: TrainSpec, *, log_path: Path, dry_run: bool, resume: bool) -> None:
    cart_dir = Path(cfg["panel"]["results_dir"]) / "cartography"
    cart_dir.mkdir(parents=True, exist_ok=True)
    for name, definition in cfg["cartography"]["definitions"].items():
        csv_path = run_cartography_scores(cfg, source_spec, name, definition, log_path=log_path, dry_run=dry_run, resume=resume)
        run_subset_selection(cfg, csv_path, name, definition, log_path=log_path, dry_run=dry_run, resume=resume)
    publish_primary_cartography(cfg, dry_run=dry_run)

def parse_full_steps(source_spec: TrainSpec, fallback: int) -> int:
    state_path = Path(source_spec.output_dir) / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("global_step"):
            return int(state["global_step"])
    return int(fallback)

def subset_path(cfg: dict[str, Any], confidence_definition: str, subset: str, frac: float, draw_id: int | None = None) -> str:
    base = Path(cfg["data"]["subsets_dir"]) / confidence_definition
    flabel = fraction_label(frac)
    if subset == "random":
        if draw_id is None:
            raise ValueError("random subset requires draw_id")
        return str(base / f"random_frac{flabel}_draw{draw_id:02d}.jsonl")
    return str(base / f"{subset}_frac{flabel}.jsonl")

def add_spec(
    specs: dict[str, TrainSpec],
    cfg: dict[str, Any],
    *,
    subset: str,
    frac: float,
    seed: int,
    budget: str,
    train_data: str,
    draw_id: int | None = None,
    confidence_definition: str | None = None,
    max_steps: int | None = None,
    save_dynamics: bool = False,
) -> None:
    model = cfg["model"]
    training = cfg["training"]
    confidence_definition = confidence_definition or cfg["cartography"]["primary"]
    run_id = make_run_id(
        model["short_name"],
        subset,
        frac,
        seed,
        budget,
        draw_id=draw_id,
        confidence_definition=confidence_definition,
        primary_confidence_definition=cfg["cartography"]["primary"],
    )
    specs[run_id] = TrainSpec(
        run_id=run_id,
        model_short=model["short_name"],
        model_name=model["name_or_path"],
        train_subset=subset,
        subset_fraction=frac,
        subset_draw_id=draw_id,
        seed=int(seed),
        train_budget_type=budget,
        train_data=train_data,
        output_dir=str(Path(cfg["panel"]["results_dir"]) / "runs" / run_id),
        confidence_definition=confidence_definition,
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=max_steps,
        save_dynamics=save_dynamics,
    )



def ranked_name(subset: str) -> str:
    return f"{subset}_ranked" if subset in {"easy", "ambiguous", "hard"} else subset


def add_full_baseline_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any]) -> None:
    exp = cfg["experiments"]["full_baseline"]
    if not exp.get("enabled", True):
        return
    source_seed = int(cfg["cartography"].get("source_seed", 42))
    for seed in exp["seeds"]:
        add_spec(
            specs,
            cfg,
            subset="full",
            frac=1.0,
            seed=int(seed),
            budget="same_epochs",
            train_data=cfg["data"]["squad_train"],
            save_dynamics=int(seed) == source_seed,
        )

def add_random_fraction_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any], exp_name: str, budget: str, *, max_steps: int | None = None) -> None:
    exp = cfg["experiments"].get(exp_name, {})
    if not exp.get("enabled", True):
        return
    primary = cfg["cartography"]["primary"]
    frac = float(exp["fraction"])
    for draw_id in range(int(exp["random_draws"])):
        for seed in exp["seeds"]:
            add_spec(
                specs,
                cfg,
                subset="random",
                frac=frac,
                draw_id=draw_id,
                seed=int(seed),
                budget=budget,
                train_data=subset_path(cfg, primary, "random", frac, draw_id),
                max_steps=max_steps,
            )

def add_named_fraction_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any], exp_name: str, subset: str, budget: str, *, max_steps: int | None = None) -> None:
    exp = cfg["experiments"].get(exp_name, {})
    if not exp.get("enabled", True):
        return
    primary = cfg["cartography"]["primary"]
    frac = float(exp["fraction"])
    subset = ranked_name(exp.get("subset", subset))
    for seed in exp["seeds"]:
        add_spec(
            specs,
            cfg,
            subset=subset,
            frac=frac,
            seed=int(seed),
            budget=budget,
            train_data=subset_path(cfg, primary, subset, frac),
            max_steps=max_steps,
        )

def add_tier_a_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any]) -> None:
    add_full_baseline_specs(specs, cfg)
    add_random_fraction_specs(specs, cfg, "random_33", "same_epochs")
    for exp_name, subset in [("easy_33", "easy_ranked"), ("ambiguous_33", "ambiguous_ranked"), ("hard_33", "hard_ranked")]:
        add_named_fraction_specs(specs, cfg, exp_name, subset, "same_epochs")

def add_same_steps_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any], same_steps: int) -> None:
    add_random_fraction_specs(specs, cfg, "random_33_same_steps", "same_steps", max_steps=same_steps)
    add_named_fraction_specs(specs, cfg, "hard_33_same_steps", "hard_ranked", "same_steps", max_steps=same_steps)

def add_budget_curve_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any]) -> None:
    primary = cfg["cartography"]["primary"]
    for exp_name in ["budget_10", "budget_20", "budget_50"]:
        exp = cfg["experiments"].get(exp_name, {})
        if not exp.get("enabled", False):
            continue
        frac = float(exp["fraction"])
        for subset in exp["subsets"]:
            for seed in exp["seeds"]:
                if subset == "random":
                    for draw_id in range(int(exp["random_draws"])):
                        add_spec(
                            specs,
                            cfg,
                            subset="random",
                            frac=frac,
                            draw_id=draw_id,
                            seed=int(seed),
                            budget="same_epochs",
                            train_data=subset_path(cfg, primary, "random", frac, draw_id),
                        )
                else:
                    subset_name = ranked_name(subset)
                    add_spec(
                        specs,
                        cfg,
                        subset=subset_name,
                        frac=frac,
                        seed=int(seed),
                        budget="same_epochs",
                        train_data=subset_path(cfg, primary, subset_name, frac),
                    )

def add_confidence_ablation_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any]) -> None:
    exp = cfg["experiments"].get("confidence_ablation", {})
    if not exp.get("enabled", True):
        return
    frac = float(exp["fraction"])
    for confidence_definition in exp["definitions"]:
        for subset in exp["subsets"]:
            subset_name = ranked_name(subset)
            for seed in exp["seeds"]:
                add_spec(
                    specs,
                    cfg,
                    subset=subset_name,
                    frac=frac,
                    seed=int(seed),
                    budget="same_epochs",
                    train_data=subset_path(cfg, confidence_definition, subset_name, frac),
                    confidence_definition=confidence_definition,
                )



def add_region_pure_specs(specs: dict[str, TrainSpec], cfg: dict[str, Any]) -> None:
    exp = cfg["experiments"].get("region_pure_198", {})
    if not exp.get("enabled", False):
        return
    primary = cfg["cartography"]["primary"]
    frac = float(exp["fraction"])
    for subset in exp.get("subsets", ["easy", "ambiguous", "hard"]):
        subset_name = subset if subset.endswith("_region_pure") else f"{subset}_region_pure"
        for seed in exp["seeds"]:
            add_spec(
                specs,
                cfg,
                subset=subset_name,
                frac=frac,
                seed=int(seed),
                budget="same_epochs",
                train_data=subset_path(cfg, primary, subset_name, frac),
            )


def build_specs(cfg: dict[str, Any], same_steps: int) -> list[TrainSpec]:
    specs: dict[str, TrainSpec] = {}
    add_tier_a_specs(specs, cfg)
    add_same_steps_specs(specs, cfg, same_steps)
    add_region_pure_specs(specs, cfg)
    add_budget_curve_specs(specs, cfg)
    add_confidence_ablation_specs(specs, cfg)
    return list(specs.values())

def run_current_results_audit(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    run_cmd(
        [
            sys.executable,
            "scripts/audit_current_results.py",
            "--results-dir",
            str(results_dir),
            "--out-dir",
            str(results_dir / "audit"),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )


def aggregate_metrics_tables(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_metrics.py",
            "--metrics-dir",
            str(results_dir / "metrics" / "raw"),
            "--out-dir",
            str(results_dir / "metrics"),
            "--eval-split-metrics",
            str(results_dir / "audit" / "eval_split_metrics.csv"),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )

def run_mechanism_analysis(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    source_spec = full_seed42_spec(cfg)
    clean_pred = results_dir / "predictions" / f"{source_spec.run_id}__squad_dev.jsonl"
    adv_preds = [
        results_dir / "predictions" / f"{source_spec.run_id}__addsent.jsonl",
        results_dir / "predictions" / f"{source_spec.run_id}__addonesent.jsonl",
    ]
    mechanism_features = results_dir / "metrics" / "mechanism_features.csv"
    if not dry_run and not (clean_pred.exists() and all(p.exists() for p in adv_preds)):
        return
    run_cmd(
        [
            sys.executable,
            "scripts/build_mechanism_features.py",
            "--clean-predictions",
            str(clean_pred),
            "--adversarial-predictions",
            *[str(p) for p in adv_preds],
            "--cartography-scores",
            str(results_dir / "cartography" / "subset_assignments.csv"),
            "--out",
            str(mechanism_features),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )
    run_cmd(
        [sys.executable, "scripts/mechanism_univariate.py", "--features", str(mechanism_features), "--out", str(results_dir / "metrics" / "mechanism_univariate.csv")],
        log_path=log_path,
        dry_run=dry_run,
    )

def run_final_audits(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    run_cmd([sys.executable, "scripts/audit_evalsets.py", "--results-dir", str(results_dir)], log_path=log_path, dry_run=dry_run)
    run_current_results_audit(cfg, log_path=log_path, dry_run=dry_run)
    run_cmd([sys.executable, "scripts/validate_results_tree.py", "--results-dir", str(results_dir)], log_path=log_path, dry_run=dry_run)

def aggregate(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    run_current_results_audit(cfg, log_path=log_path, dry_run=dry_run)
    aggregate_metrics_tables(cfg, log_path=log_path, dry_run=dry_run)
    run_mechanism_analysis(cfg, log_path=log_path, dry_run=dry_run)
    run_final_audits(cfg, log_path=log_path, dry_run=dry_run)

def prepare_results_dir(cfg: dict[str, Any], config_path: str) -> Path:
    results_dir = Path(cfg["panel"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, results_dir / "configs" / Path(config_path).name)
    return results_dir

def run_training_and_eval_specs(
    cfg: dict[str, Any],
    specs: list[TrainSpec],
    *,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> None:
    for spec in specs:
        train_model(cfg, spec, log_path=log_path, dry_run=dry_run, resume=resume)
        for evalset, eval_path in cfg["data"]["evalsets"].items():
            eval_model(cfg, spec, evalset, eval_path, log_path=log_path, dry_run=dry_run, resume=resume)

def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir = prepare_results_dir(cfg, args.config)
    log_path = results_dir / "logs" / "command_log.txt"
    resume = not args.no_resume and bool(cfg["panel"].get("resume", True))

    ensure_data(cfg, log_path=log_path, dry_run=args.dry_run)
    if not args.dry_run:
        write_environment_logs(results_dir)

    source_spec = full_seed42_spec(cfg)
    train_model(cfg, source_spec, log_path=log_path, dry_run=args.dry_run, resume=resume)
    run_cartography(cfg, source_spec, log_path=log_path, dry_run=args.dry_run, resume=resume)

    same_steps = parse_full_steps(source_spec, int(cfg["training"].get("same_steps_fallback_max_steps", 8214)))
    print(f"[same_steps target] {same_steps}")
    specs = build_specs(cfg, same_steps)
    if args.limit_runs is not None:
        specs = specs[: args.limit_runs]

    run_training_and_eval_specs(cfg, specs, log_path=log_path, dry_run=args.dry_run, resume=resume)
    aggregate(cfg, log_path=log_path, dry_run=args.dry_run)
    print(f"Panel complete. Results root: {results_dir}")

if __name__ == "__main__":
    main()
