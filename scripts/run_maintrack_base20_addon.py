#!/usr/bin/env python3
"""Run only the missing ELECTRA-base 20% budget-curve add-on panel.

This deliberately leaves ``configs/maintrack.full.yaml`` unchanged so existing
completion and prune markers retain the exact configuration hash under which
they were produced.  The add-on specs are constructed with the same helper
used by the canonical budget stage, yielding the same run IDs and scientific
hashes that an in-config 20% ELECTRA-base point would have produced.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_maintrack_suite as suite  # noqa: E402

DEFAULT_ADDON = REPO_ROOT / "configs" / "maintrack.electra_base_20pct_addon.yaml"
DEFAULT_MIN_FREE_GIB = 20.0


def load_addon(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Add-on config must be a mapping: {path}")
    return payload


def addon_specs(config: dict[str, Any], addon: dict[str, Any]) -> list[suite.TrainSpec]:
    model_key = str(addon["model"])
    tasks = [str(task) for task in addon["tasks"]]
    fraction = float(addon["fraction"])
    subsets = [str(subset) for subset in addon["subsets"]]
    random_draws = int(addon["random_draws"])
    seed = int(addon["seed"])

    specs: dict[str, suite.TrainSpec] = {}
    for task in tasks:
        selection_map = suite.self_map(config, task, model_key)
        for subset in subsets:
            draws = range(random_draws) if subset == "random" else [None]
            for draw_id in draws:
                suite.append_unique(
                    specs,
                    suite.selected_spec(
                        config,
                        stage="budget",
                        family="budget_curve",
                        task=task,
                        model_key=model_key,
                        selection_map=selection_map,
                        subset=subset,
                        fraction=fraction,
                        seed=seed,
                        draw_id=draw_id,
                    ),
                )
    return list(specs.values())


def validate_design(config: dict[str, Any], addon: dict[str, Any], specs: list[suite.TrainSpec]) -> None:
    model_key = str(addon["model"])
    fraction = float(addon["fraction"])
    tasks = {str(task) for task in addon["tasks"]}
    random_draws = int(addon["random_draws"])

    if model_key != "electra_base" or abs(fraction - 0.20) > 1e-12:
        raise RuntimeError("This add-on is intentionally restricted to ELECTRA-base at fraction 0.20")
    if tasks != {"qa", "classification"}:
        raise RuntimeError(f"Expected QA + classification tasks, got {sorted(tasks)}")

    expected_per_task = random_draws + 3  # random draws + easy/ambiguous/hard
    expected_total = len(tasks) * expected_per_task
    if len(specs) != expected_total:
        raise RuntimeError(f"Expected {expected_total} add-on specs, got {len(specs)}")

    original_ids = {spec.run_id for spec in suite.build_budget_specs(config)}
    collisions = sorted(spec.run_id for spec in specs if spec.run_id in original_ids)
    if collisions:
        raise RuntimeError(f"Add-on unexpectedly overlaps the original budget panel: {collisions}")

    missing_inputs = sorted({spec.train_data for spec in specs if not Path(spec.train_data).exists()})
    if missing_inputs:
        raise RuntimeError(
            "Missing pre-materialized 20% subset files; refusing to rebuild prior stages:\n  "
            + "\n  ".join(missing_inputs)
        )


def require_original_budget_complete(config: dict[str, Any]) -> None:
    original = suite.build_budget_specs(config)
    incomplete = [
        spec.run_id
        for spec in original
        if not (suite.train_complete(config, spec) and suite.all_evals_complete(config, spec))
    ]
    if incomplete:
        raise RuntimeError(
            "Original budget panel is not fully complete; refusing to run an add-on that could "
            "be mistaken for a general budget resume. Incomplete original runs:\n  "
            + "\n  ".join(incomplete)
        )
    print(f"original budget panel: {len(original)}/{len(original)} complete", flush=True)


def write_manifest(results_dir: Path, specs: list[suite.TrainSpec]) -> Path:
    path = results_dir / "status" / "electra_base_20pct_addon_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(spec) for spec in specs]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def completion_summary(config: dict[str, Any], specs: list[suite.TrainSpec]) -> tuple[int, int]:
    complete = sum(
        suite.train_complete(config, spec) and suite.all_evals_complete(config, spec) for spec in specs
    )
    return complete, len(specs) - complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon", default=str(DEFAULT_ADDON))
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    addon_path = Path(args.addon).resolve()
    addon = load_addon(addon_path)
    base_config_path = (REPO_ROOT / str(addon["base_config"])).resolve()
    config = suite.load_config(base_config_path)
    results_dir = Path(config["suite"]["results_dir"])
    specs = addon_specs(config, addon)

    validate_design(config, addon, specs)
    require_original_budget_complete(config)
    manifest = write_manifest(results_dir, specs)
    complete, remaining = completion_summary(config, specs)

    print("=== ELECTRA-base 20% budget add-on ===", flush=True)
    print(f"base config: {base_config_path}", flush=True)
    print(f"add-on config: {addon_path}", flush=True)
    print(f"specs: {len(specs)} (complete={complete}, remaining={remaining})", flush=True)
    print(f"manifest: {manifest}", flush=True)
    for spec in specs:
        status = "complete" if suite.train_complete(config, spec) and suite.all_evals_complete(config, spec) else "pending"
        print(f"  [{status}] {spec.run_id}", flush=True)

    if remaining == 0:
        print("All ELECTRA-base 20% add-on experiments are already complete.", flush=True)
        return

    if not args.dry_run:
        stale_dirs, stale_bytes = suite.cleanup_stale_hf_scratch()
        if stale_dirs:
            print(
                f"[map scratch cleanup] removed {stale_dirs} stale dirs "
                f"({stale_bytes / 1024**3:.2f} GiB)",
                flush=True,
            )
        free = shutil.disk_usage(results_dir).free
        minimum = int(args.min_free_gib * 1024**3)
        print(f"filesystem free: {free / 1024**3:.2f} GiB", flush=True)
        if free < minimum:
            raise RuntimeError(
                f"Only {free / 1024**3:.2f} GiB free; require at least {args.min_free_gib:.2f} GiB"
            )

    tracker = suite.StatusTracker(results_dir, config, dry_run=args.dry_run)
    log_path = results_dir / "logs" / "electra_base_20pct_addon_commands.log"
    gpu_ids = suite.parse_gpu_ids(args.gpu_ids)
    suite.run_parallel_specs(
        config,
        specs,
        tracker=tracker,
        log_path=log_path,
        dry_run=args.dry_run,
        gpu_ids=gpu_ids,
        evaluate=True,
    )

    if not args.dry_run:
        complete, remaining = completion_summary(config, specs)
        if remaining:
            raise RuntimeError(f"Add-on ended with {remaining} incomplete specs")
    print("ELECTRA-base 20% add-on complete.", flush=True)


if __name__ == "__main__":
    main()
