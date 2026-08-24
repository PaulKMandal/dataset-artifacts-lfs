#!/usr/bin/env python3
"""Audit whether an interrupted main-track suite can truly resume in place.

The audit looks only at artifacts that already claim prior completion. Jobs with
no completion marker are ordinary not-yet-completed work. Existing completion
markers must either validate under the current scientific hash or under an
explicitly declared maintenance-only compatibility hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_maintrack_suite as suite  # noqa: E402


STAGES = ("sources", "core", "capacity", "budget", "ablation")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maintrack.full.yaml")
    parser.add_argument(
        "--max-damaged-prior-trains",
        type=int,
        default=4,
        help=(
            "Abort if more than this many runs with pre-existing _TRAIN_SUCCESS markers "
            "have artifact-integrity damage. Such runs must be rerun because their prior "
            "bytes can no longer be trusted."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = suite.load_config(config_path)
    results_dir = Path(config["suite"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    prior_train_markers = 0
    reusable_train_markers = 0
    damaged_train_markers: list[str] = []
    incompatible_train_markers: list[str] = []

    prior_eval_pairs = 0
    reusable_eval_pairs = 0
    damaged_eval_pairs: list[str] = []
    incompatible_eval_pairs: list[str] = []

    specs_by_run_id = {}
    for stage in STAGES:
        for spec in suite.stage_specs(config, stage):
            specs_by_run_id.setdefault(spec.run_id, spec)
    all_specs = list(specs_by_run_id.values())
    for spec in all_specs:
        marker = suite.train_success_path(spec)
        if marker.exists():
            prior_train_markers += 1
            try:
                payload = read_json(marker)
            except (OSError, ValueError, json.JSONDecodeError):
                damaged_train_markers.append(spec.run_id)
            else:
                if not suite.train_hash_is_compatible(config, spec, payload.get("train_hash")):
                    incompatible_train_markers.append(spec.run_id)
                elif suite.train_complete(config, spec):
                    reusable_train_markers += 1
                else:
                    damaged_train_markers.append(spec.run_id)

        for evalset, eval_path in suite.evalsets_for(config, spec):
            eval_id = f"{spec.run_id}__{evalset}"
            metrics = results_dir / "metrics" / "raw" / f"{eval_id}.json"
            predictions = results_dir / "predictions" / f"{eval_id}.jsonl"
            if not metrics.exists() and not predictions.exists():
                continue
            if not metrics.exists() or not predictions.exists():
                damaged_eval_pairs.append(eval_id)
                continue
            prior_eval_pairs += 1
            try:
                payload = read_json(metrics)
            except (OSError, ValueError, json.JSONDecodeError):
                damaged_eval_pairs.append(eval_id)
                continue
            if payload.get("eval_hash") not in suite.compatible_eval_hashes(
                config, spec, evalset, eval_path
            ):
                incompatible_eval_pairs.append(eval_id)
                continue
            try:
                valid_predictions = payload.get("predictions_hash") == suite.sha256_file(
                    predictions
                )
            except OSError:
                valid_predictions = False
            if valid_predictions:
                reusable_eval_pairs += 1
            else:
                damaged_eval_pairs.append(eval_id)

    prior_map_markers = 0
    reusable_map_markers = 0
    incompatible_map_markers: list[str] = []
    damaged_map_markers: list[str] = []
    for spec in suite.all_source_specs(config):
        for definition_name, definition in suite.map_definitions_for_source(config, spec):
            identifier = suite.map_id(spec.task, spec.model_key, spec.seed, definition_name)
            csv_path = suite.map_csv(config, identifier)
            marker = csv_path.parent / "_MAP_SUCCESS.json"
            if not marker.exists() and not csv_path.exists():
                continue
            if not marker.exists() or not csv_path.exists():
                damaged_map_markers.append(identifier)
                continue
            prior_map_markers += 1
            try:
                payload = read_json(marker)
            except (OSError, ValueError, json.JSONDecodeError):
                damaged_map_markers.append(identifier)
                continue
            if payload.get("map_hash") not in suite.compatible_cartography_completion_hashes(
                config, spec, definition
            ):
                incompatible_map_markers.append(identifier)
                continue
            if payload.get("scores_sha256") == suite.sha256_file(csv_path):
                reusable_map_markers += 1
            else:
                damaged_map_markers.append(identifier)

    smoke = suite.smoke_marker(config)
    smoke_state = "absent"
    if smoke.exists():
        try:
            payload = read_json(smoke)
        except (OSError, ValueError, json.JSONDecodeError):
            smoke_state = "damaged"
        else:
            if payload.get("smoke_hash") not in suite.compatible_smoke_completion_hashes(config):
                smoke_state = "incompatible"
            elif suite.smoke_complete(config, smoke):
                smoke_state = "reusable"
            else:
                smoke_state = "damaged"

    print("=== MAIN-TRACK RESUME AUDIT ===")
    print(
        f"prior train markers: {prior_train_markers}; reusable: {reusable_train_markers}; "
        f"damaged: {len(damaged_train_markers)}; incompatible: "
        f"{len(incompatible_train_markers)}"
    )
    print(
        f"prior complete eval pairs: {prior_eval_pairs}; reusable: {reusable_eval_pairs}; "
        f"damaged/partial: {len(damaged_eval_pairs)}; incompatible: "
        f"{len(incompatible_eval_pairs)}"
    )
    print(
        f"prior map markers: {prior_map_markers}; reusable: {reusable_map_markers}; "
        f"damaged/partial: {len(damaged_map_markers)}; incompatible: "
        f"{len(incompatible_map_markers)}"
    )
    print(f"smoke marker: {smoke_state}")

    if damaged_train_markers:
        print("\nPrior training markers with damaged/mismatched artifacts (must rerun only these):")
        for run_id in damaged_train_markers:
            print(f"  {run_id}")
    if damaged_eval_pairs:
        print("\nDamaged/partial prior canonical evaluations (must rerun only these evals):")
        for eval_id in damaged_eval_pairs:
            print(f"  {eval_id}")
    if damaged_map_markers:
        print("\nDamaged/partial prior maps:")
        for identifier in damaged_map_markers:
            print(f"  {identifier}")

    incompatible = (
        incompatible_train_markers + incompatible_eval_pairs + incompatible_map_markers
    )
    if smoke_state == "incompatible":
        incompatible.append("SMOKE_SUCCESS")
    if incompatible:
        print("\nERROR: prior completion markers are scientifically incompatible:", file=sys.stderr)
        for item in incompatible:
            print(f"  {item}", file=sys.stderr)
        raise SystemExit(2)

    if len(damaged_train_markers) > args.max_damaged_prior_trains:
        print(
            f"\nERROR: {len(damaged_train_markers)} prior successful training runs have "
            f"artifact damage; limit is {args.max_damaged_prior_trains}. Refusing a broad "
            "retrain.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    if smoke_state == "damaged":
        print(
            "\nERROR: prior smoke marker/artifacts are damaged; inspect before launch.",
            file=sys.stderr,
        )
        raise SystemExit(4)

    print("\nRESUME_AUDIT_OK")
    print(
        "Existing compatible completion markers will be reused. Jobs with no marker are "
        "unfinished work, not restarted completed work."
    )


if __name__ == "__main__":
    main()
