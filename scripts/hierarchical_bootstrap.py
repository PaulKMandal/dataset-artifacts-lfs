#!/usr/bin/env python3
"""Hierarchical bootstrap over subset draw, training seed, and evaluation item."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_runs(metrics_dir: Path, filters: dict, metric: str) -> list[dict]:
    runs = []
    for path in sorted(metrics_dir.glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in filters.items()):
            continue
        predictions = read_jsonl(Path(metadata["predictions_path"]))
        values = []
        for row in predictions:
            if metric == "correct":
                value = float(row["predicted_label"] == row["label"])
            else:
                value = float(row[metric])
            values.append(value)
        if not values:
            raise ValueError(f"No prediction values in {metadata['predictions_path']}")
        runs.append({"metadata": metadata, "values": np.asarray(values, dtype=float)})
    if not runs:
        raise ValueError(f"No runs matched filters: {filters}")
    return runs


def hierarchy(runs: list[dict]) -> dict[str, dict[int, list[np.ndarray]]]:
    output = defaultdict(lambda: defaultdict(list))
    for run in runs:
        metadata = run["metadata"]
        draw = metadata.get("subset_draw_id")
        selection_unit = "fixed" if draw is None else f"draw{int(draw):02d}"
        output[selection_unit][int(metadata["seed"])].append(run["values"])
    return output


def bootstrap_condition(runs: list[dict], rng: np.random.Generator) -> float:
    nested = hierarchy(runs)
    units = sorted(nested)
    sampled_unit_names = rng.choice(units, size=len(units), replace=True)
    estimates = []
    for unit_name in sampled_unit_names:
        seeds = sorted(nested[unit_name])
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        for seed in sampled_seeds:
            replicates = nested[unit_name][int(seed)]
            values = replicates[int(rng.integers(len(replicates)))]
            sampled_examples = rng.choice(values, size=len(values), replace=True)
            estimates.append(float(sampled_examples.mean()))
    return float(np.mean(estimates))


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.percentile(values, probability))


def run_contrast(metrics_dir: Path, contrast: dict, n: int, seed: int) -> dict:
    metric = contrast["metric"]
    common = contrast.get("common", {})
    runs_a = load_runs(metrics_dir, common | contrast["a"], metric)
    runs_b = load_runs(metrics_dir, common | contrast["b"], metric)
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for index in range(n):
        draws[index] = bootstrap_condition(runs_a, rng) - bootstrap_condition(runs_b, rng)
    scale = 100.0
    return {
        "name": contrast["name"],
        "metric": metric,
        "condition_a": contrast["a"],
        "condition_b": contrast["b"],
        "common": common,
        "n_runs_a": len(runs_a),
        "n_runs_b": len(runs_b),
        "mean_difference_points": float(draws.mean() * scale),
        "ci_low_points": percentile(draws, 2.5) * scale,
        "ci_high_points": percentile(draws, 97.5) * scale,
        "probability_a_gt_b": float((draws > 0).mean()),
        "bootstrap_samples": n,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--contrasts", required=True, help="JSON list of contrast specifications")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    contrasts = json.loads(Path(args.contrasts).read_text(encoding="utf-8"))
    rows = [
        run_contrast(Path(args.metrics_dir), contrast, args.n, args.seed + index)
        for index, contrast in enumerate(contrasts)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
