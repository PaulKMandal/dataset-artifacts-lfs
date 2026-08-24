#!/usr/bin/env python3
"""Build a consensus cartography map from aligned source maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamics import categorize_examples  # noqa: E402


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("idx").sort_index()
    required = {"confidence", "variability", "correctness"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def consensus(maps: list[tuple[str, Path]], allow_intersection: bool) -> pd.DataFrame:
    loaded = [(name, path, load_map(path)) for name, path in maps]
    index = loaded[0][2].index
    for name, _, frame in loaded[1:]:
        if not index.equals(frame.index):
            if not allow_intersection:
                raise ValueError(f"Map indices do not match exactly for {name!r}")
            index = index.intersection(frame.index)
    if index.empty:
        raise ValueError("Consensus maps have no common examples")

    metrics = {}
    for metric in ["confidence", "variability", "correctness"]:
        percentile_ranks = [
            frame.loc[index, metric].rank(pct=True, method="average") for _, _, frame in loaded
        ]
        metrics[metric] = pd.concat(percentile_ranks, axis=1).mean(axis=1)
    output = pd.DataFrame(metrics, index=index)
    category_input = {
        int(idx): {
            "avg_confidence": row.confidence,
            "variability": row.variability,
            "correctness": row.correctness,
        }
        for idx, row in output.iterrows()
    }
    regions = categorize_examples(category_input)
    output.insert(0, "n_records", len(loaded))
    output["region"] = [regions[int(idx)] for idx in output.index]
    output["confidence_field"] = "mean_percentile_rank"
    output["aggregation"] = "consensus_across_maps"
    return output.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="append", required=True, type=named_path)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-intersection", action="store_true")
    args = parser.parse_args()
    if len(args.map) < 2:
        raise ValueError("Consensus requires at least two maps")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = consensus(args.map, args.allow_intersection)
    frame.to_csv(out, index=False)
    manifest = {
        "output": str(out),
        "num_examples": len(frame),
        "method": "mean_percentile_rank",
        "allow_intersection": args.allow_intersection,
        "inputs": [
            {"name": name, "path": str(path), "sha256": sha256_file(path)}
            for name, path in args.map
        ],
    }
    out.with_name(f"{out.stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
