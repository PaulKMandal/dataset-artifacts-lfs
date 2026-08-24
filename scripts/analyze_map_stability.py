#!/usr/bin/env python3
"""Compute pairwise rank, region, and selected-set stability across data maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"idx", "confidence", "variability", "correctness"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["idx"].duplicated().any():
        raise ValueError(f"{path} contains duplicate idx values")
    return frame.set_index("idx").sort_index()


def ambiguous_indices(frame: pd.DataFrame, fraction: float) -> set[int]:
    k = max(1, int(round(len(frame) * fraction)))
    ranked = frame.assign(mid_distance=(frame["confidence"] - 0.5).abs()).sort_values(
        ["variability", "mid_distance"], ascending=[False, True], kind="mergesort"
    )
    return set(int(idx) for idx in ranked.index[:k])


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    result = spearmanr(left, right) if method == "spearman" else kendalltau(left, right)
    statistic = float(result.statistic)
    return statistic if math.isfinite(statistic) else None


def compare_maps(name_a: str, map_a: pd.DataFrame, name_b: str, map_b: pd.DataFrame, fractions):
    common = map_a.index.intersection(map_b.index)
    if len(common) < 2:
        raise ValueError(f"Maps {name_a!r} and {name_b!r} have fewer than two shared examples")
    left = map_a.loc[common]
    right = map_b.loc[common]
    base = {
        "map_a": name_a,
        "map_b": name_b,
        "n_a": len(map_a),
        "n_b": len(map_b),
        "n_common": len(common),
        "common_fraction_a": len(common) / len(map_a),
        "common_fraction_b": len(common) / len(map_b),
        "confidence_spearman": correlation(left["confidence"], right["confidence"], "spearman"),
        "variability_spearman": correlation(left["variability"], right["variability"], "spearman"),
        "correctness_spearman": correlation(left["correctness"], right["correctness"], "spearman"),
        "variability_kendall": correlation(left["variability"], right["variability"], "kendall"),
        "region_agreement": float((left.get("region", "") == right.get("region", "")).mean())
        if "region" in left and "region" in right
        else None,
    }
    rows = []
    for fraction in fractions:
        row = dict(base)
        row["fraction"] = fraction
        row["ambiguous_jaccard"] = jaccard(
            ambiguous_indices(left, fraction), ambiguous_indices(right, fraction)
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="append", required=True, type=named_path)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.333, 0.5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    maps = [(name, load_map(path)) for name, path in args.map]
    rows = []
    for (name_a, map_a), (name_b, map_b) in combinations(maps, 2):
        rows.extend(compare_maps(name_a, map_a, name_b, map_b, args.fractions))
    if not rows:
        raise ValueError("At least two --map arguments are required")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
