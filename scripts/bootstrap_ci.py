#!/usr/bin/env python3
"""Compute paired bootstrap confidence intervals for prediction-level EM/F1.

Input JSONL should contain one record per example with at least:
  exact_match: 0/1 or float
  f1: float

Usage:
  python scripts/bootstrap_ci.py --predictions results/predictions/run.jsonl --n 10000
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap(rows, field, n, seed):
    rng = random.Random(seed)
    vals = [float(r[field]) for r in rows]
    estimates = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        estimates.append(mean(sample))
    return {
        "mean": mean(vals),
        "ci_low": percentile(estimates, 2.5),
        "ci_high": percentile(estimates, 97.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    out = {
        "predictions": args.predictions,
        "num_examples": len(rows),
        "exact_match": bootstrap(rows, "exact_match", args.n, args.seed),
        "f1": bootstrap(rows, "f1", args.n, args.seed),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
