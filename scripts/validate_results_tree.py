#!/usr/bin/env python3
"""Validate that expected result files exist.

Usage:
  python scripts/validate_results_tree.py --results-dir results
"""

import argparse
from pathlib import Path

REQUIRED = [
    "configs",
    "metrics/main_table.csv",
    "metrics/seed_level_metrics.csv",
    "cartography/cartography_scores.csv",
    "cartography/subset_assignments.csv",
    "logs/command_log.txt",
    "logs/environment.txt",
    "logs/git_commit.txt",
    "logs/table_audit.md",
    "audit/eval_split_metrics.csv",
    "audit/paired_robustness_metrics.csv",
    "audit/subset_purity_overlap.csv",
    "audit/windowing_audit.csv",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    root = Path(args.results_dir)
    missing = []
    for rel in REQUIRED:
        path = root / rel
        if not path.exists():
            missing.append(rel)

    if missing:
        print("Missing required paths:")
        for rel in missing:
            print(f"  - {rel}")
        raise SystemExit(1)

    print("Results tree validation passed.")


if __name__ == "__main__":
    main()
