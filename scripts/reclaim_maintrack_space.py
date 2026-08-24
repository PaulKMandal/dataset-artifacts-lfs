#!/usr/bin/env python3
"""Safely reclaim disk from completed main-track runs before resume."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# When this file is executed as ``python scripts/reclaim_maintrack_space.py``,
# Python places ``scripts/`` rather than the repository root on sys.path.
# Add the repository root explicitly so the package-style import below works
# both for direct execution and for ``python -m scripts.reclaim_maintrack_space``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_maintrack_suite as suite


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maintrack.full.yaml")
    parser.add_argument("--warn-free-gib", type=float, default=6.0)
    args = parser.parse_args()

    config = suite.load_config(Path(args.config))
    results_dir = Path(config["suite"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(results_dir)
    report = suite.reclaim_completed_run_space(config)
    after = shutil.disk_usage(results_dir)

    print("=== main-track disk reclamation ===")
    print(f"filesystem free before: {human_bytes(before.free)}")
    print(f"completed models pruned: {report['pruned_runs']}")
    print(f"completed model bytes:  {human_bytes(report['model_bytes'])}")
    print(f"partial weight files:   {report['partial_weight_files']}")
    print(f"partial weight bytes:   {human_bytes(report['partial_weight_bytes'])}")
    print(f"eval scratch dirs:      {report['eval_scratch_dirs']}")
    print(f"eval scratch bytes:     {human_bytes(report['eval_scratch_bytes'])}")
    print(f"total reclaimed:        {human_bytes(report['total_bytes'])}")
    print(f"filesystem free after:  {human_bytes(after.free)}")
    if after.free < int(args.warn_free_gib * 1024**3):
        print(
            f"WARNING: only {human_bytes(after.free)} remains; "
            f"less than the {args.warn_free_gib:.1f} GiB operating headroom target."
        )


if __name__ == "__main__":
    main()
