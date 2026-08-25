#!/usr/bin/env python3
"""Safely reclaim disk from completed main-track runs before resume."""

from __future__ import annotations

import argparse
import os
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

from scripts import run_maintrack_suite as suite  # noqa: E402


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def persistent_hf_datasets_cache() -> Path:
    configured = os.environ.get("HF_DATASETS_CACHE")
    if configured:
        return Path(configured)
    return REPO_ROOT / ".cache" / "huggingface" / "datasets"


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def reclaim_persistent_transform_cache() -> tuple[int, int]:
    """Remove only derived Dataset.map Arrow caches, preserving base datasets."""
    cache_root = persistent_hf_datasets_cache()
    if not cache_root.exists():
        return 0, 0
    files = sorted(cache_root.rglob("cache-*.arrow"))
    bytes_removed = sum(path.stat().st_size for path in files if path.is_file())
    removed = 0
    for path in files:
        if not path.is_file():
            continue
        path.unlink()
        removed += 1
    return removed, bytes_removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maintrack.full.yaml")
    parser.add_argument("--warn-free-gib", type=float, default=20.0)
    parser.add_argument("--require-free-gib", type=float, default=0.0)
    args = parser.parse_args()

    config = suite.load_config(Path(args.config))
    results_dir = Path(config["suite"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(results_dir)
    transform_cache_files, transform_cache_bytes = reclaim_persistent_transform_cache()
    report = suite.reclaim_completed_run_space(config)
    after = shutil.disk_usage(results_dir)

    print("=== main-track disk reclamation ===")
    print(f"filesystem free before: {human_bytes(before.free)}")
    print(f"persistent transform cache files: {transform_cache_files}")
    print(f"persistent transform cache bytes: {human_bytes(transform_cache_bytes)}")
    print(f"completed models pruned: {report['pruned_runs']}")
    print(f"completed model bytes:  {human_bytes(report['model_bytes'])}")
    print(f"partial weight files:   {report['partial_weight_files']}")
    print(f"partial weight bytes:   {human_bytes(report['partial_weight_bytes'])}")
    print(f"eval scratch dirs:      {report['eval_scratch_dirs']}")
    print(f"eval scratch bytes:     {human_bytes(report['eval_scratch_bytes'])}")
    print(f"predictions compressed: {report['compressed_prediction_files']}")
    print(f"prediction bytes before:{human_bytes(report['prediction_bytes_before']):>12}")
    print(f"prediction bytes after: {human_bytes(report['prediction_bytes_after']):>12}")
    print(f"prediction bytes freed: {human_bytes(report['prediction_bytes_freed'])}")
    print(f"invalid predictions kept: {report['prediction_files_skipped_invalid']}")
    print(
        f"total reclaimed:        "
        f"{human_bytes(report['total_bytes'] + transform_cache_bytes)}"
    )
    print(f"filesystem free after:  {human_bytes(after.free)}")
    if after.free < int(args.warn_free_gib * 1024**3):
        print(
            f"WARNING: only {human_bytes(after.free)} remains; "
            f"less than the {args.warn_free_gib:.1f} GiB operating headroom target."
        )
    if args.require_free_gib and after.free < int(args.require_free_gib * 1024**3):
        raise SystemExit(
            f"ERROR: only {human_bytes(after.free)} free after reclamation; "
            f"need at least {args.require_free_gib:.1f} GiB"
        )


if __name__ == "__main__":
    main()
