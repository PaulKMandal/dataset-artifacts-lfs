#!/usr/bin/env python3
"""Compact status view for the staged main-track suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def gpu_lines() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return output.splitlines()
    except Exception as exc:  # noqa: BLE001 - monitoring should remain available
        return [f"unavailable ({exc})"]


def load_status(config_path: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = Path(config["suite"]["results_dir"])
    status_path = results_dir / "status" / "status.json"
    if not status_path.exists():
        return results_dir, {"jobs": {}, "updated_at_utc": None}
    return results_dir, json.loads(status_path.read_text(encoding="utf-8"))


def status_payload(config_path: Path) -> dict:
    results_dir, state = load_status(config_path)
    jobs = state.get("jobs", {})
    counts = Counter(job.get("status", "unknown") for job in jobs.values())
    by_stage = defaultdict(Counter)
    for job in jobs.values():
        by_stage[job.get("stage", "unknown")][job.get("status", "unknown")] += 1
    active = [
        {"job_id": job_id, **job} for job_id, job in jobs.items() if job.get("status") == "running"
    ]
    failed = [
        {"job_id": job_id, **job} for job_id, job in jobs.items() if job.get("status") == "failed"
    ]
    return {
        "results_dir": str(results_dir),
        "updated_at_utc": state.get("updated_at_utc"),
        "counts": dict(counts),
        "by_stage": {stage: dict(values) for stage, values in sorted(by_stage.items())},
        "active": active,
        "failed": failed[-5:],
        "gpus": gpu_lines(),
    }


def render(payload: dict) -> str:
    lines = [
        f"Results: {payload['results_dir']}",
        f"Updated: {payload['updated_at_utc'] or 'not started'}",
        "Jobs: "
        + ", ".join(f"{name}={count}" for name, count in sorted(payload["counts"].items())),
        "",
        "By stage:",
    ]
    for stage, counts in payload["by_stage"].items():
        lines.append(
            f"  {stage:12s} "
            + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    lines.extend(["", "GPUs:"])
    lines.extend(f"  {line}" for line in payload["gpus"])
    lines.extend(["", "Active:"])
    if payload["active"]:
        for job in payload["active"]:
            lines.append(
                f"  gpu={job.get('gpu_id', '?')} {job['job_id']} since {job.get('started_at_utc', '?')}"
            )
    else:
        lines.append("  none")
    if payload["failed"]:
        lines.extend(["", "Failures:"])
        for job in payload["failed"]:
            lines.append(f"  {job['job_id']}: {job.get('error', 'unknown error')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maintrack.full.yaml")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    while True:
        payload = status_payload(Path(args.config))
        print(json.dumps(payload, indent=2) if args.json else render(payload), flush=True)
        if args.watch is None:
            return
        time.sleep(max(args.watch, 1.0))
        print("\033[2J\033[H", end="", flush=True)


if __name__ == "__main__":
    main()
