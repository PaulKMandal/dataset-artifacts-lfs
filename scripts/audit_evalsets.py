#!/usr/bin/env python3
"""Write a lightweight evalset/table-label audit for the result tree."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_COUNTS = {
    "squad_dev": 10570,
    "addsent": 3560,
    "addonesent": 1787,
}

def load_metrics(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((results_dir / "metrics" / "raw").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_file"] = str(path)
        rows.append(row)
    return rows

def group_by_evalset(rows: list[dict]) -> dict[str, list[dict]]:
    by_evalset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_evalset[row.get("evalset", "UNKNOWN")].append(row)
    return by_evalset

def evalset_status(evalset: str, group: list[dict]) -> tuple[list[str], list[str], list[str], str]:
    observed_counts = sorted({str(r.get("num_eval_examples")) for r in group})
    paths = sorted({str(r.get("dataset_path")) for r in group})
    hashes = sorted({str(r.get("dataset_hash")) for r in group})
    expected = EXPECTED_COUNTS.get(evalset)
    status = "OK"
    if expected is not None and observed_counts != [str(expected)]:
        status = "CHECK_COUNT"
    if len(paths) != 1 or len(hashes) != 1:
        status = "CHECK_MULTIPLE_DATASETS"
    return observed_counts, paths, hashes, status

def audit_table(by_evalset: dict[str, list[dict]]) -> list[str]:
    lines = [
        "# Table/evalset audit",
        "",
        "This audit is generated from normalized metrics JSON files. It checks evalset names, example counts, dataset paths, and dataset hashes before tables are reported.",
        "",
        "## Evalset identity checks",
        "",
        "| evalset | expected n | observed n values | dataset paths | dataset hashes | status |",
        "|---|---:|---|---|---|---|",
    ]
    for evalset in sorted(by_evalset):
        counts, paths, hashes, status = evalset_status(evalset, by_evalset[evalset])
        expected = EXPECTED_COUNTS.get(evalset)
        lines.append(
            f"| {evalset} | {expected if expected is not None else ''} | {', '.join(counts)} | {', '.join(paths)} | {', '.join(hashes)} | {status} |"
        )
    return lines

def label_check_lines(by_evalset: dict[str, list[dict]]) -> list[str]:
    lines = [
        "",
        "## AddSent/AddOneSent label check",
        "",
        "A likely label swap is flagged if AddSent and AddOneSent example counts do not match the expected adversarial SQuAD counts or if either evalset is backed by more than one dataset hash.",
        "",
    ]
    addsent_ok = by_evalset.get("addsent") and sorted({r.get("num_eval_examples") for r in by_evalset["addsent"]}) == [EXPECTED_COUNTS["addsent"]]
    addonesent_ok = by_evalset.get("addonesent") and sorted({r.get("num_eval_examples") for r in by_evalset["addonesent"]}) == [EXPECTED_COUNTS["addonesent"]]
    if addsent_ok and addonesent_ok:
        lines.append("Conclusion: evalset labels are consistent with expected AddSent/AddOneSent example counts.")
    else:
        lines.append("Conclusion: CHECK REQUIRED before reporting AddSent/AddOneSent table labels.")
    return lines

def write_audit(results_dir: Path) -> Path:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / "table_audit.md"
    by_evalset = group_by_evalset(load_metrics(results_dir))
    lines = audit_table(by_evalset) + label_check_lines(by_evalset)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    print(f"Wrote {write_audit(Path(args.results_dir))}")

if __name__ == "__main__":
    main()
