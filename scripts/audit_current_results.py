#!/usr/bin/env python3
"""Audit round-2 cartography results before paper reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_metrics import squad_exact_match, squad_f1

ADVERSARIAL_ID_RE = re.compile(r"-(?:high-conf|turk)")
EXPECTED_EVALSETS = {"squad_dev", "addsent", "addonesent"}


def is_adversarial_id(example_id: str) -> bool:
    """Return true for known Adversarial SQuAD variant id patterns."""
    return bool(ADVERSARIAL_ID_RE.search(str(example_id)))


def base_id(example_id: str) -> str:
    """Map an adversarial variant id to the base SQuAD question id."""
    return ADVERSARIAL_ID_RE.split(str(example_id), maxsplit=1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metric(row: dict[str, Any], name: str) -> Any:
    return row.get(name, row.get(f"eval_{name}"))


def audit_raw_metrics(results_dir: Path, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    rows = []
    for path in sorted((results_dir / "metrics" / "raw").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        rows.append(
            {
                "metrics_file": path.name,
                "train_run_id": row.get("train_run_id"),
                "model_short": row.get("model_short"),
                "model": row.get("model"),
                "train_subset": row.get("train_subset"),
                "subset_protocol": row.get("subset_protocol"),
                "subset_fraction": row.get("subset_fraction"),
                "subset_draw_id": row.get("subset_draw_id"),
                "seed": row.get("seed"),
                "train_budget_type": row.get("train_budget_type"),
                "evalset": row.get("evalset"),
                "exact_match": _metric(row, "exact_match"),
                "f1": _metric(row, "f1"),
                "num_eval_examples": row.get("num_eval_examples", row.get("eval_samples")),
                "dataset_path": row.get("dataset_path"),
                "dataset_hash": row.get("dataset_hash"),
                "created_at_utc": row.get("created_at_utc"),
                "predictions_path": row.get("predictions_path"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "completed_raw_metrics.csv", index=False)
    if df.empty:
        warnings.append("No raw metric JSON files found.")
        return warnings
    for train_run_id, group in df.groupby("train_run_id", dropna=False):
        missing = EXPECTED_EVALSETS - set(group["evalset"].dropna())
        if missing:
            warnings.append(f"Run {train_run_id} missing evalsets: {sorted(missing)}")
    return warnings


def _prediction_scores(row: dict[str, Any]) -> tuple[float, float]:
    if "exact_match" in row and "f1" in row:
        return float(row["exact_match"]), float(row["f1"])
    if "predicted_answer" not in row or "answers" not in row:
        return 0.0, 0.0
    pred = str(row.get("predicted_answer", ""))
    return squad_exact_match(pred, row["answers"]), squad_f1(pred, row["answers"])


def _split_prediction_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orig_by_base: dict[str, dict[str, Any]] = {}
    adv_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        em, f1 = _prediction_scores(row)
        row["_em"] = em
        row["_f1"] = f1
        ex_id = str(row.get("id", ""))
        groups["all_rows"].append(row)
        if is_adversarial_id(ex_id):
            groups["adversarial_rows_only"].append(row)
            adv_by_base[base_id(ex_id)].append(row)
        else:
            groups["original_rows_only"].append(row)
            orig_by_base[base_id(ex_id)] = row
    return groups, orig_by_base, adv_by_base


def _split_metric_row(path: Path, evalset: str, split: str, vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prediction_file": path.name,
        "evalset": evalset,
        "split": split,
        "n": len(vals),
        "exact_match": 100.0 * sum(r["_em"] for r in vals) / len(vals),
        "f1": 100.0 * sum(r["_f1"] for r in vals) / len(vals),
    }
