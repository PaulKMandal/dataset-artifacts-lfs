#!/usr/bin/env python3
"""Repair AddSent/AddOneSent adversarial-only and paired robustness metrics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import string
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable



def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", action="append", required=True, help="Panel result directory; repeat as needed.")
    ap.add_argument("--dataset-dir", default="data/qa")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--regenerate-missing", action="store_true", help="Run eval-only regeneration from checkpoints when predictions are absent.")
    ap.add_argument("--no-copy-predictions", action="store_true")
    ap.add_argument("--fp16", action="store_true", default=True)
    return ap




ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ADV_MARKER = "-high-conf-"

EVALSETS = {
    "addsent": "addsent.jsonl",
    "addonesent": "addonesent.jsonl",
}

EXPECTED_ROW_COUNTS = {
    "addsent": {"all": 3560, "original": 1000, "adversarial": 2560},
    "addonesent": {"all": 1787, "original": 1000, "adversarial": 787},
}

CONDITION_LABELS = {
    "full": "full data",
    "random": "random 33%",
    "easy": "easy-ranked 33%",
    "ambiguous": "ambiguous-ranked 33%",
    "hard": "hard-ranked 33%",
}


SPLIT_COUNT_FIELDS = [
    "evalset", "dataset_path", "all_rows", "original_rows", "adversarial_rows",
    "n_base_ids_with_original", "n_base_ids_with_adversarial", "n_pairable_base_ids",
    "split_method", "status",
]

PREDICTION_AUDIT_FIELDS = [
    "run_id", "evalset", "dataset_path", "predictions_path", "n_dataset_rows",
    "n_prediction_rows", "n_joined_rows", "n_missing_predictions", "n_extra_predictions", "status",
]

SEED_METRIC_FIELDS = [
    "panel", "model", "condition", "subset_source", "train_seed", "random_draw_id",
    "evalset", "split_type", "n", "exact_match", "f1", "dataset_path",
    "predictions_path", "run_id",
]

MAIN_TABLE_FIELDS = [
    "panel", "model", "condition", "subset_source", "evalset", "split_type",
    "n_runs", "mean_exact_match", "std_exact_match", "mean_f1", "std_f1", "total_rows_scored",
]

PAIRED_FIELDS = [
    "panel", "evalset", "model", "condition", "subset_source", "train_seed",
    "random_draw_id", "run_id", "dataset_path", "predictions_path", "n_pairable_base_ids",
    "n_original_correct_base_ids", "adversarial_em_over_paired_adv_rows",
    "adversarial_f1_over_paired_adv_rows", "any_adv_failure_given_original_correct",
    "all_adv_correct_given_original_correct", "mean_adv_em_given_original_correct",
    "mean_adv_f1_given_original_correct",
]

DELTA_FIELDS = [
    "panel", "model", "evalset", "split_type", "condition", "subset_source", "train_seed",
    "cartographic_exact_match", "mean_random_exact_match_same_seed", "delta_exact_match",
    "cartographic_f1", "mean_random_f1_same_seed", "delta_f1", "n_random_draws_same_seed",
    "mean_delta_exact_match_over_seeds", "std_delta_exact_match_over_seeds",
    "mean_delta_f1_over_seeds", "std_delta_f1_over_seeds", "n_training_seeds",
]

DRAW_FIELDS = [
    "panel", "model", "evalset", "split_type", "random_draw_id", "n_training_seeds",
    "mean_exact_match", "mean_f1",
]

WIN_FIELDS = [
    "panel", "model", "evalset", "split_type", "condition", "n_training_seeds",
    "n_seeds_delta_f1_positive", "n_seeds_delta_exact_match_positive", "mean_delta_f1",
    "mean_delta_exact_match",
]

MISSING_FIELDS = ["panel", "run_id", "evalset", "issue", "path", "detail"]
MANIFEST_FIELDS = [
    "panel", "run_id", "evalset", "source_predictions_path", "packaged_predictions_path", "n_rows", "sha256",
]




def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = s.translate(PUNCT_TABLE)
    s = ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match_score(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_scores(prediction: str, golds: list[str]) -> tuple[int, float]:
    if not golds:
        golds = [""]
    return (
        max(exact_match_score(prediction, g) for g in golds),
        max(f1_score(prediction, g) for g in golds),
    )


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")



def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit("repair implementation is incomplete; apply the remaining commits")


if __name__ == "__main__":
    main()
