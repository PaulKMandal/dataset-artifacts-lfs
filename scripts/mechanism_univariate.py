#!/usr/bin/env python3
"""Compute univariate ROC-AUC/AP for mechanism features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_FEATURES = [
    "confidence",
    "variability",
    "correctness",
    "context_length",
    "answer_length",
    "answer_position_normalized",
    "question_answer_sentence_overlap",
    "question_distractor_overlap",
    "num_added_sentences",
]

def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0.0
    for p in pos:
        greater += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(greater / (len(pos) * len(neg)))

def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    precisions = []
    tp = 0
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else float("nan")

def summarize_feature(evalset: str, group: pd.DataFrame, feature: str) -> dict | None:
    y = group["adversarial_failure"].astype(float).to_numpy()
    vals = pd.to_numeric(group[feature], errors="coerce")
    mask = vals.notna().to_numpy()
    if mask.sum() < 2 or len(np.unique(vals[mask])) < 2:
        return None
    score = vals[mask].astype(float).to_numpy()
    y_masked = y[mask]
    auc = roc_auc(y_masked, score)
    ap = average_precision(y_masked, score)
    auc_flipped = roc_auc(y_masked, -score)
    ap_flipped = average_precision(y_masked, -score)
    return {
        "evalset": evalset,
        "feature": feature,
        "n": int(mask.sum()),
        "positive_rate": float(y_masked.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "roc_auc_abs_best_direction": max(auc, auc_flipped),
        "average_precision_flipped": ap_flipped,
    }

def build_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for evalset, group in df.groupby("evalset"):
        y = group["adversarial_failure"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        for feature in CANDIDATE_FEATURES:
            if feature in group:
                row = summarize_feature(evalset, group, feature)
                if row is not None:
                    rows.append(row)
    return rows

FIELDNAMES = [
    "evalset",
    "feature",
    "n",
    "positive_rate",
    "roc_auc",
    "average_precision",
    "roc_auc_abs_best_direction",
    "average_precision_flipped",
]

def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    if not rows:
        print(f"No usable univariate mechanism rows; wrote empty diagnostic table to {out}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    write_rows(build_rows(pd.read_csv(args.features)), out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
