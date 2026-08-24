#!/usr/bin/env python3
"""Measure statistical, article, and lexical coverage of selected subsets."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_metrics import normalize_answer  # noqa: E402

TOKEN_PATTERN = re.compile(r"\w+")
SUBSET_COLUMNS = {
    "easy": "selected_easy",
    "ambiguous": "selected_ambiguous",
    "hard": "selected_hard",
    "ambiguous_coverage": "selected_ambiguous_coverage",
}


def read_jsonl_by_idx(path: Path) -> dict[int, dict]:
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                output[int(row.get("idx", row_number))] = row
    return output


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def categorical_js(full: pd.Series, subset: pd.Series) -> float:
    categories = sorted(set(full.fillna("")) | set(subset.fillna("")))
    full_counts = full.fillna("").value_counts(normalize=True).reindex(categories, fill_value=0)
    subset_counts = subset.fillna("").value_counts(normalize=True).reindex(categories, fill_value=0)
    return float(jensenshannon(full_counts, subset_counts, base=2.0) ** 2)


def normalized_wasserstein(full: pd.Series, subset: pd.Series) -> float:
    full_values = pd.to_numeric(full, errors="coerce").dropna().to_numpy()
    subset_values = pd.to_numeric(subset, errors="coerce").dropna().to_numpy()
    if not len(full_values) or not len(subset_values):
        return math.nan
    scale = float(np.std(full_values))
    distance = float(wasserstein_distance(full_values, subset_values))
    return distance / scale if scale > 0 else distance


def tokens(row: dict) -> list[str]:
    text = " ".join(
        [
            row.get("question", ""),
            row.get("context", ""),
            " ".join(row.get("answers", {}).get("text", [])),
        ]
    )
    return TOKEN_PATTERN.findall(normalize_answer(text))


def token_stats(indices: set[int], all_indices: set[int], records: dict[int, dict]) -> dict:
    full_counts = Counter(token for idx in all_indices for token in tokens(records[idx]))
    subset_counts = Counter(token for idx in indices for token in tokens(records[idx]))
    vocab_coverage = len(set(subset_counts) & set(full_counts)) / max(len(full_counts), 1)
    vocabulary = sorted(set(full_counts) | set(subset_counts))
    full = np.array([full_counts[token] for token in vocabulary], dtype=float)
    subset = np.array([subset_counts[token] for token in vocabulary], dtype=float)
    full /= max(full.sum(), 1.0)
    subset /= max(subset.sum(), 1.0)
    return {
        "vocabulary_coverage": vocab_coverage,
        "unigram_js_divergence": float(jensenshannon(full, subset, base=2.0) ** 2),
    }


def analyze(assignments: pd.DataFrame, records: dict[int, dict]) -> list[dict]:
    rows = []
    for fraction, full in assignments.groupby("subset_fraction"):
        all_indices = set(int(idx) for idx in full["idx"])
        missing = all_indices - set(records)
        if missing:
            raise ValueError(f"Training data is missing {len(missing)} assignment idx values")
        for subset, column in SUBSET_COLUMNS.items():
            if column not in full:
                continue
            chosen = full[bool_series(full[column])]
            if chosen.empty:
                continue
            indices = set(int(idx) for idx in chosen["idx"])
            full_titles = {records[idx].get("title", "") for idx in all_indices}
            chosen_titles = {records[idx].get("title", "") for idx in indices}
            row = {
                "subset": subset,
                "subset_fraction": float(fraction),
                "subset_size": len(indices),
                "question_type_js_divergence": categorical_js(
                    full["question_type"], chosen["question_type"]
                ),
                "question_type_coverage": chosen["question_type"].nunique()
                / max(full["question_type"].nunique(), 1),
                "title_coverage": len(chosen_titles & full_titles) / max(len(full_titles), 1),
            }
            for feature in [
                "context_length",
                "answer_length",
                "answer_position_normalized",
            ]:
                if feature in full:
                    row[f"{feature}_normalized_wasserstein"] = normalized_wasserstein(
                        full[feature], chosen[feature]
                    )
            row.update(token_stats(indices, all_indices, records))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments)
    records = read_jsonl_by_idx(Path(args.train_data))
    rows = analyze(assignments, records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
