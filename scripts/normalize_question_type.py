#!/usr/bin/env python3
"""Normalize SQuAD questions into a small taxonomy for analysis tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_question_type(question: str) -> str:
    """Map a question string to a stable coarse wh-type taxonomy."""
    q = re.sub(r"^[^A-Za-z0-9]+", "", str(question).strip().lower())
    toks = re.findall(r"[a-z0-9]+", q)
    if not toks:
        return "other"
    first = toks[0]
    second = toks[1] if len(toks) > 1 else ""
    if first == "how" and second in {"many", "long", "much"}:
        return f"how_{second}"
    if first in {"what", "who", "when", "where", "why", "which", "how"}:
        return first
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "question" not in df.columns:
        raise SystemExit("Input file must contain a `question` column.")
    df["question_type"] = df["question"].map(normalize_question_type)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df["question_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
