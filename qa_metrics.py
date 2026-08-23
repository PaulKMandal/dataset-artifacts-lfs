"""SQuAD-style exact-match and token-F1 helpers.

These helpers are intentionally small and dependency-free so they can be used
both by ``run.py`` when saving raw predictions and by analysis scripts that need
per-example correctness labels.
"""

from __future__ import annotations

import collections
import re
import string
from typing import Iterable

def normalize_answer(text: str) -> str:
    """Lowercase and remove punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)

def metric_max_over_ground_truths(prediction: str, ground_truths: Iterable[str], metric_fn) -> float:
    scores = [metric_fn(prediction, ground_truth) for ground_truth in ground_truths]
    return max(scores) if scores else 0.0

def squad_exact_match(prediction: str, answers: dict) -> float:
    return metric_max_over_ground_truths(prediction, answers.get("text", []), exact_match_score)

def squad_f1(prediction: str, answers: dict) -> float:
    return metric_max_over_ground_truths(prediction, answers.get("text", []), f1_score)
