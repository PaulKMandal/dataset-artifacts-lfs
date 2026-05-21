#!/usr/bin/env python3
"""Build per-example clean-to-adversarial failure features.

This script aligns raw prediction JSONL files by SQuAD example id. It does not
pretend that train-set cartography scores directly exist for dev/adversarial
items; cartography columns are left blank unless a provided cartography file has
matching example IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_metrics import normalize_answer
from scripts.normalize_question_type import normalize_question_type

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-predictions", required=True)
    parser.add_argument("--adversarial-predictions", nargs="+", required=True)
    parser.add_argument("--cartography-scores", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def word_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())

def overlap(a: str, b: str) -> float:
    aw = word_set(a)
    bw = word_set(b)
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)

def sentences(text: str) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        sent = text[start:end].strip()
        if sent:
            spans.append((start, end, sent))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        spans.append((start, len(text), tail))
    return spans

def answer_sentence(context: str, answer_start: int | None) -> str:
    if answer_start is None:
        return ""
    for start, end, sent in sentences(context):
        if start <= answer_start <= end:
            return sent
    return ""

def added_sentences(clean_context: str, adv_context: str) -> list[str]:
    clean = {normalize_answer(sent) for _, _, sent in sentences(clean_context)}
    return [sent for _, _, sent in sentences(adv_context) if normalize_answer(sent) not in clean]

def answer_info(row: dict) -> tuple[str, int | None]:
    answers = row.get("answers", {})
    texts = answers.get("text", []) if isinstance(answers, dict) else []
    starts = answers.get("answer_start", []) if isinstance(answers, dict) else []
    text = texts[0] if texts else ""
    start = int(starts[0]) if starts else None
    return text, start

def load_cartography_by_id(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row.get("example_id", row.get("id", "")): row for row in reader if row.get("example_id", row.get("id", ""))}

def mechanism_row(clean: dict, adv: dict, evalset: str, cart: dict) -> dict:
    answer_text, answer_start = answer_info(clean)
    context_length = len(clean.get("context", "").split())
    answer_length = len(answer_text.split())
    answer_position_normalized = answer_start / max(len(clean.get("context", "")), 1) if answer_start is not None else math.nan
    ans_sent = answer_sentence(clean.get("context", ""), answer_start)
    added = added_sentences(clean.get("context", ""), adv.get("context", ""))
    q = clean.get("question", "")
    baseline_correct = float(clean.get("exact_match", 0.0))
    adversarial_correct = float(adv.get("exact_match", 0.0))
    return {
        "example_id": adv["id"],
        "evalset": evalset,
        "baseline_correct": baseline_correct,
        "adversarial_correct": adversarial_correct,
        "adversarial_failure": float(baseline_correct >= 1.0 and adversarial_correct < 1.0),
        "confidence": cart.get("confidence", ""),
        "variability": cart.get("variability", ""),
        "correctness": cart.get("correctness", ""),
        "context_length": context_length,
        "answer_length": answer_length,
        "answer_position_normalized": answer_position_normalized,
        "question_type": normalize_question_type(q),
        "question_answer_sentence_overlap": overlap(q, ans_sent),
        "question_distractor_overlap": max([overlap(q, sent) for sent in added], default=0.0),
        "num_added_sentences": len(added),
    }

def build_rows(args: Namespace) -> list[dict]:
    clean_by_id = {row["id"]: row for row in read_jsonl(Path(args.clean_predictions))}
    cart_path = Path(args.cartography_scores) if args.cartography_scores else None
    cart_by_id = load_cartography_by_id(cart_path)
    out_rows = []
    for adv_path_str in args.adversarial_predictions:
        adv_path = Path(adv_path_str)
        evalset = adv_path.stem.split("__")[-1]
        for adv in read_jsonl(adv_path):
            clean = clean_by_id.get(adv["id"])
            if clean is not None:
                out_rows.append(mechanism_row(clean, adv, evalset, cart_by_id.get(adv["id"], {})))
    return out_rows

def write_rows(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise SystemExit("No aligned clean/adversarial rows were found")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    write_rows(build_rows(args), out_path)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
