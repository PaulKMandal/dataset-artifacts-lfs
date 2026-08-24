#!/usr/bin/env python3
"""Build a one-to-one QNLI-style formulation from SQuAD-style QA records.

Training emits exactly one binary sentence-classification row per source QA
example and preserves its ``idx``. Labels are exactly balanced (up to one row)
using a stable hash. Evaluation emits a positive/negative pair per QA example.
This keeps the source text and example identity fixed while changing only the
output structure from span prediction to sentence-pair classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_metrics import normalize_answer  # noqa: E402

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if "idx" not in row:
                    row["idx"] = len(rows)
                if not row.get("answers", {}).get("text"):
                    raise ValueError(f"{path}:{line_number} has no answer text")
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def answer_spans(row: dict) -> list[tuple[int, int, str]]:
    answers = row["answers"]
    texts = list(answers.get("text", []))
    starts = [int(start) for start in answers.get("answer_start", [])]
    if not texts or len(texts) != len(starts):
        raise ValueError(f"Invalid answer annotations for {row['id']!r}")

    context = row["context"]
    spans = []
    for text, start in zip(texts, starts, strict=True):
        end = start + len(text)
        if start < 0 or context[start:end] != text:
            raise ValueError(
                f"Invalid answer span [{start}, {end}) for {row['id']!r}: {text!r}"
            )
        spans.append((start, end, text))
    return spans


def sentence_spans(
    text: str, *, protected_spans: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ()
) -> list[tuple[int, int, str]]:
    """Split text into sentence spans without cutting protected gold spans.

    ``protected_spans`` are character ranges that must remain inside a single
    candidate sentence. This is a final safety invariant for extractive-QA gold
    answers: a candidate boundary cannot legitimately bisect a contiguous gold
    answer string.
    """
    spans = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        boundary = match.start()
        if any(span_start < boundary < span_end for span_start, span_end in protected_spans):
            continue

        end = boundary
        raw_sentence = text[start:end]
        sentence = raw_sentence.strip()
        if sentence:
            left_trim = len(raw_sentence) - len(raw_sentence.lstrip())
            spans.append((start + left_trim, end, sentence))
        start = match.end()

    raw_tail = text[start:]
    tail = raw_tail.strip()
    if tail:
        left_trim = len(raw_tail) - len(raw_tail.lstrip())
        spans.append((start + left_trim, len(text), tail))
    return spans or [(0, len(text), text)]


def row_sentence_spans(row: dict) -> list[tuple[int, int, str]]:
    protected = [(start, end) for start, end, _ in answer_spans(row)]
    return sentence_spans(row["context"], protected_spans=protected)


def answer_sentence(row: dict) -> tuple[str, int]:
    start, end, answer = answer_spans(row)[0]
    for sentence_index, (sentence_start, sentence_end, sentence) in enumerate(
        row_sentence_spans(row)
    ):
        if sentence_start <= start and end <= sentence_end:
            return sentence, sentence_index
    raise ValueError(f"Answer {answer!r} is not contained in any sentence for {row['id']!r}")


def word_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def lexical_overlap(question: str, sentence: str) -> float:
    question_words = word_set(question)
    sentence_words = word_set(sentence)
    if not question_words or not sentence_words:
        return 0.0
    return len(question_words & sentence_words) / len(question_words | sentence_words)


def fallback_pool(rows: list[dict]) -> list[tuple[str, str]]:
    return [(row["id"], answer_sentence(row)[0]) for row in rows]


def negative_sentence(row: dict, pool: list[tuple[str, str]]) -> tuple[str, str]:
    gold_spans = [(start, end) for start, end, _ in answer_spans(row)]
    candidates = [
        sentence
        for sentence_start, sentence_end, sentence in row_sentence_spans(row)
        if not any(
            sentence_start < gold_end and gold_start < sentence_end
            for gold_start, gold_end in gold_spans
        )
    ]
    if candidates:
        candidates.sort(
            key=lambda sentence: (-lexical_overlap(row["question"], sentence), sentence)
        )
        return candidates[0], "same_context_hard_negative"

    for source_id, sentence in pool:
        if source_id != row["id"]:
            return sentence, "cross_context_fallback"
    raise ValueError("Cannot construct a negative sentence from a one-example dataset")


def stable_positive_ids(rows: list[dict], seed: int) -> set[str]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    return {row["id"] for row in ranked[: (len(ranked) + 1) // 2]}


def make_record(
    row: dict,
    *,
    candidate_sentence: str,
    label: int,
    candidate_kind: str,
    record_id: str,
    idx: int,
) -> dict:
    return {
        "id": record_id,
        "idx": int(idx),
        "source_idx": int(row["idx"]),
        "source_example_id": row["id"],
        "title": row.get("title", ""),
        "context": row["context"],
        "question": row["question"],
        "answers": row["answers"],
        "candidate_sentence": candidate_sentence,
        "candidate_kind": candidate_kind,
        "label": int(label),
        "task_formulation": "squad_sentence_classification",
    }


def build_train(rows: list[dict], seed: int) -> list[dict]:
    pool = fallback_pool(rows)
    positive_ids = stable_positive_ids(rows, seed)
    output = []
    for row in rows:
        if row["id"] in positive_ids:
            candidate, kind, label = answer_sentence(row)[0], "answer_sentence", 1
        else:
            candidate, kind = negative_sentence(row, pool)
            label = 0
        output.append(
            make_record(
                row,
                candidate_sentence=candidate,
                label=label,
                candidate_kind=kind,
                record_id=row["id"],
                idx=int(row["idx"]),
            )
        )
    return output


def build_eval(rows: list[dict]) -> list[dict]:
    pool = fallback_pool(rows)
    output = []
    for row in rows:
        positive = answer_sentence(row)[0]
        negative, negative_kind = negative_sentence(row, pool)
        for suffix, candidate, kind, label in [
            ("pos", positive, "answer_sentence", 1),
            ("neg", negative, negative_kind, 0),
        ]:
            output.append(
                make_record(
                    row,
                    candidate_sentence=candidate,
                    label=label,
                    candidate_kind=kind,
                    record_id=f"{row['id']}::{suffix}",
                    idx=len(output),
                )
            )
    return output


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH")
    return name, Path(path)


def manifest_row(source: Path, output: Path, rows: list[dict]) -> dict:
    kinds = Counter(row["candidate_kind"] for row in rows)
    labels = Counter(str(row["label"]) for row in rows)
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "num_examples": len(rows),
        "labels": dict(sorted(labels.items())),
        "candidate_kinds": dict(sorted(kinds.items())),
    }


def _collect_materialization_errors(
    datasets: list[tuple[str, list[dict]]], *, max_reported: int = 20
) -> None:
    """Fail once with a corpus-wide diagnostic instead of one row at a time."""
    failures: list[str] = []
    total_failures = 0
    for dataset_name, rows in datasets:
        pool = []
        valid_answer_ids = set()
        for row in rows:
            try:
                pool.append((row["id"], answer_sentence(row)[0]))
                valid_answer_ids.add(row["id"])
            except ValueError as exc:
                total_failures += 1
                if len(failures) < max_reported:
                    failures.append(f"{dataset_name}:{row['id']}: {exc}")

        for row in rows:
            if row["id"] not in valid_answer_ids:
                continue
            try:
                negative_sentence(row, pool)
            except ValueError as exc:
                total_failures += 1
                if len(failures) < max_reported:
                    failures.append(f"{dataset_name}:{row['id']}: {exc}")

    if total_failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        omitted = total_failures - len(failures)
        if omitted:
            details += f"\n  - ... {omitted} additional failures omitted"
        raise ValueError(
            f"Sentence-classification preflight found {total_failures} materialization failures:\n"
            f"{details}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--evalset", action="append", default=[], type=parse_named_path)
    parser.add_argument("--out-dir", default="data/sentence_classification")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_source = Path(args.train_qa)
    train_source_rows = read_jsonl(train_source)
    eval_sources = [(name, source, read_jsonl(source)) for name, source in args.evalset]

    _collect_materialization_errors(
        [("squad_train", train_source_rows)]
        + [(name, rows) for name, _, rows in eval_sources]
    )

    train_rows = build_train(train_source_rows, args.seed)
    train_out = out_dir / "squad_train.jsonl"
    write_jsonl(train_out, train_rows)
    manifest = {"squad_train": manifest_row(train_source, train_out, train_rows)}

    for name, source, source_rows in eval_sources:
        eval_rows = build_eval(source_rows)
        output = out_dir / f"{name}.jsonl"
        write_jsonl(output, eval_rows)
        manifest[name] = manifest_row(source, output, eval_rows)

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
