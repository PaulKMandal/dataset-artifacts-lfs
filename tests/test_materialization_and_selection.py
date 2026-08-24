from pathlib import Path

import pytest

from scripts.materialize_qa_data import local_records, mrqa_records, write_jsonl
from scripts.materialize_sentence_classification import build_eval, build_train
from scripts.select_qa_subsets import (
    proportional_quotas,
    select_coverage_constrained_ambiguous,
)


def qa_rows():
    return [
        {
            "id": "q0",
            "idx": 0,
            "title": "A",
            "context": "Ada wrote the program. Charles reviewed the notes.",
            "question": "Who wrote the program?",
            "answers": {"text": ["Ada"], "answer_start": [0]},
        },
        {
            "id": "q1",
            "idx": 1,
            "title": "A",
            "context": "The launch was Tuesday. Rain delayed the rehearsal.",
            "question": "When was the launch?",
            "answers": {"text": ["Tuesday"], "answer_start": [15]},
        },
        {
            "id": "q2",
            "idx": 2,
            "title": "B",
            "context": "The archive is in Rome. The copy is in Paris.",
            "question": "Where is the archive?",
            "answers": {"text": ["Rome"], "answer_start": [18]},
        },
        {
            "id": "q3",
            "idx": 3,
            "title": "B",
            "context": "One answer is short. Another sentence is longer.",
            "question": "What is short?",
            "answers": {"text": ["answer"], "answer_start": [4]},
        },
    ]


def test_matched_classification_is_balanced_and_preserves_source_indices():
    rows = qa_rows()
    train = build_train(rows, seed=1729)
    assert len(train) == len(rows)
    assert sorted(row["idx"] for row in train) == [0, 1, 2, 3]
    assert sum(row["label"] for row in train) == 2
    assert all(row["source_idx"] == row["idx"] for row in train)
    assert all(row["task_formulation"] == "squad_sentence_classification" for row in train)


def test_matched_classification_eval_emits_positive_negative_pairs():
    output = build_eval(qa_rows())
    assert len(output) == 8
    assert [row["label"] for row in output].count(1) == 4
    assert [row["label"] for row in output].count(0) == 4
    assert len({row["id"] for row in output}) == 8


def test_mrqa_records_find_all_answer_alias_spans():
    dataset = {
        "validation": [
            {
                "qid": "m0",
                "subset": "NewsQA",
                "context": "The answer is cobalt.",
                "question": "What is the answer?",
                "answers": ["cobalt", "not present"],
            }
        ]
    }
    output = list(mrqa_records(dataset, "validation"))
    assert output[0]["answers"] == {"text": ["cobalt"], "answer_start": [14]}


def test_local_jsonl_source_is_normalized(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"id":"q0","title":"T","context":"The answer is cobalt.",'
        '"question":"What is the answer?","answers":{"text":["cobalt"],'
        '"answer_start":[14]}}\n',
        encoding="utf-8",
    )
    assert list(local_records(source))[0]["answers"] == {
        "text": ["cobalt"],
        "answer_start": [14],
    }


def test_jsonl_write_is_atomic_when_validation_fails(tmp_path):
    destination = tmp_path / "qa.jsonl"
    destination.write_text("previous-good-data\n", encoding="utf-8")
    invalid = {
        "id": "q0",
        "title": "T",
        "context": "The answer is cobalt.",
        "question": "What is the answer?",
        "answers": {"text": ["cobalt"], "answer_start": [0]},
    }
    with pytest.raises(ValueError, match="Invalid answer span"):
        write_jsonl([invalid], destination)
    assert destination.read_text(encoding="utf-8") == "previous-good-data\n"
    assert not list(Path(tmp_path).glob("*.partial"))


def test_coverage_constrained_selection_uses_exact_proportional_quotas():
    by_idx = {row["idx"]: row for row in qa_rows()}
    scores = {
        idx: {
            "confidence": 0.5,
            "variability": float(idx),
            "correctness": 0.5,
        }
        for idx in by_idx
    }
    selected = select_coverage_constrained_ambiguous(by_idx, scores, k=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert sum(proportional_quotas({("a", "b", "c"): 3, ("d", "e", "f"): 1}, 2).values()) == 2
