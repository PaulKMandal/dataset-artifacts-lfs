import json
from pathlib import Path

import pytest

from dynamics import (
    categorize_examples,
    compute_metrics,
    load_training_dynamics,
    save_cartography_csv,
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_scalar_dynamics_to_cartography(tmp_path):
    rows = [
        {"idx": 0, "confidence": 0.9, "correctness": 1.0},
        {"idx": 0, "confidence": 0.7, "correctness": 1.0},
        {"idx": 1, "confidence": 0.4, "correctness": 0.0},
    ]
    write_jsonl(tmp_path / "training_dynamics.jsonl", rows)

    dynamics = load_training_dynamics(tmp_path)
    metrics = compute_metrics(dynamics)
    categories = categorize_examples(metrics)

    assert metrics[0]["avg_confidence"] == pytest.approx(0.8)
    assert metrics[0]["variability"] == pytest.approx(0.1)
    assert metrics[0]["correctness"] == pytest.approx(1.0)
    assert categories[0] == "Easy-to-learn"
    assert categories[1] == "Hard-to-learn"

    out = save_cartography_csv(metrics, categories, tmp_path)
    assert out.exists()
    assert "cartography_scores.csv" in str(out)


def test_legacy_qa_vectors_are_supported(tmp_path):
    rows = [
        {
            "idx": 2,
            "start_position": 1,
            "end_position": 2,
            "start_prob": [0.1, 0.8, 0.1],
            "end_prob": [0.2, 0.1, 0.7],
        }
    ]
    write_jsonl(tmp_path / "training_dynamics.jsonl", rows)

    dynamics = load_training_dynamics(tmp_path)
    metrics = compute_metrics(dynamics)

    assert metrics[2]["avg_confidence"] == pytest.approx(0.75)
    assert metrics[2]["correctness"] == pytest.approx(1.0)


def test_joint_confidence_falls_back_for_scalar_nli_rows(tmp_path):
    rows = [
        {"idx": 3, "task": "nli", "confidence": 0.6, "correctness": 1.0},
    ]
    write_jsonl(tmp_path / "training_dynamics.jsonl", rows)

    dynamics = load_training_dynamics(tmp_path)
    metrics = compute_metrics(dynamics, confidence_field="joint_confidence")

    assert metrics[3]["avg_confidence"] == pytest.approx(0.6)
    assert metrics[3]["correctness"] == pytest.approx(1.0)


def test_answer_only_epoch_aggregation_removes_overflow_weighting(tmp_path):
    rows = [
        {
            "idx": 4,
            "epoch": 0.5,
            "joint_confidence": 0.2,
            "correctness": 0.0,
            "answer_in_window": False,
        },
        {
            "idx": 4,
            "epoch": 0.5,
            "joint_confidence": 0.8,
            "correctness": 1.0,
            "answer_in_window": True,
        },
        {
            "idx": 4,
            "epoch": 1.5,
            "joint_confidence": 0.6,
            "correctness": 1.0,
            "answer_in_window": True,
        },
    ]
    write_jsonl(tmp_path / "training_dynamics.jsonl", rows)

    dynamics = load_training_dynamics(tmp_path)
    metrics = compute_metrics(
        dynamics,
        confidence_field="joint_confidence",
        aggregation="answer_only_epoch_mean",
    )

    assert metrics[4]["n_records"] == 2
    assert metrics[4]["avg_confidence"] == pytest.approx(0.7)
    assert metrics[4]["variability"] == pytest.approx(0.1)
    assert metrics[4]["correctness"] == pytest.approx(1.0)


def test_epoch_aggregation_buckets_fractional_trainer_progress(tmp_path):
    rows = [
        {"idx": 8, "epoch": 0.1, "confidence": 0.2, "correctness": 0.0},
        {"idx": 8, "epoch": 0.8, "confidence": 0.8, "correctness": 1.0},
        {"idx": 8, "epoch": 1.2, "confidence": 0.6, "correctness": 1.0},
        {"idx": 8, "epoch": 1.9, "confidence": 1.0, "correctness": 1.0},
    ]
    write_jsonl(tmp_path / "training_dynamics.jsonl", rows)

    metrics = compute_metrics(load_training_dynamics(tmp_path), aggregation="epoch_mean")

    assert metrics[8]["n_records"] == 2
    assert metrics[8]["avg_confidence"] == pytest.approx(0.65)
    assert metrics[8]["variability"] == pytest.approx(0.15)
    assert metrics[8]["correctness"] == pytest.approx(0.75)


def test_answer_only_aggregation_requires_new_logger_field(tmp_path):
    write_jsonl(
        tmp_path / "training_dynamics.jsonl",
        [{"idx": 5, "confidence": 0.4, "correctness": 0.0}],
    )
    dynamics = load_training_dynamics(tmp_path)
    with pytest.raises(ValueError, match="answer_in_window"):
        compute_metrics(dynamics, aggregation="answer_only_records")
