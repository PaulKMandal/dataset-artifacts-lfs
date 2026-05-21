import json
from pathlib import Path

import pandas as pd

from scripts.audit_current_results import audit_prediction_splits, base_id, is_adversarial_id
from scripts.normalize_question_type import normalize_question_type


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_adversarial_id_helpers_handle_high_conf_and_turk():
    assert is_adversarial_id("abc-high-conf-1")
    assert is_adversarial_id("abc-turk3")
    assert base_id("abc-high-conf-1") == "abc"
    assert base_id("abc-turk3") == "abc"


def test_audit_prediction_splits_writes_adversarial_only(tmp_path):
    results = tmp_path / "results"
    out = results / "audit"
    out.mkdir(parents=True)
    write_jsonl(
        results / "predictions" / "run__addsent.jsonl",
        [
            {
                "id": "q1",
                "answers": {"text": ["Denver"], "answer_start": [0]},
                "predicted_answer": "Denver",
            },
            {
                "id": "q1-high-conf-0",
                "answers": {"text": ["Denver"], "answer_start": [0]},
                "predicted_answer": "Boston",
            },
        ],
    )
    warnings = audit_prediction_splits(results, out)
    df = pd.read_csv(out / "eval_split_metrics.csv")

    assert any("mixed original" in warning for warning in warnings)
    assert set(df["split"]) == {"all_rows", "original_rows_only", "adversarial_rows_only"}
    assert df[df["split"] == "adversarial_rows_only"]["exact_match"].item() == 0.0


def test_normalize_question_type_taxonomy():
    assert normalize_question_type("How many people lived there?") == "how_many"
    assert normalize_question_type("Forbes.com said what?") == "other"
