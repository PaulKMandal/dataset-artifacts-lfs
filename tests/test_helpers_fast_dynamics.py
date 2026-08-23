import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_helpers_with_transformers_stub():
    fake_transformers = types.ModuleType("transformers")

    class DummyEvalPrediction:
        def __init__(self, predictions=None, label_ids=None):
            self.predictions = predictions
            self.label_ids = label_ids

    class DummyTrainer:
        pass

    fake_transformers.EvalPrediction = DummyEvalPrediction
    fake_transformers.Trainer = DummyTrainer

    old_transformers = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        spec = importlib.util.spec_from_file_location("_helpers_under_test", ROOT / "helpers.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = old_transformers


def test_resolve_max_seq_length_honors_cli_value():
    helpers = load_helpers_with_transformers_stub()
    tokenizer = SimpleNamespace(model_max_length=512)
    assert helpers._resolve_max_seq_length(tokenizer, 128) == 128
    assert helpers._resolve_max_seq_length(tokenizer, None) == 512

    sentinel_tokenizer = SimpleNamespace(model_max_length=10**30)
    assert helpers._resolve_max_seq_length(sentinel_tokenizer, None) == 384


def test_qa_dynamics_logger_records_scalars_not_full_vectors():
    helpers = load_helpers_with_transformers_stub()
    logger = object.__new__(helpers.DynamicsLogger)
    logger._dynamics_rows = []
    logger.state = SimpleNamespace(epoch=0.25, global_step=7)

    start_logits = torch.tensor([[0.0, 2.0, 0.0]])
    end_logits = torch.tensor([[0.0, 0.0, 3.0]])
    helpers.DynamicsLogger._append_qa_dynamics(
        logger,
        idxs=torch.tensor([11]),
        start_positions=torch.tensor([1]),
        end_positions=torch.tensor([2]),
        start_logits=start_logits,
        end_logits=end_logits,
        answer_in_window=torch.tensor([True]),
    )

    assert len(logger._dynamics_rows) == 1
    row = logger._dynamics_rows[0]
    assert "start_prob" not in row
    assert "end_prob" not in row
    assert row["idx"] == 11
    assert row["step"] == 7
    assert row["task"] == "qa"
    assert row["correctness"] == 1.0
    assert row["pred_start"] == 1
    assert row["pred_end"] == 2
    assert row["answer_in_window"] is True

    start_gold = torch.softmax(start_logits, dim=-1)[0, 1].item()
    end_gold = torch.softmax(end_logits, dim=-1)[0, 2].item()
    assert row["confidence"] == pytest.approx(0.5 * (start_gold + end_gold))
    assert row["joint_confidence"] == pytest.approx(start_gold * end_gold)
    assert math.isfinite(row["start_logp"])
    assert math.isfinite(row["end_logp"])


def test_sentence_pair_tokenization_infers_qnli_style_columns():
    helpers = load_helpers_with_transformers_stub()

    class DummyTokenizer:
        model_max_length = 512

        def __call__(self, text_a, text_b, **kwargs):
            assert text_a == ["Who wrote it?"]
            assert text_b == ["Ada wrote it."]
            assert kwargs["max_length"] == 64
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    output = helpers.prepare_dataset_nli(
        {
            "question": ["Who wrote it?"],
            "candidate_sentence": ["Ada wrote it."],
            "label": [1],
            "idx": [9],
        },
        DummyTokenizer(),
        max_seq_length=64,
    )

    assert output["labels"] == [1]
    assert output["idx"] == [9]


def test_compute_accuracy_handles_auxiliary_idx_labels():
    helpers = load_helpers_with_transformers_stub()
    prediction = SimpleNamespace(
        predictions=np.array([[0.1, 0.9], [0.8, 0.2]]),
        label_ids=(np.array([1, 0]), np.array([10, 11])),
    )
    assert helpers.compute_accuracy(prediction)["accuracy"] == 1.0
