import ast
from pathlib import Path

import pytest

from dataset_cache import map_with_cache_recovery


class CacheReadError(Exception):
    pass


class FakeDataset:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def map(self, function, **kwargs):
        self.calls.append((function, kwargs))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_map_retries_corrupt_cache_without_reusing_it(capsys):
    def transform(row):
        return row

    dataset = FakeDataset([CacheReadError("truncated Arrow stream"), "fresh-dataset"])

    result = map_with_cache_recovery(
        dataset,
        transform,
        cache_error_types=CacheReadError,
        description="evaluation feature preprocessing",
        batched=True,
        num_proc=2,
    )

    assert result == "fresh-dataset"
    assert len(dataset.calls) == 2
    assert "load_from_cache_file" not in dataset.calls[0][1]
    assert dataset.calls[1][1]["load_from_cache_file"] is False
    assert "unreadable Hugging Face Datasets Arrow cache" in capsys.readouterr().out


def test_map_does_not_hide_non_cache_error():
    class RealPreprocessingError(Exception):
        pass

    dataset = FakeDataset([RealPreprocessingError("bad source row")])
    with pytest.raises(RealPreprocessingError, match="bad source row"):
        map_with_cache_recovery(
            dataset,
            lambda row: row,
            cache_error_types=CacheReadError,
            description="training feature preprocessing",
        )
    assert len(dataset.calls) == 1


def test_map_does_not_loop_when_cache_was_already_disabled():
    dataset = FakeDataset([CacheReadError("still broken")])
    with pytest.raises(CacheReadError, match="still broken"):
        map_with_cache_recovery(
            dataset,
            lambda row: row,
            cache_error_types=CacheReadError,
            description="evaluation feature preprocessing",
            load_from_cache_file=False,
        )
    assert len(dataset.calls) == 1


def test_run_py_routes_every_dataset_map_through_recovery_helper():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "run.py").read_text(encoding="utf-8"))
    direct_map_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "map"
    ]
    assert direct_map_calls == []


def test_recovers_real_truncated_hf_arrow_cache(tmp_path):
    hf_datasets = pytest.importorskip("datasets")
    pa = pytest.importorskip("pyarrow")

    source = hf_datasets.Dataset.from_dict({"value": [1, 2, 3]})
    cache_path = tmp_path / "cache-corrupt.arrow"

    def add_one(batch):
        return {"mapped": [value + 1 for value in batch["value"]]}

    first = source.map(add_one, batched=True, cache_file_name=str(cache_path))
    assert first["mapped"] == [2, 3, 4]
    assert cache_path.stat().st_size > 0

    # Reproduce the crash artifact that caused the main-track AddSent failure:
    # an existing HF Dataset.map Arrow cache file truncated to zero bytes.
    cache_path.write_bytes(b"")

    recovered = map_with_cache_recovery(
        source,
        add_one,
        cache_error_types=pa.ArrowInvalid,
        description="real truncated-cache regression",
        batched=True,
        cache_file_name=str(cache_path),
    )

    assert recovered["mapped"] == [2, 3, 4]
    assert cache_path.stat().st_size > 0


def test_map_routes_derived_cache_to_ephemeral_scratch(monkeypatch, tmp_path):
    dataset = FakeDataset(["mapped"])
    monkeypatch.setenv("DATASET_ARTIFACTS_MAP_CACHE_DIR", str(tmp_path))

    def transform(row):
        return row

    result = map_with_cache_recovery(
        dataset,
        transform,
        cache_error_types=CacheReadError,
        description="evaluation feature preprocessing",
        batched=True,
    )

    assert result == "mapped"
    cache_file = Path(dataset.calls[0][1]["cache_file_name"])
    assert cache_file.parent == tmp_path
    assert cache_file.name == "evaluation-feature-preprocessing.arrow"
