import json

import pandas as pd
import pytest

from scripts.analyze_map_stability import ambiguous_indices, compare_maps, correlation
from scripts.combine_cartography_maps import consensus
from scripts.hierarchical_bootstrap import run_contrast


def map_frame(confidence, variability):
    return pd.DataFrame(
        {
            "idx": [0, 1, 2, 3],
            "confidence": confidence,
            "variability": variability,
            "correctness": [1.0, 0.5, 0.5, 0.0],
            "region": ["Easy-to-learn", "Ambiguous", "Ambiguous", "Hard-to-learn"],
        }
    ).set_index("idx")


def test_map_stability_reports_rank_and_selected_set_overlap():
    left = map_frame([0.9, 0.5, 0.4, 0.1], [0.1, 0.9, 0.8, 0.2])
    right = map_frame([0.8, 0.45, 0.5, 0.2], [0.2, 0.8, 0.9, 0.1])
    assert ambiguous_indices(left, 0.5) == {1, 2}
    rows = compare_maps("left", left, "right", right, [0.5])
    assert rows[0]["ambiguous_jaccard"] == 1.0
    assert rows[0]["n_common"] == 4
    assert -1.0 <= rows[0]["variability_spearman"] <= 1.0


def test_map_stability_uses_json_null_for_constant_correlations():
    constant = pd.Series([0.5, 0.5, 0.5])
    varying = pd.Series([0.1, 0.2, 0.3])
    assert correlation(constant, varying, "spearman") is None


def test_consensus_uses_aligned_percentile_ranks(tmp_path):
    first = map_frame([0.9, 0.5, 0.4, 0.1], [0.1, 0.9, 0.8, 0.2]).reset_index()
    second = map_frame([0.8, 0.45, 0.5, 0.2], [0.2, 0.8, 0.9, 0.1]).reset_index()
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    output = consensus([("first", first_path), ("second", second_path)], False)
    assert list(output["idx"]) == [0, 1, 2, 3]
    assert set(output["aggregation"]) == {"consensus_across_maps"}
    assert output["variability"].between(0, 1).all()


def write_prediction(path, values):
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(values):
            handle.write(json.dumps({"id": str(index), "exact_match": value, "f1": value}) + "\n")


def test_hierarchical_bootstrap_preserves_direction(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for condition, values in [("ambiguous", [1, 1, 1, 0]), ("random", [0, 0, 0, 1])]:
        for seed in [13, 21]:
            predictions = tmp_path / f"{condition}_{seed}.jsonl"
            write_prediction(predictions, values)
            metadata = {
                "train_subset": condition,
                "evalset": "addsent",
                "seed": seed,
                "subset_draw_id": None,
                "predictions_path": str(predictions),
            }
            (metrics_dir / f"{condition}_{seed}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

    result = run_contrast(
        metrics_dir,
        {
            "name": "ambiguous-minus-random",
            "metric": "exact_match",
            "common": {"evalset": "addsent"},
            "a": {"train_subset": "ambiguous"},
            "b": {"train_subset": "random"},
        },
        n=500,
        seed=7,
    )
    assert result["mean_difference_points"] == pytest.approx(50.0, abs=6.0)
    assert result["probability_a_gt_b"] > 0.9
