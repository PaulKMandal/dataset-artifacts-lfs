#!/usr/bin/env python3
"""Aggregate multi-task main-track metrics and formulation interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLUMNS = [
    "task",
    "model_key",
    "family",
    "train_subset",
    "subset_fraction",
    "train_budget_type",
    "selection_map",
    "evalset",
]


def load_metrics(path: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(path.glob("*.json")):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    if not rows:
        raise ValueError(f"No normalized metrics found in {path}")
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, seed: int, n: int = 10000) -> tuple[float, float]:
    array = values.dropna().astype(float).to_numpy()
    if not len(array):
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [metric for metric in ["exact_match", "f1", "accuracy"] if metric in frame]
    for row_number, (keys, group) in enumerate(frame.groupby(GROUP_COLUMNS, dropna=False)):
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        row["n_runs"] = group["train_run_id"].nunique()
        row["n_seeds"] = group["seed"].nunique()
        row["n_subset_draws"] = group["subset_draw_id"].dropna().nunique()
        for metric in metric_columns:
            values = group[metric].dropna().astype(float)
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            low, high = bootstrap_ci(values, seed=20260823 + row_number)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS)


def condition_seed_means(
    frame: pd.DataFrame,
    *,
    task: str,
    model_key: str,
    subset: str,
    evalset: str,
    metric: str,
) -> pd.Series:
    selected = frame[
        (frame["task"] == task)
        & (frame["model_key"] == model_key)
        & (frame["train_subset"] == subset)
        & (frame["evalset"] == evalset)
        & (frame["train_budget_type"] == "same_epochs")
    ]
    if subset == "ambiguous":
        selected = selected[selected["family"] == "primary"]
    elif subset == "full":
        selected = selected[selected["family"] == "full_baseline"]
    return selected.groupby("seed")[metric].mean()


def formulation_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(set(frame["model_key"]))
    qa_evalsets = sorted(set(frame.loc[frame["task"] == "qa", "evalset"]))
    for model_key in models:
        for evalset in qa_evalsets:
            qa_full = condition_seed_means(
                frame,
                task="qa",
                model_key=model_key,
                subset="full",
                evalset=evalset,
                metric="exact_match",
            )
            qa_ambiguous = condition_seed_means(
                frame,
                task="qa",
                model_key=model_key,
                subset="ambiguous",
                evalset=evalset,
                metric="exact_match",
            )
            cls_full = condition_seed_means(
                frame,
                task="classification",
                model_key=model_key,
                subset="full",
                evalset=evalset,
                metric="accuracy",
            )
            cls_ambiguous = condition_seed_means(
                frame,
                task="classification",
                model_key=model_key,
                subset="ambiguous",
                evalset=evalset,
                metric="accuracy",
            )
            common = (
                qa_full.index.intersection(qa_ambiguous.index)
                .intersection(cls_full.index)
                .intersection(cls_ambiguous.index)
            )
            if common.empty:
                continue
            qa_delta = qa_ambiguous.loc[common] - qa_full.loc[common]
            cls_delta = cls_ambiguous.loc[common] - cls_full.loc[common]
            interaction = qa_delta - cls_delta
            rows.append(
                {
                    "model_key": model_key,
                    "evalset": evalset,
                    "n_paired_seeds": len(common),
                    "qa_ambiguous_minus_full_em": float(qa_delta.mean()),
                    "classification_ambiguous_minus_full_accuracy": float(cls_delta.mean()),
                    "task_interaction_points": float(interaction.mean()),
                    "interaction_std": float(interaction.std(ddof=1))
                    if len(interaction) > 1
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_metrics(Path(args.metrics_dir))
    frame.to_csv(out / "seed_level_metrics.csv", index=False)
    summary = summary_table(frame)
    summary.to_csv(out / "maintrack_summary.csv", index=False)
    (out / "maintrack_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8"
    )
    interactions = formulation_interactions(frame)
    interactions.to_csv(out / "task_formulation_interactions.csv", index=False)
    print(f"Wrote {out / 'seed_level_metrics.csv'}")
    print(f"Wrote {out / 'maintrack_summary.csv'}")
    print(f"Wrote {out / 'task_formulation_interactions.csv'}")


if __name__ == "__main__":
    main()
