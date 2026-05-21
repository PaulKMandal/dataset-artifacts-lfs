#!/usr/bin/env python3
"""Aggregate run-level metric JSON files into paper tables.

Usage:
  python scripts/aggregate_metrics.py --metrics-dir results/metrics/raw --out-dir results/metrics

Expected input: one JSON file per train-run/evalset with fields described in
RESULTS_SCHEMA.md. The output tables are generated only from saved metrics; no
manual table editing is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = [
    "model",
    "train_subset",
    "subset_protocol",
    "subset_fraction",
    "subset_draw_id",
    "seed",
    "train_budget_type",
]
GROUP_COLS = [
    "model",
    "train_subset",
    "subset_protocol",
    "subset_fraction",
    "train_budget_type",
    "confidence_definition",
]


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["metrics_path"] = str(path)
        row["exact_match"] = row.get("exact_match", row.get("eval_exact_match"))
        row["f1"] = row.get("f1", row.get("eval_f1"))
        row["metric_split"] = "all_rows"
        rows.append(row)
    if not rows:
        raise SystemExit(f"No JSON metrics found in {metrics_dir}")
    return pd.DataFrame(rows)




def add_split_metrics(raw: pd.DataFrame, split_metrics_path: Path | None) -> pd.DataFrame:
    if split_metrics_path is None or not split_metrics_path.exists():
        return raw
    splits = pd.read_csv(split_metrics_path)
    if splits.empty:
        return raw
    splits["train_run_id"] = splits["prediction_file"].map(
        lambda name: Path(str(name)).stem.rsplit("__", 1)[0]
    )
    meta_cols = [
        c for c in raw.columns
        if c not in {"exact_match", "f1", "num_eval_examples", "metric_split"}
    ]
    meta = raw[meta_cols].drop_duplicates(["train_run_id", "evalset"])
    split_rows = splits.merge(meta, on=["train_run_id", "evalset"], how="left")
    split_rows = split_rows.rename(columns={"split": "metric_split", "n": "num_eval_examples"})
    split_rows = split_rows[split_rows["metric_split"] != "all_rows"]
    wanted = list(dict.fromkeys(list(raw.columns) + list(split_rows.columns)))
    return pd.concat([raw.reindex(columns=wanted), split_rows.reindex(columns=wanted)], ignore_index=True)


def add_adversarial_drops(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in KEY_COLS if c in df.columns]
    work = df[df.get("metric_split", "all_rows") == "all_rows"].copy()
    helper_cols = []
    for col in key_cols:
        helper = f"__key_{col}"
        helper_cols.append(helper)
        work[helper] = work[col].astype(object).where(work[col].notna(), "__NONE__")

    pivot = (
        work.sort_values("evalset")
        .drop_duplicates(helper_cols + ["evalset"], keep="first")
        .set_index(helper_cols + ["evalset"])[["exact_match", "f1"]]
        .unstack("evalset")
    )
    pivot.columns = [f"{metric}_{evalset}" for metric, evalset in pivot.columns]
    pivot = pivot.reset_index()

    for evalset in ["addsent", "addonesent"]:
        if f"f1_squad_dev" in pivot and f"f1_{evalset}" in pivot:
            pivot[f"{evalset}_drop_f1"] = pivot["f1_squad_dev"] - pivot[f"f1_{evalset}"]
        if f"exact_match_squad_dev" in pivot and f"exact_match_{evalset}" in pivot:
            pivot[f"{evalset}_drop_em"] = pivot["exact_match_squad_dev"] - pivot[f"exact_match_{evalset}"]

    drop_cols = [c for c in pivot.columns if c.endswith("_drop_f1") or c.endswith("_drop_em")]
    drops = work.merge(pivot[helper_cols + drop_cols], on=helper_cols, how="left")
    drop_only = drops[helper_cols + drop_cols].drop_duplicates(helper_cols)
    full = df.copy()
    for col, helper in zip(key_cols, helper_cols):
        full[helper] = full[col].astype(object).where(full[col].notna(), "__NONE__")
    full = full.merge(drop_only, on=helper_cols, how="left")
    return full.drop(columns=helper_cols)


def bootstrap_ci(values: pd.Series, n: int = 10000, seed: int = 12345) -> tuple[float, float]:
    vals = values.dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def make_main_table(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in GROUP_COLS if c in df.columns]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = group["seed"].nunique() if "seed" in group else None
        if "subset_draw_id" in group:
            row["n_random_draws"] = group["subset_draw_id"].dropna().nunique()
        else:
            row["n_random_draws"] = None
        split_col = "metric_split" if "metric_split" in group.columns else None
        split_groups = group.groupby(["evalset", split_col], dropna=False) if split_col else group.groupby("evalset", dropna=False)
        for keys2, sub in split_groups:
            if split_col:
                evalset, metric_split = keys2
            else:
                evalset, metric_split = keys2, "all_rows"
            prefix = f"{str(evalset).lower()}__{str(metric_split).lower()}"
            for metric, suffix in [("exact_match", "em"), ("f1", "f1")]:
                vals = sub[metric].dropna().astype(float)
                row[f"{prefix}__{suffix}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{prefix}__{suffix}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(vals)
                row[f"{prefix}__{suffix}_ci_low"] = ci_low
                row[f"{prefix}__{suffix}_ci_high"] = ci_high
        for drop_col in ["addsent_drop_f1", "addonesent_drop_f1", "addsent_drop_em", "addonesent_drop_em"]:
            if drop_col in group:
                vals = group[drop_col].dropna().astype(float)
                if len(vals):
                    row[f"{drop_col}_mean"] = vals.mean()
                    row[f"{drop_col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    ci_low, ci_high = bootstrap_ci(vals)
                    row[f"{drop_col}_ci_low"] = ci_low
                    row[f"{drop_col}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def make_random_subset_distribution(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "model",
        "subset_fraction",
        "subset_draw_id",
        "seed",
        "train_budget_type",
        "confidence_definition",
        "evalset",
        "exact_match",
        "f1",
        "metric_split",
    ]
    available = [c for c in needed if c in df.columns]
    random_df = df[df["train_subset"] == "random"].copy()
    if random_df.empty:
        return pd.DataFrame(columns=needed)
    return random_df[available].sort_values([c for c in available if c not in {"exact_match", "f1"}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-split-metrics", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_dir)
    split_path = Path(args.eval_split_metrics) if args.eval_split_metrics else None
    df = add_split_metrics(df, split_path)
    df = add_adversarial_drops(df)
    df.to_csv(out_dir / "seed_level_metrics.csv", index=False)

    main_table = make_main_table(df)
    main_table.to_csv(out_dir / "main_table.csv", index=False)
    with (out_dir / "main_table.json").open("w", encoding="utf-8") as f:
        json.dump(main_table.to_dict(orient="records"), f, indent=2)

    random_dist = make_random_subset_distribution(df)
    random_dist.to_csv(out_dir / "random_subset_distribution.csv", index=False)

    confidence_ablation = df[df.get("confidence_definition", "").notna()] if "confidence_definition" in df else pd.DataFrame()
    if not confidence_ablation.empty:
        confidence_ablation.to_csv(out_dir / "confidence_definition_ablation.csv", index=False)

    print(f"Wrote {out_dir / 'seed_level_metrics.csv'}")
    print(f"Wrote {out_dir / 'main_table.csv'}")
    print(f"Wrote {out_dir / 'main_table.json'}")
    print(f"Wrote {out_dir / 'random_subset_distribution.csv'}")


if __name__ == "__main__":
    main()
