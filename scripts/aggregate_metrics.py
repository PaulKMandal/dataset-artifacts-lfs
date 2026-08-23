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


KEY_COLS = ["model", "train_subset", "subset_fraction", "subset_draw_id", "seed", "train_budget_type"]
GROUP_COLS = ["model", "train_subset", "subset_fraction", "train_budget_type", "confidence_definition"]


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["metrics_path"] = str(path)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No JSON metrics found in {metrics_dir}")
    return pd.DataFrame(rows)


def add_adversarial_drops(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in KEY_COLS if c in df.columns]
    work = df.copy()
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
    long = work.merge(pivot[helper_cols + drop_cols], on=helper_cols, how="left")
    return long.drop(columns=helper_cols)


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
        for evalset in sorted(group["evalset"].dropna().unique()):
            sub = group[group["evalset"] == evalset]
            prefix = evalset.lower()
            for metric, suffix in [("exact_match", "em"), ("f1", "f1")]:
                vals = sub[metric].dropna().astype(float)
                row[f"{prefix}_{suffix}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{prefix}_{suffix}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(vals)
                row[f"{prefix}_{suffix}_ci_low"] = ci_low
                row[f"{prefix}_{suffix}_ci_high"] = ci_high
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
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_dir)
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
