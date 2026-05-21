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
    "confidence_definition",
]
GROUP_COLS = [
    "model",
    "train_subset",
    "subset_protocol",
    "subset_fraction",
    "train_budget_type",
    "confidence_definition",
]
PAIRED_COLS = [
    "paired_base_questions",
    "adversarial_variants",
    "original_em_on_paired",
    "original_f1_on_paired",
    "adversarial_variant_em",
    "adversarial_variant_f1",
    "all_adv_variants_correct_rate",
    "any_adv_variant_correct_rate",
    "correct_original_but_any_adv_fail_rate",
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


def add_paired_metrics(df: pd.DataFrame, paired_metrics_path: Path | None) -> pd.DataFrame:
    if paired_metrics_path is None or not paired_metrics_path.exists():
        return df
    paired = pd.read_csv(paired_metrics_path)
    if paired.empty:
        return df
    paired["train_run_id"] = paired["prediction_file"].map(lambda name: Path(str(name)).stem.rsplit("__", 1)[0])
    keep = ["train_run_id", "evalset"] + [c for c in PAIRED_COLS if c in paired.columns]
    out = df.merge(paired[keep], on=["train_run_id", "evalset"], how="left")
    for col in PAIRED_COLS:
        if col in out.columns:
            out.loc[out["metric_split"] != "all_rows", col] = np.nan
    return out


def key_helper_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    key_cols = [c for c in KEY_COLS if c in df.columns]
    work = df.copy()
    helpers = []
    for col in key_cols:
        helper = f"__key_{col}"
        helpers.append(helper)
        work[helper] = work[col].astype(object).where(work[col].notna(), "__NONE__")
    return work, helpers


def add_adversarial_drops(df: pd.DataFrame) -> pd.DataFrame:
    work, helpers = key_helper_frame(df)
    work["eval_metric_key"] = work["evalset"].astype(str) + "__" + work["metric_split"].astype(str)
    pivot = (
        work.sort_values("eval_metric_key")
        .drop_duplicates(helpers + ["eval_metric_key"], keep="first")
        .set_index(helpers + ["eval_metric_key"])[["exact_match", "f1"]]
        .unstack("eval_metric_key")
    )
    pivot.columns = [f"{metric}_{eval_key}" for metric, eval_key in pivot.columns]
    pivot = pivot.reset_index()
    for evalset in ["addsent", "addonesent"]:
        for split in ["all_rows", "adversarial_rows_only"]:
            src_f1 = "f1_squad_dev__all_rows"
            adv_f1 = f"f1_{evalset}__{split}"
            src_em = "exact_match_squad_dev__all_rows"
            adv_em = f"exact_match_{evalset}__{split}"
            if src_f1 in pivot and adv_f1 in pivot:
                pivot[f"{evalset}_{split}_drop_f1"] = pivot[src_f1] - pivot[adv_f1]
            if src_em in pivot and adv_em in pivot:
                pivot[f"{evalset}_{split}_drop_em"] = pivot[src_em] - pivot[adv_em]
    drop_cols = [c for c in pivot.columns if c.endswith("_drop_f1") or c.endswith("_drop_em")]
    return work.merge(pivot[helpers + drop_cols], on=helpers, how="left").drop(columns=helpers)


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
        for evalset, sub in group[group["metric_split"] == "all_rows"].groupby("evalset", dropna=False):
            prefix = f"{str(evalset).lower()}__paired"
            for col in PAIRED_COLS:
                if col in sub.columns:
                    vals = sub[col].dropna().astype(float)
                    row[f"{prefix}__{col}_mean"] = vals.mean() if len(vals) else np.nan
                    row[f"{prefix}__{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    ci_low, ci_high = bootstrap_ci(vals)
                    row[f"{prefix}__{col}_ci_low"] = ci_low
                    row[f"{prefix}__{col}_ci_high"] = ci_high
        for drop_col in [c for c in group.columns if c.endswith("_drop_f1") or c.endswith("_drop_em")]:
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
    parser.add_argument("--paired-metrics", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_dir)
    split_path = Path(args.eval_split_metrics) if args.eval_split_metrics else None
    paired_path = Path(args.paired_metrics) if args.paired_metrics else None
    df = add_split_metrics(df, split_path)
    df = add_paired_metrics(df, paired_path)
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
