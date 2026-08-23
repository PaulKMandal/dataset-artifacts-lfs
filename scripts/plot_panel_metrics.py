#!/usr/bin/env python3
"""Create lightweight paper-review figures from a panel main table.

This is intentionally conservative: it reads the aggregate CSVs already produced
by scripts/aggregate_metrics.py and writes PNGs plus a normalized plotting CSV.
It does not recompute metrics and does not require raw predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EVALSETS = [
    ("squad_dev", "SQuAD dev"),
    ("addsent", "AddSent"),
    ("addonesent", "AddOneSent"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-table", required=True, help="Path to metrics/main_table.csv")
    parser.add_argument("--seed-table", default=None, help="Optional path to metrics/seed_level_metrics.csv")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def condition_label(row: pd.Series) -> str:
    subset = str(row.get("train_subset", "unknown"))
    frac = row.get("subset_fraction", "")
    budget = str(row.get("train_budget_type", "same_epochs"))
    if subset == "full":
        label = "full"
    elif subset == "random":
        label = f"random {frac:g}" if isinstance(frac, float) else "random"
    else:
        label = f"{subset} {frac:g}" if isinstance(frac, float) else subset
    if budget != "same_epochs":
        label += f"\n{budget}"
    return label


def normalized_plot_rows(main: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in main.iterrows():
        label = condition_label(row)
        for evalset, eval_label in EVALSETS:
            f1_col = f"{evalset}_f1_mean"
            em_col = f"{evalset}_em_mean"
            if f1_col not in main.columns:
                continue
            rows.append(
                {
                    "model": row.get("model"),
                    "train_subset": row.get("train_subset"),
                    "subset_fraction": row.get("subset_fraction"),
                    "train_budget_type": row.get("train_budget_type"),
                    "condition_label": label,
                    "evalset": evalset,
                    "evalset_label": eval_label,
                    "f1_mean": row.get(f1_col),
                    "em_mean": row.get(em_col) if em_col in main.columns else None,
                    "n_seeds": row.get("n_seeds"),
                    "n_random_draws": row.get("n_random_draws"),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bars(df: pd.DataFrame, *, out_path: Path, title: str, evalsets: list[str]) -> None:
    sub = df[df["evalset"].isin(evalsets)].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="condition_label", columns="evalset_label", values="f1_mean", aggfunc="first")
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(max(8, 0.8 * len(pivot)), 5), rot=35)
    ax.set_ylabel("F1")
    ax.set_xlabel("Training condition")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Eval set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_random_distribution(seed_table: Path, out_dir: Path) -> None:
    if not seed_table or not seed_table.exists():
        return
    df = pd.read_csv(seed_table)
    needed = {"train_subset", "evalset", "f1"}
    if not needed.issubset(df.columns):
        return
    random_df = df[df["train_subset"].astype(str) == "random"].copy()
    if random_df.empty:
        return
    for evalset, eval_label in EVALSETS:
        vals = random_df[random_df["evalset"] == evalset]["f1"].dropna()
        if vals.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=min(20, max(5, len(vals) // 2)))
        plt.xlabel("F1")
        plt.ylabel("Count")
        plt.title(f"Random-subset F1 distribution: {eval_label}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"random_f1_distribution_{evalset}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_table = Path(args.main_table)
    main = pd.read_csv(main_table)
    plot_rows = normalized_plot_rows(main)
    plot_rows.to_csv(out_dir / "panel_plot_rows.csv", index=False)

    plot_grouped_bars(
        plot_rows,
        out_path=out_dir / "f1_by_condition_all_evalsets.png",
        title="F1 by training condition",
        evalsets=["squad_dev", "addsent", "addonesent"],
    )
    plot_grouped_bars(
        plot_rows,
        out_path=out_dir / "adversarial_f1_by_condition.png",
        title="Adversarial F1 by training condition",
        evalsets=["addsent", "addonesent"],
    )
    seed_table = Path(args.seed_table) if args.seed_table else None
    plot_random_distribution(seed_table, out_dir)
    print(f"Wrote panel figures to {out_dir}")


if __name__ == "__main__":
    main()
