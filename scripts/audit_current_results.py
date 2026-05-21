#!/usr/bin/env python3
"""Audit round-2 cartography results before paper reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_metrics import squad_exact_match, squad_f1

ADVERSARIAL_ID_RE = re.compile(r"-(?:high-conf|turk)")
EXPECTED_EVALSETS = {"squad_dev", "addsent", "addonesent"}


def is_adversarial_id(example_id: str) -> bool:
    """Return true for known Adversarial SQuAD variant id patterns."""
    return bool(ADVERSARIAL_ID_RE.search(str(example_id)))


def base_id(example_id: str) -> str:
    """Map an adversarial variant id to the base SQuAD question id."""
    return ADVERSARIAL_ID_RE.split(str(example_id), maxsplit=1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metric(row: dict[str, Any], name: str) -> Any:
    return row.get(name, row.get(f"eval_{name}"))


def audit_raw_metrics(results_dir: Path, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    rows = []
    for path in sorted((results_dir / "metrics" / "raw").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        rows.append(
            {
                "metrics_file": path.name,
                "train_run_id": row.get("train_run_id"),
                "model_short": row.get("model_short"),
                "model": row.get("model"),
                "train_subset": row.get("train_subset"),
                "subset_protocol": row.get("subset_protocol"),
                "subset_fraction": row.get("subset_fraction"),
                "subset_draw_id": row.get("subset_draw_id"),
                "seed": row.get("seed"),
                "train_budget_type": row.get("train_budget_type"),
                "evalset": row.get("evalset"),
                "exact_match": _metric(row, "exact_match"),
                "f1": _metric(row, "f1"),
                "num_eval_examples": row.get("num_eval_examples", row.get("eval_samples")),
                "dataset_path": row.get("dataset_path"),
                "dataset_hash": row.get("dataset_hash"),
                "created_at_utc": row.get("created_at_utc"),
                "predictions_path": row.get("predictions_path"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "completed_raw_metrics.csv", index=False)
    if df.empty:
        warnings.append("No raw metric JSON files found.")
        return warnings
    for train_run_id, group in df.groupby("train_run_id", dropna=False):
        missing = EXPECTED_EVALSETS - set(group["evalset"].dropna())
        if missing:
            warnings.append(f"Run {train_run_id} missing evalsets: {sorted(missing)}")
    return warnings


def _prediction_scores(row: dict[str, Any]) -> tuple[float, float]:
    if "exact_match" in row and "f1" in row:
        return float(row["exact_match"]), float(row["f1"])
    if "predicted_answer" not in row or "answers" not in row:
        return 0.0, 0.0
    pred = str(row.get("predicted_answer", ""))
    return squad_exact_match(pred, row["answers"]), squad_f1(pred, row["answers"])


def _split_prediction_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orig_by_base: dict[str, dict[str, Any]] = {}
    adv_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        em, f1 = _prediction_scores(row)
        row["_em"] = em
        row["_f1"] = f1
        ex_id = str(row.get("id", ""))
        groups["all_rows"].append(row)
        if is_adversarial_id(ex_id):
            groups["adversarial_rows_only"].append(row)
            adv_by_base[base_id(ex_id)].append(row)
        else:
            groups["original_rows_only"].append(row)
            orig_by_base[base_id(ex_id)] = row
    return groups, orig_by_base, adv_by_base


def _split_metric_row(path: Path, evalset: str, split: str, vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prediction_file": path.name,
        "evalset": evalset,
        "split": split,
        "n": len(vals),
        "exact_match": 100.0 * sum(r["_em"] for r in vals) / len(vals),
        "f1": 100.0 * sum(r["_f1"] for r in vals) / len(vals),
    }


def _paired_metric_row(
    path: Path,
    evalset: str,
    paired: list[str],
    orig_by_base: dict[str, dict[str, Any]],
    adv_by_base: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    n_adv = sum(len(adv_by_base[b]) for b in paired)
    return {
        "prediction_file": path.name,
        "evalset": evalset,
        "paired_base_questions": len(paired),
        "adversarial_variants": n_adv,
        "original_em_on_paired": 100.0 * sum(orig_by_base[b]["_em"] for b in paired) / len(paired),
        "original_f1_on_paired": 100.0 * sum(orig_by_base[b]["_f1"] for b in paired) / len(paired),
        "adversarial_variant_em": 100.0 * sum(r["_em"] for b in paired for r in adv_by_base[b]) / n_adv,
        "adversarial_variant_f1": 100.0 * sum(r["_f1"] for b in paired for r in adv_by_base[b]) / n_adv,
        "all_adv_variants_correct_rate": 100.0 * sum(all(r["_em"] == 1.0 for r in adv_by_base[b]) for b in paired) / len(paired),
        "any_adv_variant_correct_rate": 100.0 * sum(any(r["_em"] == 1.0 for r in adv_by_base[b]) for b in paired) / len(paired),
        "correct_original_but_any_adv_fail_rate": 100.0 * sum(
            orig_by_base[b]["_em"] == 1.0 and any(r["_em"] == 0.0 for r in adv_by_base[b])
            for b in paired
        ) / len(paired),
    }


def audit_prediction_splits(results_dir: Path, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    split_rows = []
    paired_rows = []
    for path in sorted((results_dir / "predictions").glob("*.jsonl")):
        evalset = path.stem.rsplit("__", 1)[-1]
        rows = load_jsonl(path)
        if not rows:
            warnings.append(f"Prediction file {path.name} is empty.")
            continue
        groups, orig_by_base, adv_by_base = _split_prediction_rows(rows)
        for split in ["all_rows", "original_rows_only", "adversarial_rows_only"]:
            vals = groups.get(split, [])
            if vals:
                split_rows.append(_split_metric_row(path, evalset, split, vals))
        if evalset in {"addsent", "addonesent"}:
            if groups["original_rows_only"] and groups["adversarial_rows_only"]:
                warnings.append(
                    f"{path.name}: mixed original ({len(groups['original_rows_only'])}) and "
                    f"adversarial ({len(groups['adversarial_rows_only'])}) rows."
                )
            paired = sorted(set(orig_by_base) & set(adv_by_base))
            if paired:
                paired_rows.append(_paired_metric_row(path, evalset, paired, orig_by_base, adv_by_base))
    pd.DataFrame(split_rows).to_csv(out_dir / "eval_split_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out_dir / "paired_robustness_metrics.csv", index=False)
    return warnings


def _target_region(selection: str) -> str | None:
    if selection.startswith("easy"):
        return "Easy-to-learn"
    if selection.startswith("ambiguous"):
        return "Ambiguous"
    if selection.startswith("hard"):
        return "Hard-to-learn"
    return None


def audit_subset_purity(results_dir: Path, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    path = results_dir / "cartography" / "subset_assignments.csv"
    if not path.exists():
        warnings.append("Missing cartography/subset_assignments.csv")
        return warnings
    df = pd.read_csv(path)
    if "subset_fraction" not in df or "region" not in df:
        warnings.append("subset_assignments.csv must contain subset_fraction and region columns.")
        return warnings
    flags = [c for c in df.columns if c.startswith("selected_")]
    rows = []
    for frac, group in df.groupby("subset_fraction"):
        for flag in flags:
            selected = group[group[flag].astype(bool)]
            selection = flag.removeprefix("selected_")
            target_region = _target_region(selection)
            target_n = int((selected["region"] == target_region).sum()) if target_region else None
            rows.append(
                {
                    "subset_fraction": frac,
                    "selection": selection,
                    "n": int(len(selected)),
                    "target_region": target_region,
                    "target_region_n": target_n,
                    "purity": (target_n / len(selected)) if target_region and len(selected) else None,
                    "mean_confidence": selected["confidence"].mean() if "confidence" in selected else None,
                    "mean_variability": selected["variability"].mean() if "variability" in selected else None,
                    "mean_correctness": selected["correctness"].mean() if "correctness" in selected else None,
                }
            )
        for left_i, left in enumerate(flags):
            for right in flags[left_i + 1 :]:
                overlap = int((group[left].astype(bool) & group[right].astype(bool)).sum())
                if overlap:
                    warnings.append(f"Subset overlap at fraction {frac}: {left} x {right} = {overlap}")
                rows.append(
                    {
                        "subset_fraction": frac,
                        "selection": f"overlap_{left.removeprefix('selected_')}__{right.removeprefix('selected_')}",
                        "n": overlap,
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "subset_purity_overlap.csv", index=False)
    return warnings


def audit_windowing(results_dir: Path, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    scores_path = results_dir / "cartography" / "cartography_scores.csv"
    if not scores_path.exists():
        warnings.append("Missing cartography/cartography_scores.csv")
        return warnings
    scores = pd.read_csv(scores_path)
    rows = []
    if "n_records" in scores:
        for n_records, count in scores["n_records"].value_counts().sort_index().items():
            subset = scores[scores["n_records"] == n_records]
            rows.append(
                {
                    "n_records": n_records,
                    "n_examples": int(count),
                    "mean_confidence": subset["confidence"].mean() if "confidence" in subset else None,
                    "mean_variability": subset["variability"].mean() if "variability" in subset else None,
                    "mean_non_gold_records": subset["n_non_gold_records"].mean()
                    if "n_non_gold_records" in subset
                    else None,
                }
            )
        if (scores["n_records"] > 3).any():
            warnings.append("Some examples have n_records > 3; report QA overflow-window handling.")
    if "n_non_gold_records" not in scores.columns:
        warnings.append(
            "cartography_scores.csv lacks n_non_gold_records; rerun dynamics with gold_span_feature logging for full audit."
        )
    pd.DataFrame(rows).to_csv(out_dir / "windowing_audit.csv", index=False)
    return warnings


def write_table_audit(results_dir: Path, out_dir: Path) -> list[str]:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "completed_raw_metrics.csv"
    splits = out_dir / "eval_split_metrics.csv"
    lines = ["# Table audit", "", "Generated by `scripts/audit_current_results.py`.", ""]
    if raw.exists():
        df = pd.read_csv(raw)
        lines.extend(["## Raw metric sources", ""])
        if df.empty:
            lines.append("No raw metrics found.")
        else:
            lines.append("| train_run_id | evalset | n | EM | F1 | dataset_path | dataset_hash |")
            lines.append("|---|---:|---:|---:|---:|---|---|")
            for _, row in df.sort_values(["train_run_id", "evalset"]).iterrows():
                lines.append(
                    f"| {row.get('train_run_id', '')} | {row.get('evalset', '')} | "
                    f"{row.get('num_eval_examples', '')} | {row.get('exact_match', '')} | "
                    f"{row.get('f1', '')} | {row.get('dataset_path', '')} | {row.get('dataset_hash', '')} |"
                )
    if splits.exists():
        df = pd.read_csv(splits)
        lines.extend(["", "## Prediction-file split metrics", ""])
        if not df.empty:
            lines.append("| prediction_file | evalset | split | n | EM | F1 |")
            lines.append("|---|---:|---|---:|---:|---:|")
            for _, row in df.sort_values(["prediction_file", "split"]).iterrows():
                lines.append(
                    f"| {row['prediction_file']} | {row['evalset']} | {row['split']} | "
                    f"{row['n']} | {row['exact_match']:.4f} | {row['f1']:.4f} |"
                )
    text = "\n".join(lines) + "\n"
    (logs_dir / "table_audit.md").write_text(text, encoding="utf-8")
    (out_dir / "table_audit.md").write_text(text, encoding="utf-8")
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    warnings.extend(audit_raw_metrics(results_dir, out_dir))
    warnings.extend(audit_prediction_splits(results_dir, out_dir))
    warnings.extend(audit_subset_purity(results_dir, out_dir))
    warnings.extend(audit_windowing(results_dir, out_dir))
    warnings.extend(write_table_audit(results_dir, out_dir))

    for rel in ["metrics/main_table.csv", "metrics/seed_level_metrics.csv"]:
        if not (results_dir / rel).exists():
            warnings.append(f"Missing expected aggregate file: {rel}")
    with (out_dir / "warnings.txt").open("w", encoding="utf-8") as f:
        for warning in warnings:
            f.write(warning + "\n")
    print(f"Wrote audit outputs to {out_dir}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
