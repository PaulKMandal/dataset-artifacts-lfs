#!/usr/bin/env python3
"""Repair AddSent/AddOneSent adversarial-only and paired robustness metrics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import string
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable



def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", action="append", required=True, help="Panel result directory; repeat as needed.")
    ap.add_argument("--dataset-dir", default="data/qa")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--regenerate-missing", action="store_true", help="Run eval-only regeneration from checkpoints when predictions are absent.")
    ap.add_argument("--no-copy-predictions", action="store_true")
    ap.add_argument("--fp16", action="store_true", default=True)
    return ap




ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ADV_MARKER = "-high-conf-"

EVALSETS = {
    "addsent": "addsent.jsonl",
    "addonesent": "addonesent.jsonl",
}

EXPECTED_ROW_COUNTS = {
    "addsent": {"all": 3560, "original": 1000, "adversarial": 2560},
    "addonesent": {"all": 1787, "original": 1000, "adversarial": 787},
}

CONDITION_LABELS = {
    "full": "full data",
    "random": "random 33%",
    "easy": "easy-ranked 33%",
    "ambiguous": "ambiguous-ranked 33%",
    "hard": "hard-ranked 33%",
}


SPLIT_COUNT_FIELDS = [
    "evalset", "dataset_path", "all_rows", "original_rows", "adversarial_rows",
    "n_base_ids_with_original", "n_base_ids_with_adversarial", "n_pairable_base_ids",
    "split_method", "status",
]

PREDICTION_AUDIT_FIELDS = [
    "run_id", "evalset", "dataset_path", "predictions_path", "n_dataset_rows",
    "n_prediction_rows", "n_joined_rows", "n_missing_predictions", "n_extra_predictions", "status",
]

SEED_METRIC_FIELDS = [
    "panel", "model", "condition", "subset_source", "train_seed", "random_draw_id",
    "evalset", "split_type", "n", "exact_match", "f1", "dataset_path",
    "predictions_path", "run_id",
]

MAIN_TABLE_FIELDS = [
    "panel", "model", "condition", "subset_source", "evalset", "split_type",
    "n_runs", "mean_exact_match", "std_exact_match", "mean_f1", "std_f1", "total_rows_scored",
]

PAIRED_FIELDS = [
    "panel", "evalset", "model", "condition", "subset_source", "train_seed",
    "random_draw_id", "run_id", "dataset_path", "predictions_path", "n_pairable_base_ids",
    "n_original_correct_base_ids", "adversarial_em_over_paired_adv_rows",
    "adversarial_f1_over_paired_adv_rows", "any_adv_failure_given_original_correct",
    "all_adv_correct_given_original_correct", "mean_adv_em_given_original_correct",
    "mean_adv_f1_given_original_correct",
]

DELTA_FIELDS = [
    "panel", "model", "evalset", "split_type", "condition", "subset_source", "train_seed",
    "cartographic_exact_match", "mean_random_exact_match_same_seed", "delta_exact_match",
    "cartographic_f1", "mean_random_f1_same_seed", "delta_f1", "n_random_draws_same_seed",
    "mean_delta_exact_match_over_seeds", "std_delta_exact_match_over_seeds",
    "mean_delta_f1_over_seeds", "std_delta_f1_over_seeds", "n_training_seeds",
]

DRAW_FIELDS = [
    "panel", "model", "evalset", "split_type", "random_draw_id", "n_training_seeds",
    "mean_exact_match", "mean_f1",
]

WIN_FIELDS = [
    "panel", "model", "evalset", "split_type", "condition", "n_training_seeds",
    "n_seeds_delta_f1_positive", "n_seeds_delta_exact_match_positive", "mean_delta_f1",
    "mean_delta_exact_match",
]

MISSING_FIELDS = ["panel", "run_id", "evalset", "issue", "path", "detail"]
MANIFEST_FIELDS = [
    "panel", "run_id", "evalset", "source_predictions_path", "packaged_predictions_path", "n_rows", "sha256",
]




def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = s.translate(PUNCT_TABLE)
    s = ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match_score(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_scores(prediction: str, golds: list[str]) -> tuple[int, float]:
    if not golds:
        golds = [""]
    return (
        max(exact_match_score(prediction, g) for g in golds),
        max(f1_score(prediction, g) for g in golds),
    )


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def answer_texts(row: dict[str, Any]) -> list[str]:
    answers = row.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        return [str(x) for x in texts]
    if isinstance(answers, list):
        out = []
        for x in answers:
            out.append(str(x.get("text", "")) if isinstance(x, dict) else str(x))
        return [x for x in out if x]
    for key in ("gold_answers", "gold", "answer", "answer_text"):
        val = row.get(key)
        if isinstance(val, list):
            return [str(x) for x in val]
        if val is not None:
            return [str(val)]
    return []


def split_type_from_id(row_id: str) -> str:
    return "adversarial" if ADV_MARKER in str(row_id) else "original"


def base_id_from_id(row_id: str) -> str:
    return str(row_id).split(ADV_MARKER, 1)[0]


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def sanitize_example(row: dict[str, Any], prediction: str | None = None) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "base_id": base_id_from_id(str(row.get("id", ""))),
        "split": split_type_from_id(str(row.get("id", ""))),
        "title": row.get("title"),
        "question": str(row.get("question", ""))[:200],
        "context_prefix": str(row.get("context", ""))[:220],
        "answers": answer_texts(row)[:3],
        "predicted_answer": prediction,
    }



@dataclass(frozen=True)
class RunInfo:
    panel: str
    result_dir: Path
    run_id: str
    run_dir: Path
    model: str
    condition: str
    subset_source: str
    train_seed: int | None
    random_draw_id: str


def parse_run_id(panel: str, result_dir: Path, run_dir: Path) -> RunInfo:
    rid = run_dir.name
    m = re.match(r"(?P<model>.+?)__(?P<condition>full|random|easy|ambiguous|hard)__", rid)
    if not m:
        model = rid.split("__", 1)[0]
        condition = "unknown"
    else:
        model = m.group("model")
        condition = m.group("condition")
    seed_m = re.search(r"__seed(?P<seed>\d+)__", rid)
    draw_m = re.search(r"__draw(?P<draw>\d+)__", rid)
    return RunInfo(
        panel=panel,
        result_dir=result_dir,
        run_id=rid,
        run_dir=run_dir,
        model=model,
        condition=condition,
        subset_source=condition_label(condition),
        train_seed=int(seed_m.group("seed")) if seed_m else None,
        random_draw_id=f"draw{int(draw_m.group('draw')):02d}" if draw_m else "",
    )


def iter_run_infos(result_dir: Path) -> Iterable[RunInfo]:
    runs_dir = result_dir / "runs"
    if not runs_dir.exists():
        return []
    return [parse_run_id(result_dir.name, result_dir, p) for p in sorted(runs_dir.iterdir()) if p.is_dir()]



def find_prediction_path(result_dir: Path, run_id: str, evalset: str) -> Path | None:
    candidates = [
        result_dir / "predictions" / f"{run_id}__{evalset}.jsonl",
        result_dir / "predictions" / f"{run_id}__{evalset}.jsonl.gz",
        result_dir / "evals" / f"{run_id}__{evalset}" / "eval_predictions.jsonl",
        result_dir / "evals" / f"{run_id}__{evalset}" / "eval_predictions.jsonl.gz",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def checkpoint_available(run_dir: Path) -> bool:
    if not run_dir.exists() or not (run_dir / "config.json").exists():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (run_dir / name).exists() and (run_dir / name).stat().st_size > 1024:
            return True
    return bool(list(run_dir.glob("*.safetensors")) or list(run_dir.glob("pytorch_model*.bin")))


def regenerate_prediction(run_info: RunInfo, evalset: str, dataset_path: Path, fp16: bool, command_log: Path) -> Path | None:
    out_dir = run_info.result_dir / "evals" / f"{run_info.run_id}__{evalset}"
    cmd = [
        sys.executable, "run.py", "--do_eval", "--task", "qa", "--dataset", str(dataset_path),
        "--model", str(run_info.run_dir), "--output_dir", str(out_dir), "--overwrite_output_dir",
        "--max_length", "384", "--per_device_eval_batch_size", "64", "--seed", str(run_info.train_seed or 42),
        "--report_to", "none",
    ]
    if fp16:
        cmd.append("--fp16")
    with command_log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)
    pred = out_dir / "eval_predictions.jsonl"
    return pred if pred.exists() and pred.stat().st_size > 0 else None



def load_dataset_rows(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(dataset_path)
    by_id = {str(r.get("id")): r for r in rows}
    return rows, by_id


def split_audit_for_dataset(evalset: str, dataset_path: Path, dataset_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    original = [r for r in dataset_rows if split_type_from_id(str(r.get("id"))) == "original"]
    adv = [r for r in dataset_rows if split_type_from_id(str(r.get("id"))) == "adversarial"]
    orig_base_ids = {base_id_from_id(str(r.get("id"))) for r in original}
    adv_base_ids = {base_id_from_id(str(r.get("id"))) for r in adv}
    expected = EXPECTED_ROW_COUNTS.get(evalset, {})
    status = "ok"
    if expected and (
        len(dataset_rows) != expected["all"] or len(original) != expected["original"] or len(adv) != expected["adversarial"]
    ):
        status = "unexpected_counts"
    duplicate_key_count = len({
        (base_id_from_id(str(r.get("id"))), r.get("question"), r.get("context"), json.dumps(r.get("answers"), sort_keys=True))
        for r in original
    })
    row = {
        "evalset": evalset,
        "dataset_path": str(dataset_path),
        "all_rows": len(dataset_rows),
        "original_rows": len(original),
        "adversarial_rows": len(adv),
        "n_base_ids_with_original": len(orig_base_ids),
        "n_base_ids_with_adversarial": len(adv_base_ids),
        "n_pairable_base_ids": len(orig_base_ids & adv_base_ids),
        "split_method": "id contains '-high-conf-' => adversarial; otherwise original; base_id is id prefix before '-high-conf-'",
        "status": status,
    }
    details = {
        "original_examples": original[:2],
        "adversarial_examples": adv[:2],
        "duplicate_original_rows": len(original) - duplicate_key_count,
    }
    return row, details


def load_eval_datasets(dataset_dir: Path, logs_dir: Path) -> tuple[dict[str, tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    datasets: dict[str, tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    split_rows: list[dict[str, Any]] = []
    split_details: dict[str, dict[str, Any]] = {}
    hash_lines: list[str] = []
    for evalset, filename in EVALSETS.items():
        path = dataset_dir / filename
        if not path.exists():
            raise SystemExit(f"Missing dataset file required for split metrics: {path}")
        rows, by_id = load_dataset_rows(path)
        datasets[evalset] = (path, rows, by_id)
        audit_row, details = split_audit_for_dataset(evalset, path, rows)
        split_rows.append(audit_row)
        split_details[evalset] = details
        hash_lines.append(f"{evalset}	{path}	{sha256_file(path)}\n")
    (logs_dir / "dataset_hashes.txt").write_text("".join(hash_lines), encoding="utf-8")
    return datasets, split_rows, split_details



def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit("repair implementation is incomplete; apply the remaining commits")


if __name__ == "__main__":
    main()
