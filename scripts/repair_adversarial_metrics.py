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



def score_predictions(
    run_info: RunInfo,
    evalset: str,
    dataset_path: Path,
    dataset_rows: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
    pred_path: Path,
    out_pred_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    pred_rows = read_jsonl(pred_path)
    preds: dict[str, str] = {}
    for row in pred_rows:
        rid = str(row.get("id"))
        pred = row.get("predicted_answer", row.get("prediction_text", row.get("prediction", "")))
        preds[rid] = str(pred)

    joined: list[dict[str, Any]] = []
    missing: list[str] = []
    for drow in dataset_rows:
        rid = str(drow.get("id"))
        if rid not in preds:
            missing.append(rid)
            continue
        pred = preds[rid]
        em, f1 = squad_scores(pred, answer_texts(drow))
        split_raw = split_type_from_id(rid)
        scored = dict(drow)
        scored.update({
            "predicted_answer": pred,
            "exact_match": em,
            "f1": f1,
            "split_type": "adversarial_only" if split_raw == "adversarial" else "original_only",
            "base_id": base_id_from_id(rid),
        })
        joined.append(scored)

    extra = sorted(set(preds) - set(dataset_by_id))
    prediction_audit = {
        "run_id": run_info.run_id,
        "evalset": evalset,
        "dataset_path": str(dataset_path),
        "predictions_path": str(out_pred_path),
        "n_dataset_rows": len(dataset_rows),
        "n_prediction_rows": len(preds),
        "n_joined_rows": len(joined),
        "n_missing_predictions": len(missing),
        "n_extra_predictions": len(extra),
        "status": "ok" if not missing and not extra else "join_mismatch",
    }

    metric_rows = []
    for split_name, filt in [
        ("all_row_legacy", lambda r: True),
        ("original_only", lambda r: r["split_type"] == "original_only"),
        ("adversarial_only", lambda r: r["split_type"] == "adversarial_only"),
    ]:
        part = [r for r in joined if filt(r)]
        metric_rows.append({
            "panel": run_info.panel,
            "model": run_info.model,
            "condition": condition_label(run_info.condition),
            "subset_source": run_info.subset_source,
            "train_seed": run_info.train_seed,
            "random_draw_id": run_info.random_draw_id,
            "evalset": evalset,
            "split_type": split_name,
            "n": len(part),
            "exact_match": 100 * mean([r["exact_match"] for r in part]),
            "f1": 100 * mean([r["f1"] for r in part]),
            "dataset_path": str(dataset_path),
            "predictions_path": str(out_pred_path),
            "run_id": run_info.run_id,
        })

    return metric_rows, prediction_audit, joined, paired_metrics(run_info, evalset, dataset_path, out_pred_path, joined)



def paired_metrics(
    run_info: RunInfo,
    evalset: str,
    dataset_path: Path,
    pred_path: Path,
    scored_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored_rows:
        grouped[str(r["base_id"])].append(r)

    pairable: list[dict[str, Any]] = []
    for base_id, rows in grouped.items():
        orig = [r for r in rows if r["split_type"] == "original_only"]
        adv = [r for r in rows if r["split_type"] == "adversarial_only"]
        if not orig or not adv:
            continue
        unique_orig = {}
        for r in orig:
            key = (base_id, r.get("question"), r.get("context"), json.dumps(r.get("answers"), sort_keys=True))
            unique_orig[key] = r
        orig_em = max(float(r["exact_match"]) for r in unique_orig.values())
        pairable.append({
            "base_id": base_id,
            "original_correct": int(orig_em == 1.0),
            "adv_ems": [float(r["exact_match"]) for r in adv],
            "adv_f1s": [float(r["f1"]) for r in adv],
        })

    orig_correct = [p for p in pairable if p["original_correct"]]
    adv_rows_pairable = [em for p in pairable for em in p["adv_ems"]]
    adv_f1_pairable = [f1 for p in pairable for f1 in p["adv_f1s"]]
    return {
        "panel": run_info.panel,
        "evalset": evalset,
        "model": run_info.model,
        "condition": condition_label(run_info.condition),
        "subset_source": run_info.subset_source,
        "train_seed": run_info.train_seed,
        "random_draw_id": run_info.random_draw_id,
        "run_id": run_info.run_id,
        "dataset_path": str(dataset_path),
        "predictions_path": str(pred_path),
        "n_pairable_base_ids": len(pairable),
        "n_original_correct_base_ids": len(orig_correct),
        "adversarial_em_over_paired_adv_rows": 100 * mean(adv_rows_pairable),
        "adversarial_f1_over_paired_adv_rows": 100 * mean(adv_f1_pairable),
        "any_adv_failure_given_original_correct": mean([int(any(em == 0 for em in p["adv_ems"])) for p in orig_correct]),
        "all_adv_correct_given_original_correct": mean([int(all(em == 1 for em in p["adv_ems"])) for p in orig_correct]),
        "mean_adv_em_given_original_correct": 100 * mean([mean(p["adv_ems"]) for p in orig_correct]),
        "mean_adv_f1_given_original_correct": 100 * mean([mean(p["adv_f1s"]) for p in orig_correct]),
    }



def aggregate_main(seed_rows: list[dict[str, Any]], split_type: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        if r["split_type"] != split_type:
            continue
        key = (r["panel"], r["model"], r["condition"], r["subset_source"], r["evalset"], r["split_type"])
        groups[key].append(r)
    out = []
    for (panel, model, condition, subset_source, evalset, split), rows in sorted(groups.items()):
        out.append({
            "panel": panel,
            "model": model,
            "condition": condition,
            "subset_source": subset_source,
            "evalset": evalset,
            "split_type": split,
            "n_runs": len(rows),
            "mean_exact_match": mean([r["exact_match"] for r in rows]),
            "std_exact_match": stdev([r["exact_match"] for r in rows]),
            "mean_f1": mean([r["f1"] for r in rows]),
            "std_f1": stdev([r["f1"] for r in rows]),
            "total_rows_scored": sum(int(r["n"]) for r in rows),
        })
    return out


def random_draw_distribution(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        if r["condition"] == "random 33%" and r["random_draw_id"]:
            groups[(r["panel"], r["model"], r["evalset"], r["split_type"], r["random_draw_id"])].append(r)
    out = []
    for (panel, model, evalset, split_type, draw), rows in sorted(groups.items()):
        out.append({
            "panel": panel,
            "model": model,
            "evalset": evalset,
            "split_type": split_type,
            "random_draw_id": draw,
            "n_training_seeds": len({r["train_seed"] for r in rows}),
            "mean_exact_match": mean([r["exact_match"] for r in rows]),
            "mean_f1": mean([r["f1"] for r in rows]),
        })
    return out



def seed_matched_deltas(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for r in seed_rows:
        base_key = (r["panel"], r["model"], r["evalset"], r["split_type"], r["train_seed"])
        if r["condition"] == "random 33%":
            random_groups[base_key].append(r)
        elif r["condition"] in {"easy-ranked 33%", "ambiguous-ranked 33%", "hard-ranked 33%"}:
            selected.append(r)

    per_seed = []
    for r in selected:
        key = (r["panel"], r["model"], r["evalset"], r["split_type"], r["train_seed"])
        rand = random_groups.get(key, [])
        if not rand:
            continue
        rand_em = mean([x["exact_match"] for x in rand])
        rand_f1 = mean([x["f1"] for x in rand])
        per_seed.append({
            "panel": r["panel"],
            "model": r["model"],
            "evalset": r["evalset"],
            "split_type": r["split_type"],
            "condition": r["condition"],
            "subset_source": r["subset_source"],
            "train_seed": r["train_seed"],
            "cartographic_exact_match": r["exact_match"],
            "mean_random_exact_match_same_seed": rand_em,
            "delta_exact_match": r["exact_match"] - rand_em,
            "cartographic_f1": r["f1"],
            "mean_random_f1_same_seed": rand_f1,
            "delta_f1": r["f1"] - rand_f1,
            "n_random_draws_same_seed": len(rand),
        })

    summary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in per_seed:
        key = (r["panel"], r["model"], r["evalset"], r["split_type"], r["condition"], r["subset_source"])
        summary_groups[key].append(r)
    stats = {
        key: {
            "mean_delta_exact_match_over_seeds": mean([r["delta_exact_match"] for r in rows]),
            "std_delta_exact_match_over_seeds": stdev([r["delta_exact_match"] for r in rows]),
            "mean_delta_f1_over_seeds": mean([r["delta_f1"] for r in rows]),
            "std_delta_f1_over_seeds": stdev([r["delta_f1"] for r in rows]),
            "n_training_seeds": len({r["train_seed"] for r in rows}),
        }
        for key, rows in summary_groups.items()
    }
    for r in per_seed:
        key = (r["panel"], r["model"], r["evalset"], r["split_type"], r["condition"], r["subset_source"])
        r.update(stats[key])
    return sorted(per_seed, key=lambda r: (r["panel"], r["model"], r["evalset"], r["split_type"], r["condition"], r["train_seed"]))


def selected_win_counts(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in seed_matched_deltas(seed_rows):
        groups[(r["panel"], r["model"], r["evalset"], r["split_type"], r["condition"])].append(r)
    out = []
    for (panel, model, evalset, split_type, condition), rows in sorted(groups.items()):
        out.append({
            "panel": panel,
            "model": model,
            "evalset": evalset,
            "split_type": split_type,
            "condition": condition,
            "n_training_seeds": len(rows),
            "n_seeds_delta_f1_positive": sum(1 for r in rows if r["delta_f1"] > 0),
            "n_seeds_delta_exact_match_positive": sum(1 for r in rows if r["delta_exact_match"] > 0),
            "mean_delta_f1": mean([r["delta_f1"] for r in rows]),
            "mean_delta_exact_match": mean([r["delta_exact_match"] for r in rows]),
        })
    return out



def schema_audit_markdown(split_details: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Split schema audit", "",
        "Classification method: rows whose `id` contains `-high-conf-` are classified as adversarial; all other rows are classified as original. `base_id` is the `id` prefix before `-high-conf-` for adversarial rows and the full `id` for original rows.",
        "", "Fields used:",
        "- row ID: `id`",
        "- base/original ID: prefix of `id` before `-high-conf-`",
        "- gold answer: `answers.text`",
        "- predicted answer: `predicted_answer` from row-level prediction JSONL",
        "- split classification: `id` contains `-high-conf-`", "",
    ]
    for evalset, details in split_details.items():
        lines.extend([f"## {evalset}", "", f"Duplicate original rows after `(base_id, question, context, answers)` deduplication: {details.get('duplicate_original_rows', 0)}"])
        for split_name, examples in [("original", details.get("original_examples", [])), ("adversarial", details.get("adversarial_examples", []))]:
            lines.extend(["", f"### {split_name} examples"])
            for i, row in enumerate(examples[:2], start=1):
                lines.extend(["", f"Example {i}:", "```json", json.dumps(sanitize_example(row), indent=2, ensure_ascii=False), "```"])
        lines.append("")
    return "\n".join(lines)


def environment_text() -> str:
    lines = [f"created_utc={datetime.now(timezone.utc).isoformat()}", f"python={sys.version}", f"cwd={Path.cwd()}"]
    try:
        lines.append("git_commit=" + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        lines.append("git_branch=" + subprocess.check_output(["git", "branch", "--show-current"], text=True).strip())
    except Exception as e:
        lines.append(f"git_unavailable={e}")
    for key in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HF_HOME", "UV_CACHE_DIR"]:
        lines.append(f"{key}={os.environ.get(key, '')}")
    return "\n".join(lines) + "\n"


def make_summary(seed_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]], split_audit_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Adversarial repair summary", "",
        "This repair computes split AddSent/AddOneSent metrics and paired robustness metrics from row-level prediction JSONL files. All-row AddSent/AddOneSent metrics are retained only as legacy/comparability metrics.", "",
    ]
    if missing_rows:
        lines.extend(["## Blockers / missing runs", "", f"Missing or failed run/eval entries: {len(missing_rows)}. See `metrics/missing_or_failed_runs.csv`.", ""])
    else:
        lines.append("No missing prediction/checkpoint blockers were detected for processed AddSent/AddOneSent evals.\n")
    lines.extend(["## Split audit", ""])
    for r in split_audit_rows:
        lines.append(f"- {r['evalset']}: all={r['all_rows']}, original={r['original_rows']}, adversarial={r['adversarial_rows']}, pairable_base_ids={r['n_pairable_base_ids']}, status={r['status']}")
    lines.extend(["", "## Ambiguous-ranked 33% vs random 33% on adversarial-only F1", ""])
    random_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    amb_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in seed_rows:
        if r["split_type"] != "adversarial_only" or r["evalset"] not in {"addsent", "addonesent"}:
            continue
        key = (r["model"], r["evalset"], r["split_type"])
        if r["condition"] == "random 33%":
            random_groups[key].append(r["f1"])
        elif r["condition"] == "ambiguous-ranked 33%":
            amb_groups[key].append(r["f1"])
    for key in sorted(set(random_groups) | set(amb_groups)):
        model, evalset, _split = key
        rmean = mean(random_groups.get(key, []))
        amean = mean(amb_groups.get(key, []))
        if math.isnan(rmean) or math.isnan(amean):
            verdict = "insufficient data"
        elif amean > rmean:
            verdict = "ambiguous-ranked 33% is above the random mean"
        elif amean < rmean:
            verdict = "ambiguous-ranked 33% is below the random mean"
        else:
            verdict = "ambiguous-ranked 33% equals the random mean"
        lines.append(f"- {model} / {evalset}: ambiguous F1={amean:.3f}, random mean F1={rmean:.3f}; {verdict}.")
    lines.extend(["", "## Paired robustness", ""])
    lines.append("Paired robustness rows were computed for pairable original/adversarial base IDs. See `metrics/paired_robustness_table.csv`." if paired_rows else "No paired robustness rows were computed.")
    lines.extend(["", "## Interpretation constraints", "", "- Do not use all-row AddSent/AddOneSent metrics as primary robustness metrics.", "- Do not treat random draw × training seed as IID; draw-level random means are preserved.", "- Use easy-ranked 33%, ambiguous-ranked 33%, hard-ranked 33%, random 33%, and full data terminology."])
    return "\n".join(lines) + "\n"



@dataclass
class RepairState:
    out_root: Path
    metrics_dir: Path
    predictions_dir: Path
    logs_dir: Path
    command_log: Path
    seed_rows: list[dict[str, Any]]
    prediction_audit_rows: list[dict[str, Any]]
    paired_rows: list[dict[str, Any]]
    missing_rows: list[dict[str, Any]]
    prediction_manifest_rows: list[dict[str, Any]]


def initialize_output(args: argparse.Namespace) -> RepairState:
    out_root = Path(args.out_dir or f"cartography_adversarial_repair_{utc_stamp()}")
    metrics_dir = out_root / "metrics"
    predictions_dir = out_root / "predictions"
    logs_dir = out_root / "logs"
    for d in (metrics_dir, predictions_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    command_log = logs_dir / "command_log.txt"
    command_log.write_text("", encoding="utf-8")
    (logs_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
    except Exception:
        git_commit = "git unavailable\n"
    (logs_dir / "git_commit.txt").write_text(git_commit, encoding="utf-8")
    return RepairState(out_root, metrics_dir, predictions_dir, logs_dir, command_log, [], [], [], [], [])



def resolve_prediction_for_eval(
    args: argparse.Namespace,
    state: RepairState,
    run_info: RunInfo,
    evalset: str,
    dataset_path: Path,
) -> Path | None:
    pred_path = find_prediction_path(run_info.result_dir, run_info.run_id, evalset)
    if pred_path is not None:
        return pred_path
    if not args.regenerate_missing:
        state.missing_rows.append({"panel": run_info.panel, "run_id": run_info.run_id, "evalset": evalset, "issue": "missing_predictions", "path": str(run_info.result_dir), "detail": ""})
        return None
    if not checkpoint_available(run_info.run_dir):
        state.missing_rows.append({"panel": run_info.panel, "run_id": run_info.run_id, "evalset": evalset, "issue": "missing_predictions_and_checkpoint", "path": str(run_info.run_dir), "detail": ""})
        return None
    try:
        return regenerate_prediction(run_info, evalset, dataset_path, args.fp16, state.command_log)
    except subprocess.CalledProcessError as e:
        state.missing_rows.append({"panel": run_info.panel, "run_id": run_info.run_id, "evalset": evalset, "issue": "regenerate_eval_failed", "path": str(run_info.run_dir), "detail": str(e)})
        return None


def process_result_dirs(
    args: argparse.Namespace,
    state: RepairState,
    datasets_by_eval: dict[str, tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]],
) -> None:
    for result_dir_raw in args.results_dir:
        result_dir = Path(result_dir_raw)
        runs_dir = result_dir / "runs"
        if not runs_dir.exists():
            state.missing_rows.append({"panel": result_dir.name, "run_id": "", "evalset": "", "issue": "missing_runs_dir", "path": str(runs_dir), "detail": ""})
            continue
        for run_info in iter_run_infos(result_dir):
            for evalset, (dataset_path, dataset_rows, dataset_by_id) in datasets_by_eval.items():
                pred_path = resolve_prediction_for_eval(args, state, run_info, evalset, dataset_path)
                if pred_path is None:
                    continue
                out_pred_path = state.predictions_dir / f"{result_dir.name}__{run_info.run_id}__{evalset}.jsonl.gz"
                metric_rows, pred_audit, scored_rows, paired_row = score_predictions(
                    run_info, evalset, dataset_path, dataset_rows, dataset_by_id, pred_path, out_pred_path
                )
                state.seed_rows.extend(metric_rows)
                state.prediction_audit_rows.append(pred_audit)
                state.paired_rows.append(paired_row)
                if not args.no_copy_predictions:
                    write_jsonl_gz(out_pred_path, scored_rows)
                    state.prediction_manifest_rows.append({
                        "panel": run_info.panel,
                        "run_id": run_info.run_id,
                        "evalset": evalset,
                        "source_predictions_path": str(pred_path),
                        "packaged_predictions_path": str(out_pred_path),
                        "n_rows": len(scored_rows),
                        "sha256": sha256_file(out_pred_path),
                    })



def write_repair_tables(state: RepairState, split_audit_rows: list[dict[str, Any]], split_details: dict[str, dict[str, Any]]) -> None:
    write_csv(state.metrics_dir / "split_count_audit.csv", split_audit_rows, SPLIT_COUNT_FIELDS)
    (state.metrics_dir / "split_schema_audit.md").write_text(schema_audit_markdown(split_details), encoding="utf-8")
    write_csv(state.metrics_dir / "prediction_count_audit.csv", state.prediction_audit_rows, PREDICTION_AUDIT_FIELDS)
    write_csv(state.metrics_dir / "seed_level_metrics_by_split.csv", state.seed_rows, SEED_METRIC_FIELDS)
    write_csv(state.metrics_dir / "main_table_adversarial_only.csv", aggregate_main(state.seed_rows, "adversarial_only"), MAIN_TABLE_FIELDS)
    write_csv(state.metrics_dir / "main_table_original_only.csv", aggregate_main(state.seed_rows, "original_only"), MAIN_TABLE_FIELDS)
    write_csv(state.metrics_dir / "main_table_all_row_legacy.csv", aggregate_main(state.seed_rows, "all_row_legacy"), MAIN_TABLE_FIELDS)
    write_csv(state.metrics_dir / "paired_robustness_table.csv", state.paired_rows, PAIRED_FIELDS)
    write_csv(state.metrics_dir / "seed_matched_delta_by_split.csv", seed_matched_deltas(state.seed_rows), DELTA_FIELDS)
    write_csv(state.metrics_dir / "random_draw_distribution_by_split.csv", random_draw_distribution(state.seed_rows), DRAW_FIELDS)
    write_csv(state.metrics_dir / "selected_condition_win_counts_by_split.csv", selected_win_counts(state.seed_rows), WIN_FIELDS)
    write_csv(state.metrics_dir / "missing_or_failed_runs.csv", state.missing_rows, MISSING_FIELDS)
    write_csv(state.predictions_dir / "MANIFEST.csv", state.prediction_manifest_rows, MANIFEST_FIELDS)


def write_repair_readmes(args: argparse.Namespace, state: RepairState, split_audit_rows: list[dict[str, Any]]) -> None:
    readme = f"""# Adversarial repair package

Created UTC: {datetime.now(timezone.utc).isoformat()}

This package contains split AddSent/AddOneSent metrics and paired robustness metrics. All-row AddSent/AddOneSent metrics are retained only as legacy/comparability metrics.

Processed result directories:
{chr(10).join(f'- {x}' for x in args.results_dir)}

Prediction regeneration requested: {args.regenerate_missing}
Missing/failed entries: {len(state.missing_rows)}
"""
    state.out_root.joinpath("README.md").write_text(readme, encoding="utf-8")
    state.out_root.joinpath("REPAIR_SUMMARY.md").write_text(make_summary(state.seed_rows, state.paired_rows, state.missing_rows, split_audit_rows), encoding="utf-8")


def make_tarball(out_root: Path) -> Path:
    tar_path = out_root.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    shutil.make_archive(str(out_root), "gztar", root_dir=out_root.parent, base_dir=out_root.name)
    return tar_path



def main() -> None:
    args = build_parser().parse_args()
    state = initialize_output(args)
    datasets_by_eval, split_audit_rows, split_details = load_eval_datasets(Path(args.dataset_dir), state.logs_dir)
    process_result_dirs(args, state, datasets_by_eval)
    write_repair_tables(state, split_audit_rows, split_details)
    write_repair_readmes(args, state, split_audit_rows)
    tar_path = make_tarball(state.out_root)
    print(tar_path)
    if state.missing_rows:
        print(f"WARNING: {len(state.missing_rows)} missing/failed entries. See {state.metrics_dir / 'missing_or_failed_runs.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
