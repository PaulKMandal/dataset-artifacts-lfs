#!/usr/bin/env python3
"""Create persistent QA subset JSONL files from cartography scores.

Selection rules are intentionally simple and auditable:

* easy: highest confidence, then lowest variability, then highest correctness
* hard: lowest confidence, then lowest correctness, then highest variability
* ambiguous: highest variability, then middle confidence preference
* random: uniform sample without replacement, one persistent file per draw

The script writes a manifest plus a subset_assignments.csv that records exactly
which original SQuAD train indices were selected for every subset/fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from argparse import Namespace
from pathlib import Path
from typing import Iterable


CARTO_SUBSETS = ("easy", "ambiguous", "hard")

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Flat SQuAD train JSONL with stable idx fields.")
    parser.add_argument("--cartography-scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--assignments-out", default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.333])
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--random-seed-base", type=int, default=7300)
    parser.add_argument("--confidence-definition", default="joint_confidence")
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    return parser.parse_args()

def fraction_label(frac: float) -> str:
    text = f"{frac:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count

def read_scores(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        scores = {}
        for row in reader:
            idx = int(row["idx"])
            scores[idx] = {
                "idx": idx,
                "confidence": float(row.get("confidence", row.get("avg_confidence", 0.0))),
                "variability": float(row.get("variability", 0.0)),
                "correctness": float(row.get("correctness", 0.0)),
                "region": row.get("region", ""),
                "n_records": int(float(row.get("n_records", 0) or 0)),
            }
    if not scores:
        raise SystemExit(f"No cartography scores loaded from {path}")
    return scores

def answer_summary(example: dict) -> tuple[str, str, str]:
    answers = example.get("answers", {})
    texts = answers.get("text", []) if isinstance(answers, dict) else []
    starts = answers.get("answer_start", []) if isinstance(answers, dict) else []
    answer_text = texts[0] if texts else ""
    answer_start = starts[0] if starts else ""
    answer_length = len(answer_text.split()) if answer_text else 0
    return answer_text, str(answer_start), str(answer_length)

def question_type(question: str) -> str:
    stripped = question.strip().lower()
    return stripped.split(maxsplit=1)[0].rstrip(":?") if stripped else ""

def select_indices(scores: dict[int, dict], subset: str, k: int) -> list[int]:
    rows = list(scores.values())
    if subset == "easy":
        rows.sort(key=lambda r: (-r["confidence"], r["variability"], -r["correctness"], r["idx"]))
    elif subset == "hard":
        rows.sort(key=lambda r: (r["confidence"], r["correctness"], -r["variability"], r["idx"]))
    elif subset == "ambiguous":
        # Prefer high variability; tie-break toward mid confidence rather than
        # trivially easy or hard examples.
        rows.sort(key=lambda r: (-r["variability"], abs(r["confidence"] - 0.5), r["idx"]))
    else:
        raise ValueError(f"Unknown cartography subset: {subset}")
    return [int(r["idx"]) for r in rows[:k]]

def subset_size(n_scores: int, frac: float, rounding: str) -> int:
    if not (0 < frac <= 1):
        raise SystemExit(f"Invalid fraction {frac}; expected 0 < fraction <= 1")
    raw_k = n_scores * frac
    if rounding == "floor":
        k = int(math.floor(raw_k))
    elif rounding == "ceil":
        k = int(math.ceil(raw_k))
    else:
        k = int(round(raw_k))
    return max(1, min(k, n_scores))

def load_inputs(args: Namespace) -> tuple[list[dict], dict[int, dict], dict[int, dict]]:
    train_data = read_jsonl(Path(args.train_data))
    by_idx = {int(row.get("idx", i)): row for i, row in enumerate(train_data)}
    scores = read_scores(Path(args.cartography_scores))
    missing = sorted(set(scores) - set(by_idx))
    if missing:
        raise SystemExit(f"Cartography scores contain {len(missing)} idx values absent from train data")
    return train_data, by_idx, scores

def select_cartography_subsets(
    by_idx: dict[int, dict],
    scores: dict[int, dict],
    out_dir: Path,
    args: Namespace,
    *,
    frac: float,
    k: int,
) -> tuple[list[dict], dict[str, set[int]]]:
    flabel = fraction_label(frac)
    manifest_rows = []
    selected_by_name: dict[str, set[int]] = {}
    for subset in CARTO_SUBSETS:
        indices = select_indices(scores, subset, k)
        selected_by_name[subset] = set(indices)
        path = out_dir / f"{subset}_frac{flabel}.jsonl"
        count = write_jsonl((by_idx[idx] for idx in indices), path)
        manifest_rows.append(
            {
                "subset": subset,
                "subset_fraction": frac,
                "subset_size": count,
                "subset_draw_id": "",
                "path": str(path),
                "confidence_definition": args.confidence_definition,
                "selection_rule": subset,
            }
        )
    return manifest_rows, selected_by_name

def select_random_subsets(
    by_idx: dict[int, dict],
    all_indices: list[int],
    out_dir: Path,
    args: Namespace,
    *,
    frac: float,
    k: int,
) -> list[dict]:
    flabel = fraction_label(frac)
    manifest_rows = []
    for draw_id in range(args.random_draws):
        rng = random.Random(args.random_seed_base + draw_id)
        indices = sorted(rng.sample(all_indices, k))
        path = out_dir / f"random_frac{flabel}_draw{draw_id:02d}.jsonl"
        count = write_jsonl((by_idx[idx] for idx in indices), path)
        manifest_rows.append(
            {
                "subset": "random",
                "subset_fraction": frac,
                "subset_size": count,
                "subset_draw_id": draw_id,
                "path": str(path),
                "confidence_definition": args.confidence_definition,
                "selection_rule": "uniform_without_replacement",
            }
        )
    return manifest_rows

def assignment_row(
    idx: int,
    example: dict,
    score: dict,
    selected_by_name: dict[str, set[int]],
    args: Namespace,
    *,
    frac: float,
    rank: int,
) -> dict:
    answer_text, answer_start, answer_length = answer_summary(example)
    return {
        "idx": idx,
        "example_id": example.get("id", ""),
        "title": example.get("title", ""),
        "question": example.get("question", ""),
        "answer_text": answer_text,
        "answer_start": answer_start,
        "context_length": len(example.get("context", "").split()),
        "answer_length": answer_length,
        "question_type": question_type(example.get("question", "")),
        "confidence": score["confidence"],
        "variability": score["variability"],
        "correctness": score["correctness"],
        "region": score.get("region", ""),
        "selection_rank": rank,
        "subset_fraction": frac,
        "confidence_definition": args.confidence_definition,
        "selection_rule": "confidence_variability_rank",
        "selected_easy": idx in selected_by_name["easy"],
        "selected_ambiguous": idx in selected_by_name["ambiguous"],
        "selected_hard": idx in selected_by_name["hard"],
    }

def append_assignment_rows(
    rows: list[dict],
    by_idx: dict[int, dict],
    scores: dict[int, dict],
    selected_by_name: dict[str, set[int]],
    all_indices: list[int],
    args: Namespace,
    *,
    frac: float,
) -> None:
    for rank, idx in enumerate(all_indices):
        rows.append(
            assignment_row(
                idx,
                by_idx[idx],
                scores[idx],
                selected_by_name,
                args,
                frac=frac,
                rank=rank,
            )
        )

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise SystemExit(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def build_outputs(args: Namespace) -> tuple[Path, Path]:
    _, by_idx, scores = load_inputs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    assignment_rows: list[dict] = []
    all_indices = sorted(scores)

    for frac in args.fractions:
        k = subset_size(len(scores), frac, args.rounding)
        carto_rows, selected_by_name = select_cartography_subsets(by_idx, scores, out_dir, args, frac=frac, k=k)
        manifest_rows.extend(carto_rows)
        manifest_rows.extend(select_random_subsets(by_idx, all_indices, out_dir, args, frac=frac, k=k))
        append_assignment_rows(assignment_rows, by_idx, scores, selected_by_name, all_indices, args, frac=frac)

    manifest_path = out_dir / "subset_manifest.csv"
    assignments_path = Path(args.assignments_out) if args.assignments_out else out_dir / "subset_assignments.csv"
    write_csv(manifest_rows, manifest_path)
    write_csv(assignment_rows, assignments_path)
    return manifest_path, assignments_path

def main() -> None:
    manifest_path, assignments_path = build_outputs(parse_args())
    print(f"Wrote {manifest_path}")
    print(f"Wrote {assignments_path}")

if __name__ == "__main__":
    main()
