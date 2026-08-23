#!/usr/bin/env python3
"""Materialize SQuAD, adversarial QA, and selected MRQA evalsets as flat JSONL.

The experiment panel consumes flat SQuAD-style JSONL so every train/eval file has
an auditable path, checksum, and example count. By default this script downloads:

* SQuAD v1.1 train/validation via ``datasets.load_dataset("squad")``
* AddSent/AddOneSent via ``stanfordnlp/squad_adversarial``

You can also point it at local SQuAD-v1-style JSON files; this is useful when the
server is expected to use a frozen private data mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path

import datasets

DEFAULT_MRQA_CONFIGS = ("newsqa", "triviaqa", "searchqa", "natural_questions")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/qa")
    parser.add_argument("--squad-train-json", default=None)
    parser.add_argument("--squad-dev-json", default=None)
    parser.add_argument("--addsent-json", default=None)
    parser.add_argument("--addonesent-json", default=None)
    parser.add_argument("--adversarialqa-json", default=None)
    parser.add_argument(
        "--include-ood",
        action="store_true",
        help="Also materialize AdversarialQA and the configured MRQA validation sets.",
    )
    parser.add_argument(
        "--mrqa-configs",
        nargs="*",
        default=list(DEFAULT_MRQA_CONFIGS),
        help="MRQA configs to materialize when --include-ood is set.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True for stanfordnlp/squad_adversarial.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def normalize_record(record: dict, idx: int | None = None) -> dict:
    answers = record.get("answers", {})
    if isinstance(answers, list):
        answers = {
            "text": [a.get("text", "") for a in answers],
            "answer_start": [int(a.get("answer_start", 0)) for a in answers],
        }
    else:
        answers = {
            "text": list(answers.get("text", [])),
            "answer_start": [int(x) for x in answers.get("answer_start", [])],
        }

    out = {
        "id": str(record.get("id", idx if idx is not None else "")),
        "title": record.get("title", ""),
        "context": record["context"],
        "question": record["question"],
        "answers": answers,
    }
    if idx is not None:
        out["idx"] = int(idx)
    elif "idx" in record and record["idx"] is not None:
        out["idx"] = int(record["idx"])
    return out


def flatten_squad_json(path: Path, *, with_idx: bool = False) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    idx = 0
    for article in payload.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                record = {
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": qa.get("answers", []),
                }
                yield normalize_record(record, idx if with_idx else None)
                idx += 1


def dataset_records(dataset, split: str, *, with_idx: bool = False) -> Iterable[dict]:
    for idx, record in enumerate(dataset[split]):
        yield normalize_record(record, idx if with_idx else None)


def mrqa_records(dataset, split: str) -> Iterable[dict]:
    """Convert tau/mrqa rows to SQuAD-style records with auditable spans."""
    for row_number, record in enumerate(dataset[split]):
        context = record["context"]
        answer_texts = list(dict.fromkeys(record.get("answers", [])))
        texts = []
        starts = []
        for answer in answer_texts:
            start = context.find(answer)
            if start >= 0:
                texts.append(answer)
                starts.append(start)
        if not texts:
            raise ValueError(
                f"MRQA row {record.get('qid', row_number)!r} has no answer string in its context"
            )
        yield {
            "id": str(record.get("qid", row_number)),
            "title": str(record.get("subset", "")),
            "context": context,
            "question": record["question"],
            "answers": {"text": texts, "answer_start": starts},
        }


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def materialize_one(name: str, records: Iterable[dict], out_dir: Path) -> dict:
    path = out_dir / f"{name}.jsonl"
    count = write_jsonl(records, path)
    return {"name": name, "path": str(path), "num_examples": count, "sha256": sha256_file(path)}


def materialize_squad(args: Namespace, out_dir: Path) -> dict[str, dict]:
    if args.squad_train_json and args.squad_dev_json:
        return {
            "squad_train": materialize_one(
                "squad_train",
                flatten_squad_json(Path(args.squad_train_json), with_idx=True),
                out_dir,
            ),
            "squad_dev": materialize_one(
                "squad_dev", flatten_squad_json(Path(args.squad_dev_json), with_idx=False), out_dir
            ),
        }
    squad = datasets.load_dataset("squad")
    return {
        "squad_train": materialize_one(
            "squad_train", dataset_records(squad, "train", with_idx=True), out_dir
        ),
        "squad_dev": materialize_one(
            "squad_dev", dataset_records(squad, "validation", with_idx=False), out_dir
        ),
    }


def materialize_addsent(args: Namespace, out_dir: Path) -> dict:
    if args.addsent_json:
        return materialize_one(
            "addsent", flatten_squad_json(Path(args.addsent_json), with_idx=False), out_dir
        )
    addsent = datasets.load_dataset(
        "stanfordnlp/squad_adversarial", "AddSent", trust_remote_code=args.trust_remote_code
    )
    return materialize_one(
        "addsent", dataset_records(addsent, "validation", with_idx=False), out_dir
    )


def materialize_addonesent(args: Namespace, out_dir: Path) -> dict:
    if args.addonesent_json:
        return materialize_one(
            "addonesent", flatten_squad_json(Path(args.addonesent_json), with_idx=False), out_dir
        )
    addonesent = datasets.load_dataset(
        "stanfordnlp/squad_adversarial", "AddOneSent", trust_remote_code=args.trust_remote_code
    )
    return materialize_one(
        "addonesent", dataset_records(addonesent, "validation", with_idx=False), out_dir
    )


def materialize_adversarialqa(args: Namespace, out_dir: Path) -> dict:
    if args.adversarialqa_json:
        return materialize_one(
            "adversarialqa",
            flatten_squad_json(Path(args.adversarialqa_json), with_idx=False),
            out_dir,
        )
    dataset = datasets.load_dataset("UCLNLP/adversarial_qa")
    return materialize_one(
        "adversarialqa",
        dataset_records(dataset, "validation", with_idx=False),
        out_dir,
    )


def materialize_mrqa(config_name: str, args: Namespace, out_dir: Path) -> dict:
    dataset = datasets.load_dataset(
        "tau/mrqa",
        config_name,
        trust_remote_code=args.trust_remote_code,
    )
    name = f"mrqa_{config_name}"
    return materialize_one(name, mrqa_records(dataset, "validation"), out_dir)


def write_manifest(manifest: dict[str, dict], out_dir: Path) -> Path:
    manifest_path = out_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = materialize_squad(args, out_dir)
    manifest["addsent"] = materialize_addsent(args, out_dir)
    manifest["addonesent"] = materialize_addonesent(args, out_dir)
    if args.include_ood:
        manifest["adversarialqa"] = materialize_adversarialqa(args, out_dir)
        for config_name in args.mrqa_configs:
            name = f"mrqa_{config_name}"
            manifest[name] = materialize_mrqa(config_name, args, out_dir)
    manifest_path = write_manifest(manifest, out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
