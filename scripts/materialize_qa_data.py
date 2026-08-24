#!/usr/bin/env python3
"""Materialize SQuAD, adversarial QA, and selected MRQA evalsets as flat JSONL.

The experiment panel consumes flat SQuAD-style JSONL so every train/eval file has
an auditable path, checksum, and example count. By default this script downloads:

* SQuAD v1.1 train/validation via ``datasets.load_dataset("squad")``
* AddSent/AddOneSent from checksum-pinned official source files, preferring a
  persistent local copy and using the legacy official endpoint only as a
  bounded fallback

You can also point it at local SQuAD-v1-style JSON files; this is useful when the
server is expected to use a frozen private data mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path

import datasets

DEFAULT_MRQA_CONFIGS = ("newsqa", "triviaqa", "searchqa", "natural_questions")
ADVERSARIAL_SQUAD_SOURCES = {
    "addsent": {
        "url": "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/",
        "sha256": "sha256:40e3602aa5195cdacd03904a9c301ceb17ccf730cc32bd3ab998b66b4401e660",
        "num_bytes": 4_073_864,
        "num_examples": 3_560,
    },
    "addonesent": {
        "url": "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/",
        "sha256": "sha256:50420ac8d8b7547cd3715347c9a276802bc9466328ba0814adce4c20495e2889",
        "num_bytes": 1_920_649,
        "num_examples": 1_787,
    },
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/qa")
    parser.add_argument("--squad-train-json", default=None)
    parser.add_argument("--squad-dev-json", default=None)
    parser.add_argument("--addsent-json", default=None)
    parser.add_argument("--addonesent-json", default=None)
    parser.add_argument("--adversarialqa-json", default=None)
    parser.add_argument(
        "--adversarial-source-dir",
        default=None,
        help=(
            "Persistent directory for checksum-pinned AddSent/AddOneSent source JSON. "
            "Defaults to OUT_DIR/../sources/squad_adversarial."
        ),
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=6,
        help="Attempts for each checksum-pinned adversarial SQuAD download.",
    )
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


def read_jsonl(path: Path, *, with_idx: bool = False) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            yield normalize_record(json.loads(line), idx if with_idx else None)


def local_records(path: Path, *, with_idx: bool = False) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path, with_idx=with_idx)
    return flatten_squad_json(path, with_idx=with_idx)


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


def validate_record(record: dict, path: Path) -> None:
    context = record["context"]
    answers = record["answers"]
    texts = answers["text"]
    starts = answers["answer_start"]
    if len(texts) != len(starts):
        raise ValueError(f"Mismatched answer text/start lengths while writing {path}")
    if not texts:
        raise ValueError(f"Answerless record {record.get('id')!r} while writing {path}")
    for text, start in zip(texts, starts, strict=True):
        if start < 0 or context[start : start + len(text)] != text:
            raise ValueError(
                f"Invalid answer span for record {record.get('id')!r} while writing {path}"
            )


def write_jsonl(records: Iterable[dict], path: Path, *, expected_count: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids: set[str] = set()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            for record in records:
                validate_record(record, path)
                identifier = str(record["id"])
                if identifier in seen_ids:
                    raise ValueError(f"Duplicate record id {identifier!r} while writing {path}")
                seen_ids.add(identifier)
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())
        if expected_count is not None and count != expected_count:
            raise ValueError(f"{path.stem} contained {count} examples; expected {expected_count}")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return count


def materialize_one(
    name: str,
    records: Iterable[dict],
    out_dir: Path,
    *,
    expected_count: int | None = None,
) -> dict:
    path = out_dir / f"{name}.jsonl"
    count = write_jsonl(records, path, expected_count=expected_count)
    return {"name": name, "path": str(path), "num_examples": count, "sha256": sha256_file(path)}


def validate_pinned_source(path: Path, source: dict) -> None:
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_size != source["num_bytes"] or actual_sha256 != source["sha256"]:
        raise ValueError(
            f"Pinned source validation failed for {path}: "
            f"size={actual_size}, sha256={actual_sha256}; "
            f"expected size={source['num_bytes']}, sha256={source['sha256']}"
        )


def download_pinned_source(name: str, destination: Path, retries: int) -> Path:
    source = ADVERSARIAL_SQUAD_SOURCES[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_pinned_source(destination, source)
        return destination
    if retries < 1:
        raise ValueError("--download-retries must be at least 1")

    errors = []
    for attempt in range(1, retries + 1):
        temp_path: Path | None = None
        try:
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "dataset-cartography-repro/1.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".partial",
                    delete=False,
                ) as output,
            ):
                temp_path = Path(output.name)
                shutil.copyfileobj(response, output)
                output.flush()
                os.fsync(output.fileno())
            validate_pinned_source(temp_path, source)
            os.replace(temp_path, destination)
            return destination
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(30, 2**attempt))

    details = "; ".join(errors)
    raise RuntimeError(
        f"Unable to obtain checksum-pinned {name}. Place the official file at "
        f"{destination} (expected {source['sha256']}) and rerun. {details}"
    )


def adversarial_source_path(args: Namespace, out_dir: Path, name: str) -> Path:
    source_dir = (
        Path(args.adversarial_source_dir)
        if args.adversarial_source_dir
        else out_dir.parent / "sources" / "squad_adversarial"
    )
    materialized_source = source_dir / f"{name}.jsonl"
    if materialized_source.exists():
        return materialized_source
    return download_pinned_source(name, source_dir / f"{name}.json", args.download_retries)


def materialize_adversarial_source(name: str, source_path: Path, out_dir: Path) -> dict:
    result = materialize_one(
        name,
        local_records(source_path, with_idx=False),
        out_dir,
        expected_count=ADVERSARIAL_SQUAD_SOURCES[name]["num_examples"],
    )
    result["source_path"] = str(source_path)
    result["source_sha256"] = sha256_file(source_path)
    return result


def materialize_squad(args: Namespace, out_dir: Path) -> dict[str, dict]:
    if args.squad_train_json and args.squad_dev_json:
        return {
            "squad_train": materialize_one(
                "squad_train",
                local_records(Path(args.squad_train_json), with_idx=True),
                out_dir,
            ),
            "squad_dev": materialize_one(
                "squad_dev", local_records(Path(args.squad_dev_json), with_idx=False), out_dir
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
    source_path = (
        Path(args.addsent_json)
        if args.addsent_json
        else adversarial_source_path(args, out_dir, "addsent")
    )
    return materialize_adversarial_source("addsent", source_path, out_dir)


def materialize_addonesent(args: Namespace, out_dir: Path) -> dict:
    source_path = (
        Path(args.addonesent_json)
        if args.addonesent_json
        else adversarial_source_path(args, out_dir, "addonesent")
    )
    return materialize_adversarial_source("addonesent", source_path, out_dir)


def materialize_adversarialqa(args: Namespace, out_dir: Path) -> dict:
    if args.adversarialqa_json:
        return materialize_one(
            "adversarialqa",
            flatten_squad_json(Path(args.adversarialqa_json), with_idx=False),
            out_dir,
        )
    dataset = datasets.load_dataset("UCLNLP/adversarial_qa", "adversarialQA")
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
