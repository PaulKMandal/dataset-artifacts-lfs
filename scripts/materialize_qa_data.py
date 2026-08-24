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
import gzip
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

DEFAULT_MRQA_CONFIGS = ("newsqa", "triviaqa", "searchqa")
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
MRQA_VALIDATION_SOURCES = {
    "newsqa": {
        "url": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz",
        "filename": "NewsQA.jsonl.gz",
        "sha256": "sha256:66bfb10cab2029bbc7d1afaece20c35fac341b1c179d15b70fde22a207f096ae",
        "num_bytes": 3_142_984,
        "num_examples": 4_212,
        "excluded_qids": (),
    },
    "triviaqa": {
        "url": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz",
        "filename": "TriviaQA-web.jsonl.gz",
        "sha256": "sha256:faf8add436de5a5fa81071a4e7190850d7e9a20acc811439e8a127ba8ec25640",
        "num_bytes": 44_971_198,
        "num_examples": 7_785,
        # The official span for this question points to the structural [DOC]
        # marker rather than an occurrence of the answer "doc".  The marker is
        # removed by the standard MRQA context cleanup, so the row is excluded
        # explicitly instead of silently manufacturing a span.
        "excluded_qids": ("355adac432e64303a0d035784b5078c2",),
    },
    "searchqa": {
        "url": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz",
        "filename": "SearchQA.jsonl.gz",
        "sha256": "sha256:c84d2cc02cac5aa9d576ce1cd22900e9d75fe8a37bc795901c36cae6ef9e5ff0",
        "num_bytes": 92_526_612,
        "num_examples": 16_980,
        "excluded_qids": (),
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
        "--mrqa-source-dir",
        default=None,
        help=(
            "Persistent directory for checksum-pinned official MRQA validation archives. "
            "Defaults to OUT_DIR/../sources/mrqa."
        ),
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=6,
        help="Attempts for each checksum-pinned source download.",
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


def clean_mrqa_spaces(text: str) -> str:
    """Apply the whitespace cleanup used by the canonical tau/mrqa loader."""
    return (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" %", "%")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace(" :", ":")
    )


def clean_mrqa_context(context: str) -> str:
    """Match the canonical MRQA structural-marker cleanup."""
    cleaned = (
        context.replace("[PAR] ", "\n\n")
        .replace("[TLE]", "Title:")
        .replace("[SEP]", "\nPassage:")
        .strip()
    )
    for tag in (
        "<Li>",
        "</Li>",
        "<OI>",
        "</OI>",
        "<Ol>",
        "</Ol>",
        "<Dd>",
        "</Dd>",
        "<UI>",
        "</UI>",
        "<Ul>",
        "</Ul>",
        "<P>",
        "</P>",
        "[DOC]",
    ):
        cleaned = cleaned.replace(tag, "")
    return clean_mrqa_spaces(cleaned.strip())


def mrqa_paragraph_records(
    paragraph: dict, *, subset: str, excluded_qids: set[str]
) -> Iterable[dict]:
    """Convert one raw MRQA paragraph using its authoritative character spans."""
    raw_context = paragraph["context"]
    context = clean_mrqa_context(raw_context)
    for qa in paragraph["qas"]:
        qid = str(qa["qid"])
        candidates = []
        for detected in qa.get("detected_answers", []):
            for raw_start, raw_end in detected.get("char_spans", []):
                if raw_start < 0 or raw_end < raw_start or raw_end >= len(raw_context):
                    raise ValueError(
                        f"MRQA row {qid!r} has invalid raw span "
                        f"[{raw_start}, {raw_end}] for context length {len(raw_context)}"
                    )
                answer = clean_mrqa_spaces(raw_context[raw_start : raw_end + 1]).strip()
                start = context.find(answer)
                if start >= 0 and (answer, start) not in candidates:
                    candidates.append((answer, start))
        if not candidates:
            if qid in excluded_qids:
                continue
            raise ValueError(f"MRQA row {qid!r} has no authoritative answer span after cleanup")
        if qid in excluded_qids:
            raise ValueError(f"MRQA row {qid!r} was marked excluded but now has a valid span")
        yield {
            "id": qid,
            "title": subset,
            "context": context,
            "question": clean_mrqa_spaces(qa["question"].strip()),
            "answers": {
                "text": [answer for answer, _ in candidates],
                "answer_start": [start for _, start in candidates],
            },
        }


def raw_mrqa_records(path: Path, source: dict) -> Iterable[dict]:
    """Read a checksum-pinned MRQA JSONL archive without the lossy HF adapter."""
    excluded_qids = set(source.get("excluded_qids", ()))
    seen_exclusions: set[str] = set()
    raw_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))["header"]
        subset = str(header["dataset"])
        for line in handle:
            paragraph = json.loads(line)
            for qa in paragraph["qas"]:
                raw_count += 1
                if str(qa["qid"]) in excluded_qids:
                    seen_exclusions.add(str(qa["qid"]))
            yield from mrqa_paragraph_records(
                paragraph,
                subset=subset,
                excluded_qids=excluded_qids,
            )
    if raw_count != source["num_examples"]:
        raise ValueError(f"MRQA source contained {raw_count} rows; expected {source['num_examples']}")
    if seen_exclusions != excluded_qids:
        raise ValueError(
            f"MRQA exclusions did not match source: found {sorted(seen_exclusions)}, "
            f"expected {sorted(excluded_qids)}"
        )


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


def download_pinned_source(label: str, source: dict, destination: Path, retries: int) -> Path:
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
        f"Unable to obtain checksum-pinned {label}. Place the official file at "
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
    return download_pinned_source(
        name,
        ADVERSARIAL_SQUAD_SOURCES[name],
        source_dir / f"{name}.json",
        args.download_retries,
    )


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
    if config_name not in MRQA_VALIDATION_SOURCES:
        raise ValueError(
            f"Unsupported checksum-pinned MRQA config {config_name!r}; "
            f"choose from {sorted(MRQA_VALIDATION_SOURCES)}"
        )
    source = MRQA_VALIDATION_SOURCES[config_name]
    source_dir = (
        Path(args.mrqa_source_dir)
        if args.mrqa_source_dir
        else out_dir.parent / "sources" / "mrqa"
    )
    source_path = download_pinned_source(
        f"MRQA {config_name}",
        source,
        source_dir / source["filename"],
        args.download_retries,
    )
    name = f"mrqa_{config_name}"
    result = materialize_one(
        name,
        raw_mrqa_records(source_path, source),
        out_dir,
        expected_count=source["num_examples"] - len(source.get("excluded_qids", ())),
    )
    result["source_path"] = str(source_path)
    result["source_sha256"] = sha256_file(source_path)
    result["excluded_qids"] = list(source.get("excluded_qids", ()))
    return result


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
