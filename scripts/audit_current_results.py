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
