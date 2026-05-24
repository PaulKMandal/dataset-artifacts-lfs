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





def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit("repair implementation is incomplete; apply the remaining commits")


if __name__ == "__main__":
    main()
