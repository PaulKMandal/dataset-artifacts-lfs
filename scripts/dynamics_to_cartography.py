#!/usr/bin/env python3
"""Thin wrapper around dynamics.py for users who prefer an explicit converter name."""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    runpy.run_path(str(repo_root / "dynamics.py"), run_name="__main__")
