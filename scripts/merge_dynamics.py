#!/usr/bin/env python3
"""Merge rank-specific dynamics files into training_dynamics.jsonl.

Useful after distributed training, where the logger writes
training_dynamics.rank0.jsonl, training_dynamics.rank1.jsonl, etc.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing training_dynamics.rank*.jsonl")
    parser.add_argument("--output", default=None, help="Merged output path")
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("training_dynamics.rank*.jsonl"))
    if not files:
        raise SystemExit(f"No rank-specific dynamics files found in {root}")
    out = Path(args.output) if args.output else root / "training_dynamics.jsonl"
    with out.open("w", encoding="utf-8") as w:
        for path in files:
            with path.open("r", encoding="utf-8") as r:
                for line in r:
                    if line.strip():
                        w.write(line)
    print(f"Merged {len(files)} files into {out}")


if __name__ == "__main__":
    main()
