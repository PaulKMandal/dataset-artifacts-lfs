"""Crash-tolerant helpers for Hugging Face Datasets transform caching."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _scratch_cache_file(description: str) -> str | None:
    root = os.environ.get("DATASET_ARTIFACTS_MAP_CACHE_DIR")
    if not root:
        return None
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", description).strip("-") or "dataset-map"
    return str(directory / f"{slug}.arrow")


def map_with_cache_recovery(
    dataset: Any,
    function: Callable[..., Any],
    *,
    cache_error_types: type[BaseException] | tuple[type[BaseException], ...],
    description: str,
    **map_kwargs: Any,
) -> Any:
    """Run ``Dataset.map`` and repair a stale/corrupt transform cache once.

    Hugging Face Datasets stores processed ``Dataset.map`` results as Arrow
    files. An abrupt host reset can leave one of those derived cache files
    truncated. The next run then fails while *loading the cache*, before the
    transformation function is called. In that case it is safe to recompute
    the transform from the authoritative source dataset.

    The retry deliberately sets ``load_from_cache_file=False``. Datasets will
    recompute the transform and replace the derived cache with a fresh result.
    If recomputation itself fails, the second exception is allowed to escape;
    this helper never hides a real preprocessing/data error.
    """

    if "cache_file_name" not in map_kwargs:
        scratch_file = _scratch_cache_file(description)
        if scratch_file is not None:
            map_kwargs = dict(map_kwargs)
            map_kwargs["cache_file_name"] = scratch_file

    try:
        return dataset.map(function, **map_kwargs)
    except cache_error_types as exc:
        if map_kwargs.get("load_from_cache_file") is False:
            raise
        print(
            f"Detected unreadable Hugging Face Datasets Arrow cache during {description}: {exc}",
            flush=True,
        )
        print(
            "Recomputing this transform once with load_from_cache_file=False.",
            flush=True,
        )
        retry_kwargs = dict(map_kwargs)
        retry_kwargs["load_from_cache_file"] = False
        return dataset.map(function, **retry_kwargs)
