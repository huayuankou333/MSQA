#!/usr/bin/env python3
"""Loading helpers for the MSQA benchmark.

The public dataset ships as ``msqa.jsonl`` (one JSON object per line). This
module loads it from a local path or, if ``datasets`` is installed, straight
from the Hugging Face Hub.

Each item has the fields:
    id, session_id, language, culture_circle, category,
    question, answer, question_zh, answer_zh, source_url, source_url_desc
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

HF_DATASET_ID = "m-a-p/MSQA"

REQUIRED_FIELDS = ("id", "language", "category", "question", "answer")


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_dataset(
    path: Optional[Path] = None,
    language: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict]:
    """Load MSQA items, optionally filtered by language and/or category.

    If ``path`` is given it is read directly (``.jsonl``). Otherwise the dataset
    is pulled from the Hugging Face Hub (requires ``pip install datasets``).
    """
    if path is not None:
        items = load_jsonl(Path(path))
    else:
        items = _load_from_hub()

    missing = [f for f in REQUIRED_FIELDS if items and f not in items[0]]
    if missing:
        raise ValueError(f"Dataset is missing required fields: {missing}")

    if language:
        items = [it for it in items if it.get("language") == language]
    if category:
        items = [it for it in items if it.get("category") == category]
    return items


def list_languages(items: List[Dict]) -> List[str]:
    return sorted({it["language"] for it in items})


def _load_from_hub() -> List[Dict]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading from the Hugging Face Hub requires the 'datasets' package "
            "(pip install datasets), or pass a local --data path."
        ) from exc

    ds = hf_load_dataset(HF_DATASET_ID, split="test")
    return [dict(row) for row in ds]
