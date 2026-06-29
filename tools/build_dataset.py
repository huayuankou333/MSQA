#!/usr/bin/env python3
"""Convert the raw MSQA workbook into the clean public release format.

This is the *authoring* tool used to produce the files distributed on the
Hugging Face Hub and shipped under ``data/``. End users do not need to run it;
they simply download the released ``msqa.jsonl`` / ``msqa.csv``.

Usage:
    python tools/build_dataset.py --input MSQA_expanded_v2.xlsx --outdir data
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

RAW_SHEET = "合集"

# Source category labels (Chinese) -> public English cultural-dimension names.
# Names and counts match the paper (History 261, Beliefs 189, Social Norms 186,
# Language 220, Cultural Products 208).
CATEGORY_MAP = {
    "历史与集体记忆": "History and Collective Memory",
    "信仰、价值观与知识体系": "Beliefs, Values, and Knowledge Systems",
    "社会规范与习俗": "Social Norms and Customs",
    "语言表达与沟通艺术": "Language Expression and Communication Arts",
    "文化产物与符号": "Cultural Products and Symbols",
}

# Raw column -> public column. Internal QC / bookkeeping columns are dropped.
COLUMN_MAP = {
    "prompt_id": "id",
    "session_id": "session_id",
    "language": "language",
    "culture_circle": "culture_circle",
    "category": "category_zh",        # replaced by English "category" below
    "prompt": "question",
    "answer": "answer",
    "question_zh": "question_zh",
    "answer_zh": "answer_zh",
    "source_url": "source_url",
    "source_url_desc": "source_url_desc",
}

PUBLIC_FIELDS = [
    "id",
    "session_id",
    "language",
    "culture_circle",
    "category",
    "question",
    "answer",
    "question_zh",
    "answer_zh",
    "source_url",
    "source_url_desc",
]


def build(input_path: Path, outdir: Path) -> None:
    df = pd.read_excel(input_path, sheet_name=RAW_SHEET, engine="openpyxl")
    df = df.rename(columns=COLUMN_MAP)

    unknown = sorted(set(df["category_zh"]) - set(CATEGORY_MAP))
    if unknown:
        raise ValueError(f"Unmapped category labels: {unknown}")
    df["category"] = df["category_zh"].map(CATEGORY_MAP)

    # Keep only public fields, fill NaNs with empty strings, stringify everything
    # (some source cells are parsed as datetimes/numbers by openpyxl).
    out = df[PUBLIC_FIELDS].copy()
    out = out.where(pd.notna(out), "")
    for col in out.columns:
        out[col] = out[col].map(lambda v: "" if v == "" else str(v).strip())

    outdir.mkdir(parents=True, exist_ok=True)
    records = out.to_dict(orient="records")

    jsonl_path = outdir / "msqa.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = outdir / "msqa.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} items")
    print(f"  - {jsonl_path}")
    print(f"  - {csv_path}")
    print("\nLanguages:")
    print(out["language"].value_counts().to_string())
    print("\nCultural dimensions:")
    print(out["category"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public MSQA dataset files.")
    parser.add_argument("--input", type=Path, required=True, help="Raw MSQA_expanded_v2.xlsx")
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.input, args.outdir)
