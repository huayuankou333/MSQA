#!/usr/bin/env python3
"""Stage 3 - compute MSQA metrics (CO / NA / IN / CGA / F-score).

Consumes the judged JSONL from ``msqa.judge`` and reports, per model and per
language / cultural dimension:

    CO   correct answers / all answers
    NA   not-attempted (refusal, hedging, no concrete claim) / all answers
    IN   incorrect concrete answers / all answers
    CGA  CO / (CO + IN)               (correct given attempted)
    F    harmonic mean of CO and CGA  (the headline number in the paper)

A judge marks each answer correct (CO) or not. Wrong answers are split into NA
vs IN by a second LLM pass (``--rejudge``, on by default). With ``--no-rejudge``
every wrong answer is conservatively counted as IN, so CGA == CO and
F == CO -- useful for a quick, API-free sanity check.

Example:
    python -m msqa.score \
        --judged runs/gpt-5.2_pt.judged.jsonl \
        --judge-model gemini-3.1-pro \
        --output runs/gpt-5.2_pt.metrics.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm_client import LLMClient

LOGGER = logging.getLogger("msqa.score")

NA_PROMPT_TEMPLATE = """You are judging an MSQA benchmark answer that was already marked not fully correct.

Classify it as exactly one label:
- NA: The answer does not fully give the reference answer, but it also does not assert a wrong answer. This includes refusal, uncertainty, saying it cannot answer, or giving only non-committal context.
- IN: The answer gives a concrete answer that is wrong, contradicts the reference, or contains a wrong answer.

Return only valid JSON: {{"label":"NA" or "IN","reason":"short reason"}}

Question:
{question}

Reference answer:
{reference_answer}

Candidate answer:
{candidate_answer}
"""


def _read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_na_label(text: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if isinstance(data, dict):
        label = str(data.get("label", "")).strip().upper()
        if label in {"NA", "IN"}:
            return label, str(data.get("reason", "")).strip()
    fallback = re.search(r"\b(NA|IN)\b", raw, flags=re.I)
    if fallback:
        return fallback.group(1).upper(), raw[:200]
    raise ValueError(f"Could not parse NA/IN response: {text[:200]}")


def _cache_key(rec: Dict) -> str:
    return f"{rec.get('model')}::{rec.get('language')}::{rec.get('id')}::{rec.get('run', 1)}"


def _load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if path.exists():
        for rec in _read_jsonl(path):
            if rec.get("key") and rec.get("label"):
                cache[rec["key"]] = rec["label"]
    return cache


def classify(records: List[Dict], args: argparse.Namespace) -> List[Dict]:
    """Attach a 'classification' (CO/NA/IN) to every judged record."""
    cache: Dict[str, str] = {}
    client: Optional[LLMClient] = None
    if args.rejudge:
        cache = _load_cache(args.cache)
        args.cache.parent.mkdir(parents=True, exist_ok=True)

    rejudged = 0
    out: List[Dict] = []
    for rec in records:
        score = rec.get("judge_score")
        rec = dict(rec)
        if score == 1:
            rec["classification"] = "CO"
            out.append(rec)
            continue

        # Wrong or unjudged -> NA/IN.
        if not args.rejudge or score is None:
            rec["classification"] = "IN"
            out.append(rec)
            continue

        key = _cache_key(rec)
        label = cache.get(key)
        if label is None and (args.limit_rejudge is None or rejudged < args.limit_rejudge):
            if client is None:
                client = LLMClient(purpose="judge")
            prompt = NA_PROMPT_TEMPLATE.format(
                question=rec.get("question", ""),
                reference_answer=rec.get("answer", ""),
                candidate_answer=rec.get("response", ""),
            )
            try:
                text = client.call(
                    messages=[{"role": "user", "content": prompt}],
                    model=args.judge_model,
                    max_tokens=256,
                    temperature=0.0,
                    max_retries=args.max_retries,
                )
                label, _reason = parse_na_label(text)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("NA/IN rejudge failed for %s (%s); counting as IN.", key, exc)
                label = "IN"
            with args.cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "label": label}, ensure_ascii=False) + "\n")
            rejudged += 1
        rec["classification"] = label or "IN"
        out.append(rec)
    if args.rejudge:
        LOGGER.info("Rejudged %s wrong answers into NA/IN this run.", rejudged)
    return out


def _metric_row(group: List[Dict], extra: Dict) -> Dict:
    total = len(group)
    co = sum(1 for r in group if r["classification"] == "CO")
    na = sum(1 for r in group if r["classification"] == "NA")
    inc = sum(1 for r in group if r["classification"] == "IN")
    co_rate = co / total if total else float("nan")
    cga = co / (co + inc) if (co + inc) else float("nan")
    f_score = (2 * co_rate * cga / (co_rate + cga)) if (co_rate + cga) else float("nan")
    return {
        **extra,
        "n": total,
        "CO": round(co_rate, 4),
        "NA": round(na / total, 4) if total else float("nan"),
        "IN": round(inc / total, 4) if total else float("nan"),
        "CGA": round(cga, 4),
        "F_score": round(f_score, 4),
    }


def compute_metrics(records: List[Dict]) -> Dict[str, List[Dict]]:
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    by_lang: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    by_cat: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in records:
        model = r.get("model", "")
        by_model[model].append(r)
        by_lang[(model, r.get("language", ""))].append(r)
        by_cat[(model, r.get("category", ""))].append(r)

    overall = [_metric_row(g, {"model": m}) for m, g in sorted(by_model.items())]
    per_language = [_metric_row(g, {"model": m, "language": l})
                    for (m, l), g in sorted(by_lang.items())]
    per_category = [_metric_row(g, {"model": m, "category": c})
                    for (m, c), g in sorted(by_cat.items())]
    return {"overall": overall, "per_language": per_language, "per_category": per_category}


def _write_csv(path: Path, rows: List[Dict]) -> None:
    import csv
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if not args.judged.exists():
        raise SystemExit(f"Judged file not found: {args.judged}")
    records = _read_jsonl(args.judged)
    LOGGER.info("Loaded %s judged records.", len(records))

    classified = classify(records, args)
    metrics = compute_metrics(classified)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    _write_csv(Path(f"{stem}.csv"), metrics["overall"])
    _write_csv(Path(f"{stem}.per_language.csv"), metrics["per_language"])
    _write_csv(Path(f"{stem}.per_category.csv"), metrics["per_category"])
    with Path(f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\n=== Overall (per model) ===")
    for row in metrics["overall"]:
        print(f"  {row['model']:<24} F={row['F_score']}  CO={row['CO']}  "
              f"CGA={row['CGA']}  NA={row['NA']}  IN={row['IN']}  n={row['n']}")
    print(f"\nWrote metrics next to {stem}.csv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MSQA CO/NA/IN/CGA/F-score metrics.")
    parser.add_argument("--judged", type=Path, required=True,
                        help="Judged JSONL from msqa.judge (may concatenate several models).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output stem; writes .csv, .per_language.csv, .per_category.csv, .json")
    parser.add_argument("--judge-model", default="gemini-3.1-pro",
                        help="Model used for the NA/IN split.")
    parser.add_argument("--cache", type=Path, default=Path("runs/na_in_cache.jsonl"),
                        help="Resumable cache of NA/IN decisions.")
    parser.add_argument("--limit-rejudge", type=int, default=None,
                        help="Cap the number of NA/IN API calls this run.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-rejudge", dest="rejudge", action="store_false",
                        help="Skip NA/IN API split; count every wrong answer as IN.")
    parser.set_defaults(rejudge=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
