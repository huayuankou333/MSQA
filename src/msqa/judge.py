#!/usr/bin/env python3
"""Stage 2 - judge generated answers for semantic correctness.

An LLM judge compares each candidate answer against the gold answer and returns
a binary verdict (Yes/No). Reads the JSONL produced by ``msqa.generate`` and
writes a new JSONL with the verdict attached to every record. Resumable:
records that already carry a verdict are skipped.

Example:
    export MSQA_JUDGE_API_KEY=...  MSQA_JUDGE_BASE_URL=...
    python -m msqa.judge \
        --responses runs/gpt-5.2_pt.jsonl \
        --judge-model gemini-3.1-pro \
        --output runs/gpt-5.2_pt.judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .llm_client import LLMClient

LOGGER = logging.getLogger("msqa.judge")
_THREAD_LOCAL = threading.local()


JUDGE_PROMPT_TEMPLATE = """You are judging whether a model answer is semantically correct.

Question:
{question}

Reference answer:
{reference_answer}

Candidate answer:
{candidate_answer}

Decision rules:
1. Output Yes only when the candidate answer and reference answer are semantically equivalent for the question.
2. Accept paraphrases, minor wording differences, accent/case differences, and extra information if it does not contradict the reference.
3. Output No if the candidate is wrong, incomplete, too vague, evasive, contradicts the reference, or only gives related background without the required answer.
4. Be strict for names, dates, places, numbers, and culturally specific facts.

Return only valid JSON in this exact schema:
{{"judgment":"Yes" or "No","reason":"brief reason; empty string if Yes"}}"""


def _client(args: argparse.Namespace) -> LLMClient:
    if not hasattr(_THREAD_LOCAL, "client"):
        _THREAD_LOCAL.client = LLMClient(purpose="judge")
    return _THREAD_LOCAL.client


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def parse_verdict(text: str) -> Dict[str, str]:
    """Parse the judge output into {'judgment': 'Yes'|'No', 'reason': str}."""
    raw = (text or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif raw.startswith("```"):
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        judgment = str(parsed.get("judgment", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()
        if judgment in {"yes", "y", "correct"}:
            return {"judgment": "Yes", "reason": ""}
        if judgment in {"no", "n", "incorrect"}:
            return {"judgment": "No", "reason": reason or "Semantic mismatch."}

    # Fallback: scan free text.
    match = re.search(r'"?judgment"?\s*:\s*"?(Yes|No|Correct|Incorrect)"?', raw, re.IGNORECASE)
    if match:
        if match.group(1).lower() in {"yes", "correct"}:
            return {"judgment": "Yes", "reason": ""}
        return {"judgment": "No", "reason": "Semantic mismatch."}
    raise ValueError(f"Could not parse judge response: {text[:200]}")


def _judge_one(record: Dict, args: argparse.Namespace) -> Dict:
    out = dict(record)
    candidate = str(record.get("response", "")).strip()
    if not candidate:
        out.update({"judgment": "No", "judge_score": 0,
                    "judge_reason": "Empty candidate answer.", "judge_model": args.judge_model})
        return out
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=record.get("question", ""),
        reference_answer=record.get("answer", ""),
        candidate_answer=candidate,
    )
    try:
        text = _client(args).call(
            messages=[{"role": "user", "content": prompt}],
            model=args.judge_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        verdict = parse_verdict(text)
        out.update({
            "judgment": verdict["judgment"],
            "judge_score": 1 if verdict["judgment"] == "Yes" else 0,
            "judge_reason": verdict["reason"],
            "judge_model": args.judge_model,
        })
    except Exception as exc:  # noqa: BLE001
        out.update({"judgment": "ERROR", "judge_score": None,
                    "judge_reason": repr(exc), "judge_model": args.judge_model})
    out["judge_updated_at"] = datetime.now().isoformat(timespec="seconds")
    return out


def _read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _needs_judging(record: Dict) -> bool:
    verdict = record.get("judgment")
    return verdict not in {"Yes", "No"}


def run(args: argparse.Namespace) -> None:
    if not args.responses.exists():
        raise SystemExit(f"Responses file not found: {args.responses}")

    source = _read_jsonl(args.responses)

    # Resume from existing output if present.
    judged_by_key: Dict[tuple, Dict] = {}
    if args.output.exists() and not args.overwrite:
        for rec in _read_jsonl(args.output):
            judged_by_key[(str(rec.get("id")), int(rec.get("run", 1)))] = rec

    records: List[Dict] = []
    for rec in source:
        key = (str(rec.get("id")), int(rec.get("run", 1)))
        records.append(judged_by_key.get(key, rec))

    pending = [r for r in records if _needs_judging(r)]
    LOGGER.info("%s records; %s pending judgments; workers=%s.",
                len(records), len(pending), args.workers)
    if args.dry_run:
        for r in records[:3]:
            print(r.get("id"), "|", str(r.get("answer"))[:40], "||", str(r.get("response"))[:60])
        return

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_judge_one, r, args): idx
                       for idx, r in enumerate(records) if _needs_judging(r)}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                records[idx] = future.result()
                done += 1
                if done % 25 == 0 or done == len(pending):
                    LOGGER.info("Judged %s/%s", done, len(pending))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    LOGGER.info("Done. Wrote judged answers to %s", args.output)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge MSQA answers for correctness.")
    parser.add_argument("--responses", type=Path, required=True,
                        help="JSONL produced by msqa.generate.")
    parser.add_argument("--output", type=Path, required=True, help="Judged JSONL output path.")
    parser.add_argument("--judge-model", default="gemini-3.1-pro", help="LLM judge model id.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
