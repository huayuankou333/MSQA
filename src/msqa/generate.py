#!/usr/bin/env python3
"""Stage 1 - generate model answers for MSQA questions.

Each question is sent to the model under test ``--runs`` times (default 1; use
5 to reproduce the Best/Worst-of-N and stability analyses). Results stream to a
JSONL file and the run is fully resumable: re-running skips (id, run) pairs that
already have an answer.

Example:
    export MSQA_API_KEY=...  MSQA_BASE_URL=...
    python -m msqa.generate \
        --data data/msqa.jsonl --language pt-PT \
        --model gpt-5.2 --runs 5 \
        --output runs/gpt-5.2_pt.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .data import load_dataset
from .llm_client import LLMClient

LOGGER = logging.getLogger("msqa.generate")
_THREAD_LOCAL = threading.local()


def _client() -> LLMClient:
    if not hasattr(_THREAD_LOCAL, "client"):
        _THREAD_LOCAL.client = LLMClient(purpose="model")
    return _THREAD_LOCAL.client


def _load_done(path: Path) -> set:
    """Return {(id, run)} pairs already answered (non-empty response)."""
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("response"):
                    done.add((str(rec.get("id")), int(rec.get("run", 1))))
    return done


def _answer_one(item: Dict, run: int, args: argparse.Namespace) -> Dict:
    record = {
        "id": str(item["id"]),
        "language": item.get("language", ""),
        "category": item.get("category", ""),
        "question": item["question"],
        "answer": item["answer"],
        "model": args.model,
        "run": run,
    }
    try:
        record["response"] = _client().call(
            messages=[{"role": "user", "content": item["question"]}],
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        record["error"] = ""
    except Exception as exc:  # noqa: BLE001
        record["response"] = ""
        record["error"] = repr(exc)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return record


def run(args: argparse.Namespace) -> None:
    items = load_dataset(path=args.data, language=args.language, category=args.category)
    if not items:
        raise SystemExit("No items match the given filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else _load_done(args.output)
    mode = "w" if args.overwrite else "a"

    pending = [
        (item, run_idx)
        for item in items
        for run_idx in range(1, args.runs + 1)
        if (str(item["id"]), run_idx) not in done
    ]
    LOGGER.info(
        "%s items x %s runs = %s calls; %s already done; %s pending.",
        len(items), args.runs, len(items) * args.runs, len(done), len(pending),
    )
    if args.dry_run:
        for item in items[:5]:
            print(item["id"], "|", item["question"][:80])
        return
    if not pending:
        LOGGER.info("Nothing to do.")
        return

    write_lock = threading.Lock()
    completed = 0
    with args.output.open(mode, encoding="utf-8") as out, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(_answer_one, item, r, args): (item, r) for item, r in pending}
        for future in as_completed(futures):
            record = future.result()
            with write_lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            completed += 1
            if completed % 20 == 0 or completed == len(pending):
                LOGGER.info("Saved %s/%s (last id=%s run=%s err=%s)",
                            completed, len(pending), record["id"], record["run"],
                            bool(record["error"]))
    LOGGER.info("Done. Wrote answers to %s", args.output)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MSQA model answers.")
    parser.add_argument("--data", type=Path, default=None,
                        help="Local msqa.jsonl. Omit to pull from the Hugging Face Hub.")
    parser.add_argument("--language", default=None, help="Filter to one language, e.g. pt-PT.")
    parser.add_argument("--category", default=None, help="Filter to one cultural dimension.")
    parser.add_argument("--model", required=True, help="Model id served by your endpoint.")
    parser.add_argument("--runs", type=int, default=1, help="Calls per question. Default 1.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Ignore and overwrite existing output.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.runs <= 0:
        raise SystemExit("--runs must be positive.")
    run(args)


if __name__ == "__main__":
    main()
