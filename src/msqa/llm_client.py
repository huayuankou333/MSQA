#!/usr/bin/env python3
"""Unified, OpenAI-compatible LLM client used across the MSQA pipeline.

All three stages of the pipeline (generation, judging, scoring) talk to models
through a single chat-completions interface. Most commercial and open-weight
providers expose an OpenAI-compatible endpoint, so one thin client covers
generation models *and* the LLM judge.

No secrets live in source. Configure credentials with environment variables:

    export MSQA_API_KEY=sk-...                 # required
    export MSQA_BASE_URL=https://api.openai.com/v1   # or any OAI-compatible gateway

You can also keep per-purpose overrides (handy when the judge lives on a
different gateway than the models under test):

    export MSQA_JUDGE_API_KEY=...
    export MSQA_JUDGE_BASE_URL=...

For backward compatibility with the original release scripts, GEMINI_API_KEY /
GEMINI_BASE_URL and OPENAI_API_KEY / OPENAI_BASE_URL are also honored.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


class LLMClient:
    """Minimal retrying wrapper around an OpenAI-compatible chat endpoint.

    Parameters
    ----------
    api_key, base_url:
        Explicit credentials. When omitted, they are read from the environment
        (see module docstring). ``purpose="judge"`` additionally consults the
        ``MSQA_JUDGE_*`` variables first, so the judge can use a separate
        gateway from the models under test.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        purpose: str = "model",
    ) -> None:
        if purpose == "judge":
            api_key = api_key or _first_env(
                "MSQA_JUDGE_API_KEY", "MSQA_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"
            )
            base_url = base_url or _first_env(
                "MSQA_JUDGE_BASE_URL", "MSQA_BASE_URL", "GEMINI_BASE_URL", "OPENAI_BASE_URL"
            )
        else:
            api_key = api_key or _first_env(
                "MSQA_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"
            )
            base_url = base_url or _first_env(
                "MSQA_BASE_URL", "OPENAI_BASE_URL", "GEMINI_BASE_URL"
            )

        if not api_key:
            raise ValueError(
                "Missing API key. Set MSQA_API_KEY (or pass api_key=...). "
                "See msqa/llm_client.py for all accepted environment variables."
            )

        self.api_key = api_key
        self.base_url = _normalize_base_url(base_url or DEFAULT_BASE_URL)

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The OpenAI Python SDK is required. Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def call(
        self,
        messages: Sequence[dict],
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Send a chat-completions request and return the message text.

        Retries with exponential backoff on transient errors. Authentication /
        quota errors fail fast (retrying will not help).
        """
        request: dict = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_effort:
            request["extra_body"] = {"reasoning_effort": reasoning_effort}

        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, max_retries) + 1):
            try:
                if stream:
                    chunks: List[str] = []
                    for chunk in self._client.chat.completions.create(stream=True, **request):
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            chunks.append(delta)
                    return "".join(chunks)

                response = self._client.chat.completions.create(**request)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if _is_fatal(exc) or attempt == max(1, max_retries):
                    break
                time.sleep(retry_delay * attempt)

        raise RuntimeError(f"LLM call failed after {max_retries} attempt(s): {last_error}") from last_error


def _is_fatal(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = ["401", "403", "invalid api key", "unauthorized", "quota", "insufficient"]
    return any(marker in text for marker in markers)


# Backward-compatible aliases for the original release scripts.
GeminiClient = LLMClient
Gemini3Client = LLMClient
