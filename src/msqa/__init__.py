"""MSQA: a natively sourced multilingual and multicultural QA benchmark.

Pipeline modules:
    msqa.generate  - stage 1: produce model answers
    msqa.judge     - stage 2: LLM-judge answers for correctness
    msqa.score     - stage 3: compute CO/NA/IN/CGA/F-score metrics

Shared helpers:
    msqa.data        - load the benchmark (local JSONL or Hugging Face Hub)
    msqa.llm_client  - OpenAI-compatible client for models and the judge
"""

__version__ = "0.1.0"
