---
license: cc-by-4.0
task_categories:
  - question-answering
language:
  - en
  - zh
  - pt
  - th
  - ru
  - ko
  - fr
  - ja
  - ms
  - id
  - es
pretty_name: MSQA
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: test
        path: msqa.jsonl
---

# MSQA Dataset Card

**MSQA** (Multilingual and Multicultural Question Answering) is a benchmark of
**1,064 natively sourced questions** measuring whether large language models
possess genuine, locally grounded cultural knowledge — as opposed to fluency
that merely *looks* culturally competent.

Every question was authored or curated from native, in-language sources (not
translated from English), and each has a single verifiable answer.

## Files

| File | Description |
|------|-------------|
| `msqa.jsonl` | The benchmark, one JSON object per line (UTF-8). |
| `msqa.csv`   | The same data as CSV (UTF-8 with BOM). |

A single split is provided: **`test`** (1,064 items).

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique item id (prefixed by language, e.g. `PT-01`). |
| `session_id` | string | Source authoring/session id. |
| `language` | string | BCP-47-style code, e.g. `pt-PT`, `zh-ZH`, `en-EN`. |
| `culture_circle` | string | Cultural sphere the item targets (e.g. `Portuguese`, `Latin American`). |
| `category` | string | One of the five cultural dimensions (see below). |
| `question` | string | The question, in its native language. |
| `answer` | string | The single gold answer. |
| `question_zh` | string | Chinese translation of the question (reference aid; may be empty). |
| `answer_zh` | string | Chinese translation of the answer (reference aid; may be empty). |
| `source_url` | string | Primary source URL (may be empty). |
| `source_url_desc` | string | Short description of the source (may be empty). |

## Composition

**Languages (11):**

| Language | Count |   | Language | Count |
|----------|------:|---|----------|------:|
| English (`en-EN`)    | 151 | | Japanese (`ja-JP`)    | 83 |
| Chinese (`zh-ZH`)    | 150 | | Malay (`ms-MY`)       | 82 |
| Thai (`th-TH`)       | 95  | | Indonesian (`id-ID`)  | 81 |
| Russian (`ru-RU`)    | 92  | | Spanish (`es-ES`)     | 80 |
| Korean (`ko-KR`)     | 86  | | Portuguese (`pt-PT`)  | 80 |
| French (`fr-FR`)     | 84  | | | |

**Cultural dimensions (5):**

| Dimension | Count |
|-----------|------:|
| History and Collective Memory | 261 |
| Language Expression and Communication Arts | 220 |
| Cultural Products and Symbols | 208 |
| Beliefs, Values, and Knowledge Systems | 189 |
| Social Norms and Customs | 186 |

> **Note on difficulty tiers.** The paper discusses three difficulty tiers
> (Easy/Medium/Hard). This release does not ship a per-item difficulty column; if
> you need tier labels, refer to the paper's appendix taxonomy.

## Loading

```python
# From the Hugging Face Hub
from datasets import load_dataset
ds = load_dataset("m-a-p/MSQA", split="test")

# From a local file (this repo)
from msqa.data import load_dataset
items = load_dataset(path="data/msqa.jsonl", language="pt-PT")
```

## Provenance

Items were sourced from native-language references (encyclopedias, official
sources, cultural documentation). `source_url` records the primary source where
available. The clean release format is produced from the internal workbook by
[`tools/build_dataset.py`](../tools/build_dataset.py).

## License

Released under **CC BY 4.0**. Individual items reference third-party sources via
`source_url`; please also respect the terms of those original sources.

## Citation

See the repository [`README.md`](../README.md#citation).
