# Document Understanding Assistant

This document-understanding assistant extracts three outputs from each document: the document type (`doc_type`), the extracted fields validated against a schema, and a structured summary.

## Stakeholder

Intended for an accounts-payable or operations lead who processes vendor invoices, purchase orders, and short contracts. Measurable benefit: on the dev split the LLM extractor scores 0.85+ F1 on line_items and 0.93+ F1 on parties where the frozen rules baseline scores roughly 0.2-0.3, so far less manual keying is needed. Documents flagged by guardrails go to a human review queue (~33% of dev docs), so reviewers only touch exceptions instead of every document.

## Prerequisites

- Python 3.11+
- An OpenRouter API key (https://openrouter.ai/keys) for LLM extraction, **or** run baseline-only mode with no key at all (no key needed for the frozen rules baseline).

## Installation

Five commands, run in one PowerShell session in a fresh clone on Windows:

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `copy .env.example .env`
5. `$env:PYTHONPATH="src"; .venv\Scripts\python.exe src\dua\evaluate.py --split dev --limit 5`

Step 5 runs the frozen rules baseline over 5 dev documents and requires no API key. Set `OPENAI_API_KEY` in `.env` to enable LLM extraction with `google/gemma-4-31b-it:free` (free) or a paid model such as `openai/gpt-4o-mini`.

## Running without an API key

Leave `OPENAI_API_KEY` unset or empty. The extractor cannot reach the model, so every document falls back to the frozen rules baseline (`baseline-v1`). This produces **rules-based results, not LLM results**: doc-type accuracy ~0.86 and much lower F1 on list-valued fields (e.g. line_items ~0.2-0.3). Use this mode only to verify the pipeline; do not quote baseline numbers as LLM performance.

## Key results

### Section 1: Dev evaluation (99 of 100 docs; two models, reported separately, not blended)

| metric | free model `google/gemma-4-31b-it:free` (48 docs) | `openai/gpt-4o-mini` (51 docs) | weighted |
|---|---|---|---|
| doc-type accuracy | 0.808 | 0.863 | 0.836 |
| schema-valid rate | 1.000 (48/48) | 1.000 (47/47) | 1.000 |
| accuracy on values present in body | 0.996 (228/229) | 0.997 (285/286) | — |
| line_items F1 | 0.808 | 0.889 | 0.851 |
| parties F1 | 0.923 | 0.952 | 0.936 |
| value F1 | 0.783 | 0.706 | 0.749 |
| delivery_date F1 | 0.769 | 0.714 | 0.740 |

Review rate: 33% (~33 of 99 dev docs sent to the review queue).

Top 3 field F1 gains over the frozen rules baseline (dev): line_items (+0.58 free / +0.60 paid), parties (+0.67 free / +0.64 paid), value (+0.65 free / +0.54 paid).

### Section 2: Test evaluation — partial (API quota exhausted at 13/120 docs)

Free model `google/gemma-4-31b-it:free`, evaluated subset.

| metric | value |
|---|---|
| doc-type accuracy | 1.0 (evaluated subset) |
| schema-valid rate | 1.0 |
| accuracy on values present in body | 0.4643 |
| review rate | 0.8 |
| line_items F1 | 0.0 (evaluated subset) |
| parties F1 | 0.5714 (evaluated subset) |
| value F1 | 0.75 (evaluated subset) |

The test split is partial: only 13 of 120 test documents were evaluated before the daily free-tier quota (50 requests/day) was exhausted. Per-field F1 **gains** over baseline are not reported for test because the baseline comparison is incomplete on the partial split; the field F1 values above are raw scores on the evaluated subset, not gains.

0 of the evaluated test docs had generator gaps (no `source_truncated` or `missing_required_field` in their issues). The 0.4643 vs dev 0.996 gap is because the free model was rate-limited (HTTP 429) on nearly every test doc and fell back to the rules baseline. Warning: the 0.4643 accuracy figure reflects rules-baseline fallbacks, not LLM extraction. The free model (google/gemma-4-31b-it:free) was rate-limited (HTTP 429) on most of the 13 test docs, so the extractor fell back to the rules baseline. These numbers are not LLM results and should not be compared to the dev LLM results. A full LLM test evaluation requires either a paid API key or waiting for the free-tier daily quota to allow 120 uninterrupted requests.

## Project structure

```
document-understanding-assistant/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── evidence/
│   └── review_queue.json
├── data/
│   ├── manifest.json
│   ├── dev/
│   │   ├── docs/      (100 documents: doc_0001 … doc_0100, .pdf/.docx)
│   │   └── labels/    (100 label JSON files)
│   ├── test/
│   │   ├── docs/      (120 documents: doc_0101 … doc_0220)
│   │   └── labels/    (120 label JSON files)
│   └── challenge/
│       ├── docs/      (30 documents: doc_0221 … doc_0250)
│       └── labels/    (30 label JSON files)
├── docs/
│   ├── decision_log.md
│   ├── error_analysis.md
│   ├── requirements.md
│   ├── responsible_use.md
│   └── work_log.md
├── scripts/
│   ├── analyze_gaps.py
│   ├── check_model_routing.py
│   ├── eval_baseline_dev.py
│   ├── eval_llm_dev.py
│   ├── generate_synthetic.py
│   ├── probe_free_models.py
│   ├── reprocess_cached.py
│   └── run_dev.py
├── src/
│   └── dua/
│       ├── baseline.py
│       ├── evaluate.py
│       ├── guardrails.py
│       ├── llm.py
│       ├── llm_extractor.py
│       └── schemas.py
└── tests/
    ├── conftest.py
    ├── test_debug_logging.py
    ├── test_guardrails_phase4.py
    ├── test_injection_guardrail.py
    ├── test_line_item_normalization.py
    ├── test_llm_extractor.py
    ├── test_missing_required_field.py
    ├── test_model_routing.py
    ├── test_null_required_field.py
    ├── test_pii_redaction.py
    ├── test_request_budget.py
    ├── test_retry_backoff.py
    └── test_schema_repair.py
```

## Acceptance criteria

| target | actual | status |
|---|---|---|
| Dev doc-type accuracy ≥ 0.80 | 0.836 weighted | PASS |
| Dev schema-valid rate = 1.000 | 1.000 | PASS |
| Dev accuracy on values present in body ≥ 0.99 | 0.996 free / 0.997 paid | PASS |
| Test split fully evaluated (120/120) | 13/120 evaluated | PARTIAL |
| All tests pass | 51 passed | PASS |

## Known limitations

- Generator gaps: 130 labelled dev values (18.1% of dev) never reach the document body (97 layout omissions + 33 PDF truncation), so per-field F1 is a lower bound.
- Partial test split: only 13 of 120 test documents evaluated before the free-tier quota was exhausted.
- Free-model 429 rate: the free pool rate-limits independently of the key; the extractor retries but a day's quota can still run out.
- delivery_date zero delta on gpt-4o-mini (baseline 0.714 = LLM 0.714) due to slash-date ambiguity.
- No multilingual support: English-language documents only.
- Single label per doc: each document carries exactly one doc_type; mixed or multi-type documents are not supported.

## Future work

1. Validate on real invoices, purchase orders, and contracts instead of synthetic documents only.
2. Replace the fixed 0.6 confidence threshold with a calibrated confidence score.
3. Add multilingual support with language detection and localized date/number formats.
4. Add audit logging of the served model, prompt version, cache hit/miss, and reviewer decision per document.
5. Fix the PDF line-truncation defect before the final test or challenge evaluation.

## License

MIT. Suleman Sehar, 2026.

## Contact

Suleman Sehar — https://github.com/Suleman-Sehar
