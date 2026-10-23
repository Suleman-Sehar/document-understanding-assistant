# Responsible use notes (Phase 6)

## 1. PII

The extractor redacts PII before any LLM call. `redact_pii` in
`src/dua/guardrails.py`
scans the document body for email, phone, card, SSN, IP address, and API-key
patterns and replaces each match with a typed placeholder such as `[EMAIL]` or
`[CARD]`, so the model never sees raw secrets. The typed placeholder preserves
field shape for downstream normalization without leaking the value. A regression
test in `tests/test_pii_redaction.py` asserts every pattern is replaced and that
clean text is unchanged. Residual risk: pattern-based redaction can miss unusual
formats (international phone numbers, obfuscated emails) and can over-redact
legitimate tokens that match a pattern, so a human reviewer should still inspect
flagged documents before any downstream use.

## 2. Prompt injection

The injection guardrail fires `possible_prompt_injection` when the model answers
`business_document` on a non-empty body, treating an unclassifiable answer as a
signal rather than trusting it. `tests/test_injection_guardrail.py` covers a
business_document answer on a non-empty body being flagged and an empty document
not being flagged. Residual risk: this heuristic flags unclassifiable documents,
not actual injected text, so it can both miss real injection (false negative)
and flag legitimate unusual documents (false positive). The body-level label
`meta.injection` is the authoritative signal; a text-based scan of the body is
the correct long-term implementation.

## 3. Human oversight

Every extracted document passes through the review queue builder
(`build_review_queue_entry` in `src/dua/guardrails.py`). Documents with any
guardrail issue, a missing core field, or confidence below 0.6 are marked
`needs_human_review`. On dev this produced a 33% review rate (about 33 of 99
documents): missing_field 6, low_confidence 6, injection 4,
business_document 3, source_truncated 10 (arithmetic_inconsistency 0). A human
reviewer opens `review_queue.json`, reads the reason and the raw debug output in
`evidence/debug_<doc_id>.json`, and either approves, corrects, or rejects the
extraction before the result is used downstream.

## 4. Fairness and data

The dataset is fully synthetic: documents are generated from a fixed seed
(SEED=20261018) with 5+ layout templates per doc type, varied field labels,
mixed date/number formats, missing fields, and injection/fake-PII cases. No real
document was used to validate the extractor, so there is no evidence of
performance on real-world invoices, purchase orders, or contracts. Known gaps
affect fairness of measurement: 130 labelled dev values never reach the body
(97 layout omissions + 33 PDF truncation), so per-field F1 is a lower bound and
`parties` is systematically under-scored on narrative layouts. Results should
not be extrapolated to real documents.

## 5. Privacy

No real documents are used anywhere in this project: training, dev, test, and
challenge sets are all synthetic. The cache directory holds only synthetic
document outputs keyed by document text, model, and prompt version; no real
customer or PII-bearing data is cached. Cached entries are written atomically
(temp file plus replace) and a corrupt entry is treated as a cache miss rather
than leaking partial data. Because only synthetic data is processed, there is no
real-user privacy exposure from the cache or from debug artifacts.

## 6. Limitations

- Generator gaps: 130 labelled dev values are absent from the body, so F1 over
  all labelled values understates true extractor quality.
- API budget constraint: the free tier allows 50 requests per day per key and
  paid credits were exhausted mid-run, so dev evaluation was split across two
  models and not blended.
- No multilingual support: prompts, normalization, and date parsing assume
  English-language documents.
- Single label per doc: each document carries exactly one doc_type label; mixed
  or multi-type documents are not supported.

## 7. Production next steps

- Validate on real documents (invoices, purchase orders, contracts) before any
  production rollout; synthetic F1 does not transfer.
- Replace the fixed 0.6 confidence threshold with a calibrated confidence score
  backed by held-out data.
- Add multilingual support (language detection, localized date/number formats).
- Add audit logging: record the served model id, prompt version, cache hit/miss,
  and reviewer decision for every document.
- Fix the generator (PDF line truncation at 110 chars) before the final test or
  challenge evaluation so gaps stop inflating error rates.
