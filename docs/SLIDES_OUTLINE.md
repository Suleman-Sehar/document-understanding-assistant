# Slides Outline

## Slide 1: Problem and stakeholder

- Stakeholder: accounts-payable / operations lead processing vendor invoices,
  purchase orders, and short contracts.
- Problem: manual keying of document fields is slow and error-prone; need
  automated extraction with human review only on exceptions.
- Output per document: document type (`doc_type`), schema-validated extracted
  fields, and a structured summary.
- Success measure: LLM extractor beats the frozen rules baseline on every
  field with support; reviewers touch only ~33% of documents.

Speaker notes:
We built a document-understanding assistant for an accounts-payable or
operations lead who processes invoices, purchase orders, and contracts. The
goal is three structured outputs per document — type, fields, summary — with
guardrails sending only exceptions to a human review queue so reviewers are
not burdened with every document.

## Slide 2: Approach (pipeline, baseline vs LLM)

- Pipeline: synthetic data (fixed seed, 5+ layouts per doc type) -> LLM
  extractor -> guardrails -> review queue.
- Baseline: frozen rules/regex extractor (`baseline-v1`) with date and number
  normalization.
- LLM: two models compared — `google/gemma-4-31b-it:free` (free) and
  `openai/gpt-4o-mini` (paid), reported separately, not blended.
- Guardrails: PII redaction, prompt-injection detection, missing-core-field,
  confidence scoring, arithmetic consistency.

Speaker notes:
The approach compares a frozen rules baseline against two LLMs on the same
synthetic dev split. Synthetic data keeps privacy safe (no real documents)
while guardrails — PII redaction, injection detection, missing-field,
confidence, and arithmetic checks — decide which documents need human review.

## Slide 3: Dev results (real numbers, two models)

- Doc-type accuracy: free 0.808 / gpt-4o-mini 0.863 / weighted 0.836.
- Schema-valid rate: 1.000 on both models (48/48 and 47/47).
- Accuracy on values present in body: free 0.996 (228/229) / paid 0.997
  (285/286).
- Top F1 gains over baseline: line_items (+0.58 free / +0.60 paid), parties
  (+0.67 free / +0.64 paid), value (+0.65 free / +0.54 paid).

Speaker notes:
Dev results show the LLM extractor dramatically outruns the rules baseline,
especially on list-valued fields like line_items and parties where the
baseline scores roughly 0.2-0.3. Both models reach ~0.996-0.997 accuracy on
values actually present in the document body, and every output is schema-valid.

## Slide 4: Guardrails (review rate, triggers, PII, injection)

- Review rate: 33% (~33 of 99 dev docs sent to human review).
- Triggers: missing_field 6, low_confidence 6, injection 4,
  business_document 3, source_truncated 10.
- PII redaction: typed placeholders ([EMAIL], [CARD], etc.) before the LLM
  call; regression-tested.
- Injection guardrail: flags `business_document` on non-empty bodies
  (detects unclassifiable docs, not injected text).

Speaker notes:
About a third of dev documents go to human review. The triggers break down
into missing core fields, low confidence, possible injection,
business_document answers, and source truncation. PII is redacted to typed
placeholders before the model ever sees the text, and the injection guardrail
flags unclassifiable documents for review.

## Slide 5: Error analysis (generator gaps, delivery_date, doc-type)

- Generator gaps: 130 labelled dev values (18.1% of dev) absent from the
  body — 97 layout omissions + 33 PDF truncation (line cut at 110 chars);
  F1 is a lower bound.
- delivery_date: zero delta on gpt-4o-mini (baseline 0.714 = LLM 0.714) due
  to slash-date ambiguity DD/MM vs MM/DD.
- Doc-type on table layouts: table accuracy 3/6 free, 4/6 paid — tables
  render only rows + `TOTAL`, no type cue; model answers `business_document`
  and loses every field.

Speaker notes:
Most dev errors come from the generator, not the model: 130 labelled values
never reach the document body, so F1 understates true extractor quality.
delivery_date is weak on gpt-4o-mini due to slash-date ambiguity, and table
layouts carry no document-type cue, causing the model to answer
`business_document` and lose all structured fields.

## Slide 6: Limitations (honest, including test split and 429 issue)

- Generator gaps: 130 dev values (18.1%) never reach the body; per-field F1
  is a lower bound.
- Partial test split: only 13/120 docs evaluated; test accuracy (0.4643)
  reflects rules-baseline fallbacks, not LLM results.
- 429 fallback issue: free model rate-limited upstream on most test docs;
  extractor fell back to rules baseline.
- No multilingual support: English-language documents only; single doc_type
  label per document.

Speaker notes:
Be honest about limitations: generator gaps make F1 a lower bound, the test
split is partial and its numbers come from rules-baseline fallbacks after 429
rate limits, and the system supports only English with a single label per
document. None of these are extractor bugs — they are data and infrastructure
constraints.

## Slide 7: Decisions and trade-offs

- Document generator gaps (Option B) rather than regenerate — preserves
  frozen baseline `baseline-v1` and all 87 cached outputs.
- Report dev results separately per model, not blended, due to API credit
  exhaustion mid-run.
- Free model `google/gemma-4-31b-it:free` (50 requests/day) vs paid
  `openai/gpt-4o-mini` — free model costs $0.00 but is rate-limited.
- Fix PDF line truncation before final test/challenge evaluation (one-line
  generator change).

Speaker notes:
Key trade-off: we documented generator gaps instead of regenerating data,
because regenerating would invalidate the frozen baseline and all cached
outputs and cost two to three days of free-tier quota. Dev results are
reported per model because paid credits ran out mid-run, and the PDF
truncation fix is scheduled before any final evaluation.

## Slide 8: Next steps and 10x scale

- Validate on real invoices, purchase orders, and contracts before production
  rollout.
- Replace fixed 0.6 confidence threshold with a calibrated confidence score.
- Add multilingual support (language detection, localized date/number
  formats).
- Add audit logging: served model, prompt version, cache hit/miss, reviewer
  decision per document.
- Scale to 10x volume: batch processing, cached responses, and a paid API key
  to remove free-tier 50-requests/day limit.

Speaker notes:
Next steps: validate on real documents, calibrate the confidence threshold,
add multilingual support, and add per-document audit logging. To scale 10x,
use batch processing, rely on cached responses to avoid repeat API calls, and
move to a paid key so the free-tier daily quota no longer caps throughput.
