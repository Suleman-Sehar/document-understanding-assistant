# Mentor Review

Repo: https://github.com/Suleman-Sehar/document-understanding-assistant

## Project summary

The Document Understanding Assistant extracts three outputs per document:
document type (`doc_type`), schema-validated extracted fields, and a structured
summary for invoices, purchase orders, and short contracts. A frozen rules
baseline (`baseline-v1`) is compared against LLM extraction using two models
(`google/gemma-4-31b-it:free` and `openai/gpt-4o-mini`) on a 100-document dev
split. Guardrails (PII redaction, prompt-injection detection,
missing-core-field,
confidence, arithmetic) route ~33% of dev documents to a human
review queue.
The dataset is fully synthetic (fixed seed, 5+ layouts per doc
type); no real documents were used. Dev LLM accuracy on values
present in the body is 0.996 (free) / 0.997 (paid).

## Results

### Dev evaluation (99 of 100 docs, two models separate, not blended)

| metric | free model (48 docs) | gpt-4o-mini (51 docs) | weighted |
|---|---|---|---|
| doc-type accuracy | 0.808 | 0.863 | 0.836 |
| schema-valid rate | 1.000 | 1.000 | 1.000 |
| accuracy on values present in body | 0.996 | 0.997 | — |
| line_items F1 | 0.808 | 0.889 | 0.851 |
| parties F1 | 0.923 | 0.952 | 0.936 |
| total F1 | 0.967 | 0.971 | — |
| delivery_date F1 | 0.769 | 0.714 | — |

Review rate: 33% (~33 of 99 docs).

### Test evaluation — rules-baseline fallback, NOT LLM results

Only 13 of 120 test docs were evaluated before the free-tier daily quota
(50 requests/day) was exhausted. The free model was rate-limited (HTTP 429)
on most test docs and fell back to the rules baseline, so test accuracy
(0.4643 on values present in body) is **not** LLM performance and should not
be compared to dev LLM results. A full LLM test evaluation requires a paid
API key or waiting for quota to allow 120 uninterrupted requests.

## Known limitations

- Generator gaps: 130 labelled dev values (18.1% of dev) never reach the
  document body (97 layout omissions + 33 PDF truncation at 110 chars), so
  per-field F1 is a lower bound.
- Partial test split: 13/120 docs evaluated; results reflect rules-baseline
  fallbacks, not LLM extraction.
- 429 fallback issue on test docs: free model rate-limited upstream; most
  test docs did not reach the model.

## Questions for the mentor

1. Should we regenerate the dev split to fix the generator gaps (PDF line
   truncation at 110 chars), knowing this invalidates the frozen baseline
   `baseline-v1` and all 87 cached model outputs?
2. Do we have budget/credits for a full LLM test evaluation (120 docs), or
   should test results stay labeled as rules-baseline fallback?
3. Is the 33% review rate acceptable for an accounts-payable operations
   stakeholder, or should we tighten the confidence threshold?

## Mentor feedback:
