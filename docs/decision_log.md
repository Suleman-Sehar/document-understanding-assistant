# Decision Log

## 2026-10-18
- Phase 1: Synthetic data diversity. Decided to use 5+ layout templates per doc type with varied field labels/phrasing, mixed date/number formats, multi-page docs, missing fields, and injection/fake-PII cases. Labels stored only in separate JSON files, never filenames or document text. Challenge set uses different wording from templates. Fixed seed SEED=20261018. Generator output is byte-deterministic (MD5 match across runs).
- Phase 2 baseline: rules/regex label-aware extractor with date/number normalization; evaluated DEV only. Doc-type accuracy 0.86; F1 varies by field/layout; table/narrative/unknown layouts weaker. No test/challenge evaluation. Targets unchanged.
- Phase 3: LLM extractor started. Rules baseline frozen as tag baseline-v1. PII redaction implemented; test passes. No paid LLM calls run yet.

## 2026-10-19
- Full dev run: LLM 53/100 schema-valid, cost $0.008641, avg latency 1.53s. Baseline doc-type 0.86 vs LLM 0.27; LLM worse on most fields.
## 2026-10-19
- Root cause of 47/100 schema failures: silent fallback with no logging (raw LLM output discarded); optional fields were too strict; doc_type values were under-specified in prompt. Fixes: persist raw LLM JSON + pydantic error to evidence/debug_<doc_id>.json (separate commit); optional fields with None default, explicit ALLOWED_DOC_TYPES, post-validation doc_type repair (separate commit). Dev split only for few-shot. No test/challenge data used.

- 5-doc diagnostic after fixes: schema-valid 5/5, 0 fallbacks, cost \.00103, avg latency 3.25s. Extra fixes: missing core field relaxes to None + needs_human_review instead of discarding LLM output; empty/short documents skip the API and return an issue flag; max_tokens=1500 (OpenRouter 402 on credit); prompt version bumped so the cache invalidates when the prompt changes.
