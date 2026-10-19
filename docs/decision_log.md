# Decision Log

## 2026-10-18
- Phase 1: Synthetic data diversity. Decided to use 5+ layout templates per doc type with varied field labels/phrasing, mixed date/number formats, multi-page docs, missing fields, and injection/fake-PII cases. Labels stored only in separate JSON files, never filenames or document text. Challenge set uses different wording from templates. Fixed seed SEED=20261018. Generator output is byte-deterministic (MD5 match across runs).
- Phase 3: LLM extractor started. Rules baseline frozen as tag baseline-v1. PII redaction implemented; test passes. No paid LLM calls run yet.
