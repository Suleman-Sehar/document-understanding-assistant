# Decision Log

## 2026-10-18
- Phase 1: Synthetic data diversity. Decided to use 5+ layout templates per doc type with varied field labels/phrasing, mixed date/number formats, multi-page docs, missing fields, and injection/fake-PII cases. Labels stored only in separate JSON files, never filenames or document text. Challenge set uses different wording from templates. Fixed seed SEED=20261018. Generator output is byte-deterministic (MD5 match across runs).
- Phase 2 baseline: rules/regex label-aware extractor with date/number normalization; evaluated DEV only. Doc-type accuracy 0.86; F1 varies by field/layout; table/narrative/unknown layouts weaker. No test/challenge evaluation. Targets unchanged.
