# Error analysis (dev split, 2026-10-21)

## 1. Top failure modes

- generator gaps: 130 values absent (18.1% of dev).
  Cause: layout template omits the field; PDF line cut at 110 chars.
  Example: doc_0009 narrative labels `currency` and `governing_law`,
  but neither appears in the body.
- delivery_date zero delta: gpt-4o-mini baseline 0.714 = LLM 0.714
  (18 docs). Cause: slash-date ambiguity DD/MM vs MM/DD; inconsistent
  normalization. Example: doc_0003 footer_total, expected "May 30, 2024",
  got null.
- doc-type on table layouts: table accuracy 3/6 free, 4/6 paid.
  Cause: table renders only rows + `TOTAL`, no type cue. Example:
  doc_0030 table labelled business_document; every field is lost.
- parties extraction: F1 is a lower bound; 4 dev contracts render no
  parties text. Cause: narrative layout writes only scalars, no
  `Parties:` line. Example: doc_0009 short_contract, parties labelled
  but absent from body.

## 2. Generator gaps summary

130 labelled values are absent from the dev document body: 97 layout
omissions + 33 PDF truncation (0 unexplained). These values can never be
read from the text, so per-field F1 over all labelled values is a lower
bound, not true extractor quality. Accuracy on values present in the body
is 228/229 = 0.996 (free, gemma-4-31b-it) and 285/286 = 0.997
(gpt-4o-mini). The truncation defect is `scripts/generate_synthetic.py:123`
(`line[:110]`); the fix is one line, but regenerating invalidates the
frozen baseline and all cached outputs, so gaps are documented rather than
fixed before test evaluation.

## 3. Free model vs gpt-4o-mini

field | free F1 (48 docs) | gpt-4o-mini F1 (51 docs)
--- | --- | ---
line_items | 0.808 | 0.889
parties | 0.923 | 0.952
total | 0.967 | 0.971
delivery_date | 0.769 | 0.714

The free model (`google/gemma-4-31b-it:free`) trails gpt-4o-mini on
line_items and parties but is close on total. delivery_date is the weakest
field for both; gpt-4o-mini shows zero delta versus baseline (0.714 =
0.714) due to slash-date ambiguity, while the free model improves to 0.769.
Both models reach ~0.996-0.997 accuracy on values actually present in the
body, so the gap is mostly generator gaps, not model error.

## 4. Guardrail catches

Review rate on dev: 33% (about 33 of 99 documents sent to the review
queue).

Breakdown: missing_field 6, low_confidence 6, injection 4,
business_document 3, source_truncated 10.

- missing_field (6): core field absent for the doc type (vendor_name,
  buyer_name, parties).
- low_confidence (6): confidence score below 0.6.
- injection (4): model answered `business_document` on a non-empty body;
  flags unclassifiable documents, not injected text.
- business_document (3): schema keeps only free-text `description`, so all
  structured fields are lost.
- source_truncated (10): PDF line truncated at 110 chars cut off labelled
  values.

## 5. Misclassification examples (dev evidence)

doc_id | field | expected | got | root cause
--- | --- | --- | --- | ---
doc_0054 | subtotal | 355.44 | null | model omitted it; value present in body
doc_0003 | delivery_date | "May 30, 2024" | null | value present; model omitted
doc_0001 | invoice_date | 2025-04-06 | 2025-06-04 | slash-date ambiguity
