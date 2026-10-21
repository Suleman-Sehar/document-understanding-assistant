# Error Analysis

## Generator gap: labelled values that never reach the document body (2026-10-21)

`scripts/analyze_gaps.py --split dev --explain` rebuilds the intended documents from the manifest seed (the generator is deterministic) and compares the source text with the rendered file. Of the 717 labelled values in the dev split, 130 never appear in a rendered document body, and every one of them is classified:

| cause | values | docs | meaning |
|---|---|---|---|
| `layout_omission` | 97 | 30 | the layout template never writes the field (by design) |
| `pdf_truncation` | 33 | 17 | the PDF renderer cuts every line at 110 characters |
| `unexplained` | 0 | 0 | - |

Full report: `evidence/generator_gap_analysis_dev_2026-10-21.txt`.

**`layout_omission` (by design).** Each layout template renders a subset of the label: `table` writes only the item rows and `TOTAL:`, `sections` writes only `parties`, `effective_date` and `term_months`, `narrative` writes only scalar fields, `footer_total` writes only the total in the footer. The label carries the complete field set of the document type, so the fields a template does not render can never be recovered. This includes the previously documented `parties` case.

**`pdf_truncation` (a real generator defect).** `scripts/generate_synthetic.py:123` draws every line as `line[:110]`. A `narrative` document puts the whole record on one line, and a `two_column` row puts two fields on one line, so the tail of those lines is silently dropped from the PDF while the label keeps the values. Example `doc_0009` (PDF, narrative): the source line is 181 characters and the rendered document stops after `value is 86859.76.`, so `currency` and `governing_law` are labelled but absent. The same document rendered as DOCX has no truncation, so two documents with the same label score differently depending on their file format. Affected dev documents: doc_0001, doc_0003, doc_0008, doc_0009, doc_0015, doc_0029, doc_0032, doc_0051, doc_0052, doc_0055, doc_0062, doc_0064, doc_0068, doc_0073, doc_0081, doc_0092, doc_0097.

Fix-versus-document tradeoff:

- *Fix now*: render the full line (wrap instead of truncate) and regenerate the dev split. That changes every document, invalidates the frozen rules baseline `baseline-v1`, and invalidates all 87 cached model outputs (40 free, 47 paid), so the 100-document run has to restart over two to three days of free-tier quota.
- *Document*: keep the data as is, report the gap and report accuracy on the values that are actually present in the body.

Decision: **document (Option B)**, the same decision as the `parties` gap of 2026-10-19. The frozen baseline stays valid and the partial run is not thrown away. Consequence for the report: per-field F1 over all labelled values is a lower bound, and the honest extractor quality is the accuracy on the values that are present in the body - 228/229 (0.996) for the free model and 285/286 (0.997) for `openai/gpt-4o-mini` on the evaluated subset. The fix is a one-line change in the generator and is scheduled before any test or challenge evaluation.

## Known data limitation: narrative layout drops list-valued fields (2026-10-20)

7 contracts (4 dev, 3 test) have a generator gap where `parties` is labeled but not rendered in the document text. The `narrative` layout renders only scalar fields, so no "Parties: X and Y" text appears in the document body even though the label JSON lists the parties.

Consequence: `parties` F1 is a lower bound. A correct extractor cannot recover a value that is absent from the document. The LLM extractor handles this by leaving `parties` empty and emitting the issue `missing_required_field:parties` with `needs_human_review: true`, rather than inventing a value (covered by `tests/test_missing_required_field.py`).

Decision: Option B - no data regeneration, so the frozen rules baseline (`baseline-v1`) and all recorded comparisons stay valid. The limitation is documented here and in `docs/requirements.md`.

## New error type: OpenRouter 402, credit and in-flight budget exhausted (2026-10-20)

The 100-doc dev run against `openai/gpt-4o-mini` stopped after 47 documents. The OpenRouter account hit its credit limit: `GET /api/v1/credits` reported `total_credits: 0` with `total_usage: $0.19656653`. Two distinct 402 messages were seen:

- `This request would exceed your available credits given your current in-flight requests` - recoverable. The extractor's retry loop issues the request again and it succeeds. 24 documents needed exactly one such retry; none of them needed a schema-validation retry.
- `This request requires more credits, or fewer max_tokens` - not recoverable. All three attempts fail and the document falls back to the rules baseline.

Behaviour verified: no crash, no partial cache entry, and the rules fallback produced a valid result for every affected document. The run resumes from the cache, so only the documents without a cached output are paid for again.

## Failure mode: `table` layout documents have no document-type cue (2026-10-20)

The `table` layout renders only `ITEM | QTY | PRICE | AMOUNT` rows plus `TOTAL:`. The words "invoice" and "purchase order" never appear, so the model cannot tell the document type. Result on the evaluated subset:

- doc-type accuracy on the `table` layout is 3/6, against 10/10, 13/13, 7/7, 5/5 and 5/5 for the other five layouts.
- When the model answers `business_document` the schema keeps only a free-text `description`, so every field of that document is lost even though the body contains the values.
- The model copies the header names of the table into `line_items` (`price` instead of `unit_price`), which the strict `LineItem` schema rejects. Three documents (doc_0030, doc_0038, doc_0048) show this.

Root cause is the generator, not the prompt: the table layout drops the header block that carries the type and the scalar fields.

## Injection guardrail measures the wrong thing (2026-10-20)

The guardrail fires `possible_prompt_injection` when the model answers `business_document` for a non-empty body. It therefore detects *unclassifiable* documents, not injected text. On the evaluated subset: TP=0, FP=2, FN=4 - it flagged 2 documents with no injected text and missed 4 documents that contain `IGNORE PREVIOUS INSTRUCTIONS`. The label already exists (`meta.injection`), so a text-based check on the document body is the correct implementation; the current heuristic should not be presented as an injection detector.

## Cache integrity (2026-10-20)

One cache entry (doc_0015) was written with a trailing byte and could not be parsed. A corrupt entry used to raise out of `cached()` and abort the whole run; it is now treated as a cache miss, and cache writes are atomic (temp file plus replace) so a killed process cannot leave a half-written entry.

## Free model constraints on OpenRouter (2026-10-20)

- `google/gemini-flash-1.5` is retired. `GET /api/v1/models/google/gemini-flash-1.5/endpoints` returns 0 providers, so the slug cannot be called at all. The run uses `google/gemma-4-31b-it:free` (Google AI Studio, price 0, one provider) instead.
- Free models are limited to **50 requests per day per key** (`GET /api/v1/auth/key` -> `free_model_daily_requests.limit = 50`). A 100-document run needs roughly 100 requests, so it spans at least two days. Failed requests are not counted against the limit, but every successful document consumes one.
- The upstream free pool is rate limited independently of the key limit: `429 ... temporarily rate-limited upstream` from Google AI Studio. The extractor's retry loop recovers most of them; on 2026-10-20 it recovered 9 of 40 documents on the second attempt.
- Free models answer less consistently than gpt-4o-mini: answers arrive wrapped in ```json fences or with prose around the JSON (handled by `parse_llm_json`), sometimes with an empty content field, and the accuracy on the evaluated subset is slightly lower than gpt-4o-mini on the same fields.
