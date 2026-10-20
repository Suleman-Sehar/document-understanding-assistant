# Error Analysis

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
