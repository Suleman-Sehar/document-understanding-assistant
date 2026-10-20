# Error Analysis

## Known data limitation: narrative layout drops list-valued fields (2026-10-20)

7 contracts (4 dev, 3 test) have a generator gap where `parties` is labeled but not rendered in the document text. The `narrative` layout renders only scalar fields, so no "Parties: X and Y" text appears in the document body even though the label JSON lists the parties.

Consequence: `parties` F1 is a lower bound. A correct extractor cannot recover a value that is absent from the document. The LLM extractor handles this by leaving `parties` empty and emitting the issue `missing_required_field:parties` with `needs_human_review: true`, rather than inventing a value (covered by `tests/test_missing_required_field.py`).

Decision: Option B — no data regeneration, so the frozen rules baseline (`baseline-v1`) and all recorded comparisons stay valid. The limitation is documented here and in `docs/requirements.md`.
