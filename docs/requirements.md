# Requirements — Document Understanding Assistant

## Stakeholder
Accounts-Payable Lead (fictional company: Arrowstack Corp). Goal: automate extraction of structured fields from business documents (invoices, purchase orders, short contracts) to reduce manual data entry and flag risky/ambiguous documents for human review.

## Scope (in)
- Input: text-based PDF and DOCX files only.
- Doc types: invoice, purchase_order, short_contract.
- Output JSON with: doc_type, fields, summary, confidence, needs_human_review, issues[].
- Synthetic data only, fixed seed for reproducibility.
- LLM via OpenRouter (OPENAI_API_KEY + --base-url).

## Out of scope
- Scanned/image PDFs (OCR), handwritten docs, non-business documents, multi-page contracts >10 pages, email threads, automated payments/approvals, integration with ERP systems, real PII.

## Field schemas

### invoice
- vendor_name (str), invoice_number (str), invoice_date (date ISO), due_date (date ISO, optional), currency (str ISO-4217), subtotal (float), tax (float), total (float), line_items (list: {description, quantity, unit_price, amount}), po_reference (str, optional).

### purchase_order
- buyer_name (str), po_number (str), issue_date (date ISO), vendor_name (str), currency (str), total (float), line_items (list: {description, quantity, unit_price, amount}), delivery_date (date ISO, optional).

### short_contract
- parties (list str), effective_date (date ISO), term_months (int), value (float), currency (str), termination_clause (str, optional), governing_law (str, optional).

## Edge cases
- Missing required fields.
- Odd date formats (DD/MM/YYYY vs MM/DD/YYYY ambiguity).
- Non-USD currencies and mixed currency symbols.
- Garbled text / empty files / non-business files.
- Prompt injection inside document text.
- Fake PII (names, emails, phone numbers) — must not be leaked in logs/outputs.
- Arithmetic inconsistency (subtotal+tax != total).
- Duplicate/contradictory fields.

## Success metrics (targets, set before code, immutable unless logged)
- Doc-type accuracy: >= 0.95 on test set.
- Per-field F1 (micro): >= 0.90.
- Schema-valid JSON rate: 100% (all outputs validate against pydantic schema).
- Arithmetic consistency check pass rate: >= 0.98 on valid invoices/POs.
- Human-review trigger rate: between 10% and 30% on challenge set.
- Latency: median < 5s per doc (excluding LLM).

## Acceptance criteria
- CLI accepts file path, outputs valid JSON to stdout/file.
- All outputs pass pydantic validation.
- Tests pass (pytest) with coverage >= 80% on core.
- Reproducible with fixed seed; synthetic data generator provided.
- Docs in docs/ only; work_log and decision_log dated.

## Risks
- LLM hallucination on missing fields -> mitigated by confidence + review flag.
- Prompt injection -> mitigated by input sanitization and strict schema.
- Date/currency ambiguity -> mitigated by validation + review flag.
- Cost overruns from paid calls -> budget cap and approval gate.
- Synthetic data may not reflect real-world noise -> documented limitation.
- 7 contracts (4 dev, 3 test) have a generator gap where parties is labeled but not rendered in document text; parties F1 is a lower bound. See docs/error_analysis.md.

## Proposed folder tree
```
document-understanding-assistant/
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   ├── work_log.md
│   └── decision_log.md
├── src/
│   └── dua/
│       ├── __init__.py
│       ├── cli.py
│       ├── schemas.py
│       ├── extractors/
│       ├── validators/
│       ├── pipeline.py
│       └── utils/
├── tests/
│   ├── test_schemas.py
│   ├── test_extractors.py
│   ├── test_pipeline.py
│   └── test_edge_cases.py
├── data/
│   ├── synthetic/
│   └── challenge/
├── scripts/
│   └── generate_synthetic.py
├── pyproject.toml
├── README.md
└── .env.example
```
