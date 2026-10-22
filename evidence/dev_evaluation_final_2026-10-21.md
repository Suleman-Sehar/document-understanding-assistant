# Dev Evaluation Final (2026-10-21)
Note: F1 figures are lower bounds. 130 labelled values (18.1% of dev) are absent from document text due to generator gaps (97 layout omissions + 33 PDF truncation). Accuracy on values present in body is reported separately per model.

Dev split evaluated on 99/100 documents across two models (not blended). `doc_0002` excluded (mixed-model cache entry). Test and challenge not run.

## Model 1: google/gemma-4-31b-it:free (48 documents)

| field | baseline F1 | LLM F1 | delta | support |
|---|---|---|---|---|
| buyer_name | 0.667 | 0.933 | +0.266 | 16 |
| currency | 0.657 | 0.831 | +0.174 | 45 |
| delivery_date | 0.667 | 0.769 | +0.102 | 16 |
| due_date | 0.846 | 0.929 | +0.083 | 15 |
| effective_date | 0.727 | 1.000 | +0.273 | 14 |
| governing_law | 0.353 | 0.667 | +0.314 | 14 |
| invoice_date | 0.846 | 0.929 | +0.083 | 15 |
| invoice_number | 0.846 | 0.929 | +0.083 | 15 |
| issue_date | 0.667 | 0.897 | +0.230 | 16 |
| line_items | 0.229 | 0.808 | +0.579 | 31 |
| parties | 0.250 | 0.923 | +0.673 | 14 |
| po_number | 0.667 | 0.933 | +0.266 | 16 |
| po_reference | 0.000 | 0.000 | +0.000 | 0 |
| subtotal | 0.571 | 0.846 | +0.275 | 15 |
| tax | 0.846 | 0.929 | +0.083 | 15 |
| term_months | 0.923 | 1.000 | +0.077 | 14 |
| termination_clause | 0.000 | 0.000 | +0.000 | 0 |
| total | 0.708 | 0.967 | +0.259 | 31 |
| value | 0.133 | 0.783 | +0.650 | 14 |
| vendor_name | 0.760 | 0.893 | +0.133 | 31 |

- Doc-type accuracy: baseline 0.827, LLM 0.808
- Schema-valid rate: 48/48 (1.000)
- Accuracy on labelled values present in body: 274/275 (0.996)

## Model 2: openai/gpt-4o-mini (51 documents)

| field | baseline F1 | LLM F1 | delta | support |
|---|---|---|---|---|
| buyer_name | 0.714 | 0.909 | +0.195 | 18 |
| currency | 0.667 | 0.835 | +0.168 | 46 |
| delivery_date | 0.714 | 0.714 | +0.000 | 18 |
| due_date | 0.741 | 0.903 | +0.162 | 17 |
| effective_date | 0.842 | 1.000 | +0.158 | 11 |
| governing_law | 0.429 | 0.625 | +0.196 | 11 |
| invoice_date | 0.741 | 0.903 | +0.162 | 17 |
| invoice_number | 0.741 | 0.903 | +0.162 | 17 |
| issue_date | 0.714 | 0.839 | +0.125 | 18 |
| line_items | 0.293 | 0.889 | +0.596 | 35 |
| parties | 0.308 | 0.952 | +0.644 | 11 |
| po_number | 0.714 | 0.909 | +0.195 | 18 |
| po_reference | 0.000 | 0.000 | +0.000 | 0 |
| subtotal | 0.583 | 0.938 | +0.355 | 17 |
| tax | 0.741 | 0.938 | +0.197 | 17 |
| term_months | 0.952 | 1.000 | +0.048 | 11 |
| termination_clause | 0.000 | 0.000 | +0.000 | 0 |
| total | 0.704 | 0.971 | +0.267 | 35 |
| value | 0.167 | 0.706 | +0.539 | 11 |
| vendor_name | 0.727 | 0.871 | +0.144 | 35 |

- Doc-type accuracy: baseline 0.863, LLM 0.863
- Schema-valid rate: 47/47 (1.000)
- Accuracy on labelled values present in body: 285/286 (0.997)

## Combined summary (weighted average across models, not blended)

| field | weighted avg LLM F1 | free support (48) | paid support (51) | total support |
|---|---|---|---|---|
| buyer_name | 0.920 | 16 | 18 | 34 |
| currency | 0.833 | 45 | 46 | 91 |
| delivery_date | 0.740 | 16 | 18 | 34 |
| due_date | 0.915 | 15 | 17 | 32 |
| effective_date | 1.000 | 14 | 11 | 25 |
| governing_law | 0.649 | 14 | 11 | 25 |
| invoice_date | 0.915 | 15 | 17 | 32 |
| invoice_number | 0.915 | 15 | 17 | 32 |
| issue_date | 0.866 | 16 | 18 | 34 |
| line_items | 0.851 | 31 | 35 | 66 |
| parties | 0.936 | 14 | 11 | 25 |
| po_number | 0.920 | 16 | 18 | 34 |
| subtotal | 0.895 | 15 | 17 | 32 |
| tax | 0.934 | 15 | 17 | 32 |
| term_months | 1.000 | 14 | 11 | 25 |
| total | 0.969 | 31 | 35 | 66 |
| value | 0.749 | 14 | 11 | 25 |
| vendor_name | 0.881 | 31 | 35 | 66 |

- Weighted doc-type accuracy: 0.836 (48 free + 51 paid)
- Schema-valid: 48/48 + 51/51 = 99/99 (1.000)
