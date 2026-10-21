import re
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any

ALLOWED_DOC_TYPES = ["invoice", "purchase_order", "short_contract", "business_document"]

class DocResult(BaseModel):
    doc_type: str
    fields: dict
    summary: str = ""
    confidence: float = 0.5
    needs_human_review: bool = True
    issues: List[str] = Field(default_factory=list)

    @field_validator("doc_type")
    @classmethod
    def valid_doc_type(cls, v):
        if v not in ALLOWED_DOC_TYPES:
            raise ValueError(f"doc_type must be one of {ALLOWED_DOC_TYPES}, got {v!r}")
        return v

# ---- per doc type field models: required = core fields, rest Optional with None default ----

class InvoiceFields(BaseModel):
    vendor_name: str
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    line_items: List[Any] = Field(default_factory=list)
    po_reference: Optional[str] = None

class PurchaseOrderFields(BaseModel):
    buyer_name: str
    po_number: Optional[str] = None
    issue_date: Optional[str] = None
    vendor_name: Optional[str] = None
    currency: Optional[str] = None
    total: Optional[float] = None
    line_items: List[Any] = Field(default_factory=list)
    delivery_date: Optional[str] = None

class ShortContractFields(BaseModel):
    parties: List[Any]
    effective_date: Optional[str] = None
    term_months: Optional[int] = None
    value: Optional[float] = None
    currency: Optional[str] = None
    termination_clause: Optional[str] = None
    governing_law: Optional[str] = None

class BusinessDocumentFields(BaseModel):
    description: Optional[str] = None

FIELD_MODELS = {
    "invoice": InvoiceFields,
    "purchase_order": PurchaseOrderFields,
    "short_contract": ShortContractFields,
    "business_document": BusinessDocumentFields,
}

def relaxed_model(doc_type):
    """Return a model where every field is Optional (used when a required core field
    is genuinely absent from the document; the document is flagged for review instead
    of the whole LLM output being discarded)."""
    from pydantic import create_model
    base=FIELD_MODELS[doc_type]
    fields={}
    for name,finfo in base.model_fields.items():
        ann=Optional[finfo.annotation] if not str(finfo.annotation).startswith("typing.List") else List[Any]
        fields[name]=(ann, None)
    return create_model(f"Relaxed{base.__name__}", **fields)

DOC_TYPE_ALIASES = {    "contract": "short_contract",
    "agreement": "short_contract",
    "purchase order": "purchase_order",
    "purchaseorder": "purchase_order",
    "po": "purchase_order",
    "bill": "invoice",
    "receipt": "invoice",
    "business record": "business_document",
    "business doc": "business_document",
    "other": "business_document",
    "unknown": "business_document",
}

def normalize_doc_type(doc_type):
    """Map a model answer to one of the allowed doc types, without content inference.

    Returns None when the answer cannot be mapped. Used to keep the injection
    guardrail tied to what the model actually answered, before any repair."""
    if not doc_type:
        return None
    dt = str(doc_type).strip().lower().replace("-", "_").replace(" ", "_")
    if dt in ALLOWED_DOC_TYPES:
        return dt
    if dt in DOC_TYPE_ALIASES:
        return DOC_TYPE_ALIASES[dt]
    for alias, target in DOC_TYPE_ALIASES.items():
        if alias.replace(" ", "_") == dt:
            return target
    return None

def has_line_items_table(text):
    """True when the body holds a line-items table: a header row naming quantities and
    prices, or rows of 'name | qty | price | amount'.

    Such a document is an invoice or a purchase order even when the body carries no
    other type cue, so it must never end up as business_document."""
    if not text:
        return False
    lines=[l for l in str(text).splitlines() if l.strip()]
    for l in lines:
        if (re.search(r"\b(qty|quantity)\b", l, re.I)
                and re.search(r"\b(price|unit price)\b", l, re.I)
                and re.search(r"\b(amount|total)\b", l, re.I)):
            return True
    rows=0
    for l in lines:
        parts=[p.strip() for p in l.split("|")]
        if (len(parts)>=4 and re.fullmatch(r"\d+", parts[1])
                and re.fullmatch(r"[\d.,]+", parts[2]) and re.fullmatch(r"[\d.,]+", parts[3])):
            rows+=1
    return rows>=2

def repair_doc_type(doc_type, text):
    """Fix a wrong/unknown doc_type when the document content makes it fixable."""
    if not doc_type:
        return None
    dt = normalize_doc_type(doc_type)
    if dt is not None:
        if dt == "business_document":
            # generic label, but content may clearly be one of our types
            t = text.lower()
            if "purchase order" in t or "po number" in t or "po no" in t:
                return "purchase_order"
            if "contract" in t or "agreement" in t or "parties" in t:
                return "short_contract"
            if "invoice" in t:
                return "invoice"
            # a line-items table with quantities and prices is an invoice or a purchase
            # order; without any other cue the body cannot tell which one, so the
            # majority class of the dev split (purchase_order) is used
            if has_line_items_table(text):
                return "purchase_order"
        return dt
    # last resort: infer from content
    t = text.lower()
    if "purchase order" in t or "po number" in t:
        return "purchase_order"
    if "contract" in t or "agreement" in t:
        return "short_contract"
    if "invoice" in t:
        return "invoice"
    if has_line_items_table(text):
        return "purchase_order"
    return None
