import sys
sys.path.insert(0,"src")
from dua.llm_extractor import SYSTEM, normalize_line_items
from dua.schemas import has_line_items_table, repair_doc_type

TABLE_TEXT = """ITEM | QTY | PRICE | AMOUNT
Widget | 2 | 10.00 | 20.00
Gadget | 1 | 5.50 | 5.50
TOTAL: 25.50
"""

HEADER_TABLE_TEXT = """ITEM QTY PRICE AMOUNT
Widget 2 10.00 20.00
Gadget 1 5.50 5.50
"""


def test_prompt_forbids_business_document_for_line_item_table():
    assert "never business_document" in SYSTEM
    assert "line-items table" in SYSTEM


def test_prompt_requires_canonical_line_item_keys():
    assert "description, quantity, unit_price, amount" in SYSTEM
    assert 'never "price"' in SYSTEM


def test_has_line_items_table_detects_pipe_rows():
    assert has_line_items_table(TABLE_TEXT) is True


def test_has_line_items_table_detects_header_row():
    assert has_line_items_table(HEADER_TABLE_TEXT) is True


def test_has_line_items_table_false_for_plain_text():
    assert has_line_items_table("This document records the following. value is 12.00.") is False
    assert has_line_items_table("") is False


def test_repair_doc_type_uses_table_majority_class():
    assert repair_doc_type("business_document", TABLE_TEXT) == "purchase_order"


def test_repair_doc_type_keeps_business_document_without_table():
    text="This document records the following. value is 12.00."
    assert repair_doc_type("business_document", text) == "business_document"


def test_repair_doc_type_prefers_explicit_cue_over_table():
    text="INVOICE\n" + TABLE_TEXT
    assert repair_doc_type("business_document", text) == "invoice"


def test_normalize_line_items_renames_aliases():
    fields={"line_items":[{"description":"Widget","qty":2,"price":10.0,"line_total":20.0}]}
    out=normalize_line_items(fields)
    item=out["line_items"][0]
    assert item["quantity"]==2
    assert item["unit_price"]==10.0
    assert item["amount"]==20.0
    assert "price" not in item
    assert "qty" not in item


def test_normalize_line_items_canonical_key_wins_over_alias():
    fields={"line_items":[{"description":"Widget","price":9.9,"unit_price":10.0}]}
    item=normalize_line_items(fields)["line_items"][0]
    assert item["unit_price"]==10.0
    assert "price" not in item


def test_normalize_line_items_ignores_non_list():
    fields={"line_items":"not a list"}
    assert normalize_line_items(fields)["line_items"]=="not a list"
