import json, sys
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
from dua.llm_extractor import llm_extract
from dua.schemas import ALLOWED_DOC_TYPES, repair_doc_type

def test_allowed_doc_types():
    assert "business_document" in ALLOWED_DOC_TYPES
    assert set(ALLOWED_DOC_TYPES) == {"invoice","purchase_order","short_contract","business_document"}

def test_repair_business_document_to_contract():
    text="SHORT CONTRACT (AGREEMENT)\nParties: A, B\nEffective Date: 2024-01-01"
    assert repair_doc_type("business_document", text) == "short_contract"
    assert repair_doc_type("contract", text) == "short_contract"

def test_repair_applied_before_validation(tmp_path):
    doc=tmp_path/"doc_8888.txt"; doc.write_text("SHORT CONTRACT\nParties: A, B")
    obj={"doc_type":"business_document","fields":{"parties":["A","B"]},"summary":"","confidence":0.8,"needs_human_review":False,"issues":[]}
    with patch("dua.llm_extractor.OpenAI") as M:
        client=M.return_value
        client.chat.completions.create.return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))])
        out=llm_extract(doc, client=client, base_url="x")
    assert out["doc_type"] == "short_contract"
    import pathlib
