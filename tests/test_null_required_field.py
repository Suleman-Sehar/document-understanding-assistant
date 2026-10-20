import json, sys, shutil, pathlib
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
from dua.llm_extractor import llm_extract

def test_null_required_field_triggers_retry_and_relaxes(tmp_path):
    """LLM returns vendor_name: null. Attempt 1 fails validation, attempt 2 must not
    crash (the model name must not be shadowed by the pydantic field model)."""
    shutil.rmtree("cache",ignore_errors=True)
    doc=tmp_path/"doc_nullfield.txt"; doc.write_text("Vendor: Acme\nInvoice Number: 1\nTotal: 10.00")
    bad={"doc_type":"invoice","fields":{"vendor_name":None,"invoice_number":"1"},
         "summary":"","confidence":0.8,"needs_human_review":False,"issues":[]}
    good={"doc_type":"invoice","fields":{"vendor_name":None,"invoice_number":"1","total":10.0},
          "summary":"","confidence":0.8,"needs_human_review":False,"issues":[]}
    with patch("dua.llm_extractor.OpenAI") as M:
        client=M.return_value
        client.chat.completions.create.side_effect=[
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(bad)))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(good)))]),
        ]
        out=llm_extract(doc, client=client, base_url="x")
    assert out.get("_fallback") is not True
    assert out["doc_type"]=="invoice"
    # vendor_name was null in the LLM output; it is back-filled from the frozen rules
    # baseline when the document actually contains it, never invented
    assert out["fields"]["vendor_name"]=="Acme"
    pathlib.Path("evidence/debug_doc_nullfield.json").unlink(missing_ok=True)
