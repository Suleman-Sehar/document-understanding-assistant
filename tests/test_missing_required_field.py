import json, sys, pathlib
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
from dua.llm_extractor import llm_extract

# doc_0009 (dev, layout=narrative) has NO "parties" text in the document body.
# The label JSON lists parties, but the narrative layout drops list-valued fields,
# so the value is genuinely absent from the document. The extractor must therefore
# NOT invent a value: it must leave the field empty and flag the document for review.

def test_absent_parties_is_flagged_not_invented(tmp_path):
    text=("This document records the following. effective_date is 23 October 2024. "
          "term_months is 24. value is 86859.76. \n--- PAGE 2 ---\nAdditional terms.")
    doc=tmp_path/"doc_noparties.txt"; doc.write_text(text)
    obj={"doc_type":"short_contract","fields":{"effective_date":"2024-10-23","term_months":24,"value":86859.76},
         "summary":"s","confidence":0.9,"needs_human_review":False,"issues":[]}
    with patch("dua.llm_extractor.OpenAI") as M:
        client=M.return_value
        client.chat.completions.create.return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))])
        out=llm_extract(doc, client=client, base_url="x")
    assert out["fields"]["parties"] == []
    assert "missing_required_field:parties" in out["issues"]
    assert out["needs_human_review"] is True

def test_dev_contract_without_parties_text_is_confirmed_absent():
    root=pathlib.Path("data/dev")
    from dua.baseline import read_document
    for lp in sorted((root/"labels").glob("*.json")):
        d=json.loads(lp.read_text())
        if d["doc_type"]!="short_contract" or not d["fields"].get("parties"):
            continue
        t=read_document(root/"docs"/d["file"]).lower()
        if "parties" not in t:
            # value not present in document text -> must not be extractable
            assert d["meta"].get("layout") in ("narrative",) or d["meta"].get("layout") is None
            return
    raise AssertionError("expected at least one contract whose parties are absent from the text")
