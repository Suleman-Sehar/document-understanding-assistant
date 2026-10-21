import json, pathlib, sys
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
import pytest
from dua.llm_extractor import llm_extract, RequestBudgetExhausted, cached, save_cache
from dua.baseline import read_document


def _client(obj):
    client=MagicMock()
    client.chat.completions.create.return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))])
    return client


def test_budget_refusal_stops_before_any_request(tmp_path):
    doc=tmp_path/"d.txt"; doc.write_text("Invoice Number: 1")
    obj={"doc_type":"invoice","fields":{"vendor_name":"Acme"},"summary":"","confidence":0.9,"needs_human_review":False,"issues":[]}
    client=_client(obj)
    with patch("dua.llm_extractor.OpenAI") as M:
        M.return_value=client
        with pytest.raises(RequestBudgetExhausted):
            llm_extract(doc, client=client, base_url="x", budget=lambda: False)
    assert client.chat.completions.create.call_count == 0


def test_budget_refusal_after_the_allowed_attempts(tmp_path):
    doc=tmp_path/"d.txt"; doc.write_text("Invoice Number: 1")
    obj={"doc_type":"invoice","fields":{"vendor_name":"Acme"},"summary":"","confidence":0.9,"needs_human_review":False,"issues":[]}
    client=_client(obj)
    calls=[]
    def budget():
        calls.append(1)
        return len(calls) <= 1
    with patch("dua.llm_extractor.OpenAI") as M:
        M.return_value=client
        llm_extract(doc, client=client, base_url="x", budget=budget)
    assert client.chat.completions.create.call_count == 1


def test_budget_exhausted_is_not_caught_by_the_retry_loop(tmp_path):
    """A budget stop must not be swallowed into a rules fallback."""
    assert not issubclass(RequestBudgetExhausted, Exception)


def test_cache_hit_is_marked(tmp_path, monkeypatch):
    import dua.llm_extractor as le
    monkeypatch.setattr(le, "CACHE_DIR", tmp_path)
    save_cache("some text", {"doc_type":"invoice","fields":{}}, model="test/model")
    hit=cached("some text", model="test/model")
    assert hit is not None
    assert hit.get("_cache_hit") is True
    assert cached("other text", model="test/model") is None
