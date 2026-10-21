import json, pathlib, sys, time
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
from dua.llm_extractor import llm_extract, _is_provider_error, _retry_backoff


def test_provider_errors_are_recognized():
    assert _is_provider_error(RuntimeError("Error code: 429 - temporarily rate-limited upstream")) is True
    assert _is_provider_error(RuntimeError("Error code: 503 - Provider returned error")) is True
    assert _is_provider_error(ValueError("no JSON object found in the model answer")) is False


def test_backoff_has_two_waits():
    waits = _retry_backoff()
    assert len(waits) >= 2
    assert waits[1] > waits[0]


def test_provider_error_waits_before_the_next_attempt(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setenv("DUA_RETRY_BACKOFF_S", "1,2")
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    doc = tmp_path / "d.txt"; doc.write_text("Invoice Number: 1")
    obj = {"doc_type": "invoice", "fields": {"vendor_name": "Acme"}, "summary": "", "confidence": 0.9,
           "needs_human_review": False, "issues": []}
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("Error code: 429 - temporarily rate-limited upstream"),
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))]),
    ]
    with patch("dua.llm_extractor.OpenAI") as M:
        M.return_value = client
        out = llm_extract(doc, client=client, base_url="x")
    assert out["fields"].get("vendor_name") == "Acme"
    assert slept == [1.0]


def test_schema_error_does_not_wait(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    doc = tmp_path / "d.txt"; doc.write_text("Invoice Number: 1")
    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("no JSON object found in the model answer")
    with patch("dua.llm_extractor.OpenAI") as M:
        M.return_value = client
        llm_extract(doc, client=client, base_url="x")
    assert slept == []
