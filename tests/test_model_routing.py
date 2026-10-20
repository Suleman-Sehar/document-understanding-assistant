import sys, pathlib, json
sys.path.insert(0, "src")
from unittest.mock import patch, MagicMock
import dua.llm_extractor as LX


def test_default_model_is_provider_qualified():
    assert LX.DEFAULT_MODEL == "openai/gpt-4o-mini"
    assert "/" in LX.resolve_model()
    assert LX.resolve_model() == "openai/gpt-4o-mini"


def test_base_url_defaults_to_openrouter(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert LX.resolve_base_url() == "https://openrouter.ai/api/v1"


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    assert LX.resolve_model() == "openai/gpt-4o-mini"


def test_cache_key_depends_on_model():
    a = LX.cache_key("some text", model="openai/gpt-4o-mini")
    b = LX.cache_key("some text", model="openai/gpt-4o-mini-2024-07-18")
    assert a != b
    assert LX.cache_key("some text") == a


def test_api_call_uses_pinned_model_and_records_served_model(tmp_path):
    doc = tmp_path / "doc_route.txt"
    doc.write_text("Invoice Number: 1\nVendor: Acme")
    obj = {"doc_type": "invoice", "fields": {"vendor_name": "Acme", "invoice_number": "1"},
           "summary": "s", "confidence": 0.9, "needs_human_review": False, "issues": []}
    with patch("dua.llm_extractor.OpenAI") as M:
        client = M.return_value
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))],
            model="openai/gpt-4o-mini-2024-07-18",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5))
        out = LX.llm_extract(doc, client=client, base_url="https://openrouter.ai/api/v1")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert out["_model"] == "openai/gpt-4o-mini"
    assert out["_served_model"] == "openai/gpt-4o-mini-2024-07-18"


def test_client_is_built_with_openrouter_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    doc = tmp_path / "doc_route2.txt"
    doc.write_text("Invoice Number: 1\nVendor: Acme")
    obj = {"doc_type": "invoice", "fields": {"vendor_name": "Acme", "invoice_number": "1"},
           "summary": "s", "confidence": 0.9, "needs_human_review": False, "issues": []}
    with patch("dua.llm_extractor.OpenAI") as M:
        client = M.return_value
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))],
            model="openai/gpt-4o-mini",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5))
        LX.llm_extract(doc)
    assert M.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert M.call_args.kwargs["api_key"] == "test-key"
    assert client.chat.completions.create.call_args.kwargs["model"] == "openai/gpt-4o-mini"


def test_legacy_cache_entry_is_not_used_for_a_new_model(tmp_path, monkeypatch):
    text = "legacy only"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(LX, "CACHE_DIR", cache_dir)
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / f"{LX.legacy_cache_key(text)}.json").write_text(json.dumps({"fields": {}}))
    assert LX.cached(text, model="openai/gpt-4o-mini") is None
    assert LX.cached(text, model="openai/gpt-4o-mini", allow_legacy=True) is not None


def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    text = "corrupt entry"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(LX, "CACHE_DIR", cache_dir)
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / f"{LX.cache_key(text, 'openai/gpt-4o-mini')}.json").write_text('{"fields": {}}}')
    assert LX.cached(text, model="openai/gpt-4o-mini") is None


def test_cache_is_written_under_the_requested_model_key(tmp_path, monkeypatch):
    text = "model keyed write"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(LX, "CACHE_DIR", cache_dir)
    cache_dir.mkdir(exist_ok=True)
    LX.save_cache(text, {"fields": {"a": 1}}, model="openai/gpt-4o-mini")
    assert (cache_dir / f"{LX.cache_key(text, 'openai/gpt-4o-mini')}.json").exists()
    assert LX.cached(text, model="openai/gpt-4o-mini")["fields"] == {"a": 1}
