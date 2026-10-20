import sys, pathlib, json
sys.path.insert(0, "src")
from unittest.mock import patch, MagicMock
import dua.llm_extractor as LX


def test_default_model_is_provider_qualified():
    # the default may change with the model under evaluation, but it must always
    # carry the provider prefix: an unprefixed slug is auto-routed by OpenRouter
    assert "/" in LX.DEFAULT_MODEL
    assert "/" in LX.resolve_model()
    assert LX.resolve_model() == LX.DEFAULT_MODEL


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
    assert LX.cache_key("some text") == LX.cache_key("some text", model=LX.DEFAULT_MODEL)


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
        out = LX.llm_extract(doc, client=client, base_url="https://openrouter.ai/api/v1",
                             model="openai/gpt-4o-mini")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert out["_model"] == "openai/gpt-4o-mini"
    assert out["_served_model"] == "openai/gpt-4o-mini-2024-07-18"


def test_client_is_built_with_openrouter_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    doc = tmp_path / "doc_route2.txt"
    doc.write_text("Invoice Number: 1\nVendor: Acme")
    obj = {"doc_type": "invoice", "fields": {"vendor_name": "Acme", "invoice_number": "1"},
           "summary": "s", "confidence": 0.9, "needs_human_review": False, "issues": []}
    with patch("dua.llm_extractor.OpenAI") as M:
        client = M.return_value
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))],
            model=LX.DEFAULT_MODEL,
            usage=MagicMock(prompt_tokens=10, completion_tokens=5))
        LX.llm_extract(doc)
    assert M.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert M.call_args.kwargs["api_key"] == "test-key"
    assert client.chat.completions.create.call_args.kwargs["model"] == LX.DEFAULT_MODEL


def test_free_model_costs_nothing_and_paid_model_is_priced():
    assert LX.estimate_cost("google/gemma-4-31b-it:free", 1000, 1000) == 0.0
    assert LX.estimate_cost("openrouter/free", 1000, 1000) == 0.0
    assert LX.estimate_cost("openai/gpt-4o-mini", 1000, 1000) > 0
    # a model with no known price is reported as unpriced, never as free
    assert LX.estimate_cost("google/gemini-2.5-flash", 1000, 1000) is None


def test_fenced_json_is_parsed():
    assert LX.parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert LX.parse_llm_json('here you go: {"a": 2} thanks') == {"a": 2}
    assert LX.parse_llm_json('{"a": 3}') == {"a": 3}


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
