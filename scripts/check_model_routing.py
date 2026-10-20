"""Verify the LLM routing configuration without spending anything.

  .venv\\Scripts\\python.exe scripts\\check_model_routing.py

Prints the resolved model slug, the endpoint, whether the API key is present (never
its value), and the exact payload that would be sent to chat.completions.create.
With --probe it also queries OpenRouter's free /models endpoint (no inference, no
cost) to list the providers that serve the pinned slug.
"""
import sys, os, json, pathlib, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import resolve_model, resolve_base_url, DEFAULT_MODEL, DEFAULT_BASE_URL, SYSTEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="query OpenRouter /models (free, no inference)")
    args = ap.parse_args()

    model = resolve_model()
    base_url = resolve_base_url()
    key = os.getenv("OPENAI_API_KEY")

    print("model       :", model)
    print("base_url    :", base_url)
    print("OPENAI_API_KEY:", "set" if key else "NOT SET")
    print("default slug:", DEFAULT_MODEL, "| default endpoint:", DEFAULT_BASE_URL)
    if "/" not in model:
        print("WARNING: model slug is not provider-qualified - OpenRouter would auto-route it")

    from unittest.mock import MagicMock, patch
    import dua.llm_extractor as LX
    from dua.llm_extractor import llm_extract
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # keep the probe out of the repository cache: it must always exercise the
        # request path, never a cached response
        LX.CACHE_DIR = pathlib.Path(td) / "cache"
        LX.CACHE_DIR.mkdir()
        doc = pathlib.Path(td) / "doc_probe.txt"
        doc.write_text("Invoice Number: 1\nVendor: Acme\nTOTAL: 10.00")
        payload = {"doc_type": "invoice", "fields": {"vendor_name": "Acme", "invoice_number": "1"},
                   "summary": "s", "confidence": 0.9, "needs_human_review": False, "issues": []}
        with patch("dua.llm_extractor.OpenAI") as M:
            client = M.return_value
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))],
                model="openai/gpt-4o-mini-2024-07-18",
                usage=MagicMock(prompt_tokens=10, completion_tokens=5))
            # no client passed: the patched OpenAI constructor is exercised, so the
            # endpoint and key handling are shown exactly as they would be in a run
            llm_extract(doc)
        kwargs = client.chat.completions.create.call_args.kwargs
        print("\nrequest that would be sent (mocked, nothing left the machine):")
        print("  endpoint  :", M.call_args.kwargs["base_url"] if M.call_args else "n/a")
        print("  api_key   :", "set" if M.call_args and M.call_args.kwargs.get("api_key") else "NOT SET")
        print("  model     :", kwargs["model"])
        print("  temperature:", kwargs["temperature"], "| max_tokens:", kwargs["max_tokens"])
        print("  system prompt version chars:", len(SYSTEM))

    if args.probe:
        if not key:
            print("\n--probe skipped: OPENAI_API_KEY is not set")
            return
        import urllib.request
        req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print("\n--probe failed:", type(e).__name__, e)
            return
        print("\n--probe: OpenRouter /models (free, no inference)")
        for m in data.get("data", []):
            if m.get("id") in (model, "openai/gpt-4o-mini", "gpt-4o-mini"):
                endpoints = m.get("endpoints") or []
                print(f"  id={m.get('id')}  name={m.get('name')}")
                print(f"    provider: {m.get('id','').split('/')[0]}  endpoints: {len(endpoints)}")


if __name__ == "__main__":
    main()
