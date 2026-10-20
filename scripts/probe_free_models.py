"""Probe free OpenRouter models with ONE request each and report whether the answer
parses and validates against the extraction schema.

  .venv\\Scripts\\python.exe scripts\\probe_free_models.py

Free models are rate limited (50 requests/day on the free tier), so this sends a
single request per candidate and never retries.
"""
import sys, os, json, pathlib, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import SYSTEM, build_fewshot
from dua.baseline import read_document
from dua.llm import redact_pii
from dua.schemas import DocResult, FIELD_MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="data/dev/docs/doc_0001.pdf")
    ap.add_argument("--models", nargs="*", default=[
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-5-super-vl:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "thinkingmachines/inkling:free",
        "poolside/laguna-s-2.1:free",
        "liquid/lfm-2.5-2.6b:free",
        "dots-studio/dots-3-note-preview:free",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "cohere/north-mini-code:free",
    ])
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1")

    text = redact_pii(read_document(ROOT / args.doc))
    prompt = f"{build_fewshot()}\n\nNOW EXTRACT THIS DOCUMENT. Output JSON only.\nDOCUMENT:\n{text}"

    for model in args.models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0, max_tokens=1500)
        except Exception as e:
            print(f"{model:46s} REQUEST FAILED: {str(e)[:150]}")
            continue
        raw = resp.choices[0].message.content
        served = getattr(resp, "model", None)
        try:
            obj = json.loads(raw)
            validated = DocResult(**obj)
            fields = validated.fields
            fm = FIELD_MODELS[validated.doc_type]
            bad = []
            for k, v in fields.items():
                if k not in fm.model_fields:
                    continue
                if v in (None, "", []) and fm.model_fields[k].is_required():
                    bad.append(k)
            print(f"{model:46s} served={served} type={validated.doc_type} "
                  f"fields={len(fields)} missing_core={bad or 'none'}")
        except Exception as e:
            print(f"{model:46s} served={served} INVALID: {str(e)[:160]}")
            print(f"    raw: {str(raw)[:220]}")


if __name__ == "__main__":
    main()
