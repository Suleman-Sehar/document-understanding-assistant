"""CLI runner for the LLM extractor over a data split.

The model is always passed explicitly and provider-qualified: OpenRouter
auto-routes an unprefixed slug to whichever provider is cheapest/fastest,
which would silently change the model under evaluation.

  .venv\\Scripts\\python.exe scripts\\run_dev.py --split dev --limit 100
"""
import sys, json, argparse, pathlib, collections, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import llm_extract, resolve_model, resolve_base_url, DEFAULT_MODEL, DEFAULT_BASE_URL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "challenge"])
    ap.add_argument("--allow-other-split", action="store_true",
                    help="permit a non-dev split (dev is the only split allowed during evaluation)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="provider-qualified model slug, e.g. openai/gpt-4o-mini")
    ap.add_argument("--base-url", default=None,
                    help=f"OpenAI-compatible endpoint (default: OPENAI_BASE_URL or {DEFAULT_BASE_URL})")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N documents")
    ap.add_argument("--out", default=None, help="write a JSON run summary to this path")
    args = ap.parse_args()

    if args.split != "dev" and not args.allow_other_split:
        raise SystemExit(f"refusing to run the '{args.split}' split: only dev may be evaluated "
                         "(use --allow-other-split to override)")

    model = resolve_model(args.model)
    base_url = resolve_base_url(args.base_url)
    if "/" not in model:
        raise SystemExit(f"model '{model}' is not provider-qualified: OpenRouter needs the "
                         "provider prefix (e.g. openai/gpt-4o-mini)")

    root = ROOT / "data" / args.split
    labels = sorted((root / "labels").glob("*.json"))
    if args.limit:
        labels = labels[: args.limit]

    print(f"split={args.split} docs={len(labels)}")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"OPENAI_API_KEY={'set' if __import__('os').getenv('OPENAI_API_KEY') else 'NOT SET'}")

    cost = 0.0
    tokens = 0
    latencies = []
    api_calls = 0
    cache_hits = 0
    fallbacks = 0
    schema_valid = 0
    served = collections.Counter()
    started = time.time()

    for i, lp in enumerate(labels, 1):
        d = json.loads(lp.read_text())
        out = llm_extract(root / "docs" / d["file"], model=model, base_url=base_url)
        if out.get("_legacy_cache"):
            cache_hits += 1
        elif out.get("_tokens"):
            api_calls += 1
        if out.get("_fallback"):
            fallbacks += 1
        else:
            schema_valid += 1
        cost += float(out.get("_cost") or 0)
        tokens += int(out.get("_tokens") or 0)
        if out.get("_latency"):
            latencies.append(float(out["_latency"]))
        served[out.get("_served_model") or "not-recorded"] += 1
        if i % 10 == 0 or i == len(labels):
            print(f"  [{i}/{len(labels)}] cost=${cost:.6f} tokens={tokens} "
                  f"avg_latency={(sum(latencies)/len(latencies)) if latencies else 0:.2f}s")

    summary = {
        "split": args.split,
        "docs": len(labels),
        "model": model,
        "base_url": base_url,
        "served_models": dict(served),
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "schema_valid": schema_valid,
        "rules_fallbacks": fallbacks,
        "total_cost_usd": round(cost, 6),
        "total_tokens": tokens,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "wall_clock_s": round(time.time() - started, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        p = pathlib.Path(args.out)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
