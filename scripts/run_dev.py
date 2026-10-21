"""CLI runner for the LLM extractor over a data split.

The model is always passed explicitly and provider-qualified: OpenRouter
auto-routes an unprefixed slug to whichever provider is cheapest/fastest,
which would silently change the model under evaluation.

  .venv\\Scripts\\python.exe scripts\\run_dev.py --split dev --limit 100

A daily request budget keeps a free-model run inside the provider's quota. The
provider decrements its counter only for requests that reach the model, so the
budget reads the quota before every document and stops the run as soon as no
request is left. The stop is clean: no request is attempted, no error is
raised, and every document that was answered stays cached.
"""
import sys, json, argparse, pathlib, collections, time, os, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import (llm_extract, resolve_model, resolve_base_url,
                                DEFAULT_MODEL, DEFAULT_BASE_URL, RequestBudgetExhausted)

QUOTA_URL = "https://openrouter.ai/api/v1/auth/key"


class DailyRequestBudget:
    """Caps a run at the provider's daily request limit for the account.

    remaining: the provider's own counter (None when it cannot be read)
    served:    documents answered by the model in this run
    attempts:  HTTP requests issued in this run (a rate-limited attempt does not
               consume quota but does consume an attempt)
    """

    def __init__(self, max_requests, max_attempts, api_key, quota_url=QUOTA_URL, check=True):
        self.max_requests = max_requests
        self.max_attempts = max_attempts
        self.api_key = api_key
        self.quota_url = quota_url
        self.check = check
        self.served = 0
        self.attempts = 0
        self.remaining = None
        self.quota_reads = 0
        self.quota_errors = 0

    def refresh(self):
        """Read the provider's remaining quota. A failed read is fail-open."""
        if not self.check or not self.api_key:
            return self.remaining
        try:
            req = urllib.request.Request(self.quota_url,
                                         headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            data = data.get("data", data)
            free = data.get("free_model_daily_requests") or {}
            rem = free.get("remaining")
            self.remaining = int(rem) if rem is not None else None
            self.quota_reads += 1
        except Exception:
            self.quota_errors += 1
        return self.remaining

    def exhausted(self):
        if self.served >= self.max_requests:
            return True
        if self.remaining is not None and self.remaining <= 0:
            return True
        return False

    def __call__(self):
        """Called before every request attempt; False refuses the request."""
        if self.attempts >= self.max_attempts:
            return False
        self.attempts += 1
        return True


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
    ap.add_argument("--max-requests", type=int, default=None,
                    help="hard stop after this many documents answered by the model")
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="hard stop after this many HTTP request attempts (a rate-limited attempt "
                         "does not consume daily quota but does consume an attempt)")
    ap.add_argument("--no-quota-check", action="store_true",
                    help="do not read the provider quota before the run")
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="stop cleanly after this many seconds of wall clock")
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

    api_key = os.getenv("OPENAI_API_KEY")
    budget = DailyRequestBudget(max_requests=(args.max_requests if args.max_requests is not None else 10**9),
                                 max_attempts=args.max_attempts,
                                 api_key=api_key,
                                 check=not args.no_quota_check)

    print(f"split={args.split} docs={len(labels)}")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"OPENAI_API_KEY={'set' if api_key else 'NOT SET'}")
    if args.max_requests:
        print(f"request budget: {args.max_requests} requests (max {args.max_attempts} attempts)")
    if not args.no_quota_check and api_key:
        rem = budget.refresh()
        print(f"provider quota remaining: {rem if rem is not None else 'unknown'}")

    cost = 0.0
    tokens = 0
    latencies = []
    api_calls = 0
    cache_hits = 0
    fallbacks = 0
    skipped = 0
    schema_valid = 0
    served = collections.Counter()
    completed = []
    stopped_reason = "all documents processed"
    started = time.time()

    for i, lp in enumerate(labels, 1):
        if budget.exhausted():
            stopped_reason = ("daily request budget reached"
                              if (budget.served >= budget.max_requests
                                  or (budget.remaining is not None and budget.remaining <= 0))
                              else "attempt budget reached")
            print(f"stopping before {lp.stem}: {stopped_reason}")
            break
        if args.max_seconds and (time.time() - started) >= args.max_seconds:
            stopped_reason = "wall-clock budget reached"
            print(f"stopping before {lp.stem}: {stopped_reason}")
            break
        if not args.no_quota_check and api_key:
            rem = budget.refresh()
            if rem is not None and rem <= 0:
                stopped_reason = "provider quota exhausted"
                print(f"stopping before {lp.stem}: provider quota exhausted")
                break
        d = json.loads(lp.read_text())
        try:
            out = llm_extract(root / "docs" / d["file"], model=model, base_url=base_url, budget=budget)
        except RequestBudgetExhausted:
            stopped_reason = "request budget reached"
            print(f"stopping at {d['doc_id']}: {stopped_reason}")
            break
        if out.get("_cache_hit"):
            cache_hits += 1
        elif out.get("_served_model"):
            api_calls += 1
            budget.served += 1
            completed.append(d["doc_id"])
        else:
            # empty document (no API call) or a rules fallback after failed attempts
            skipped += 1
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
        "skipped_or_fallback": skipped,
        "schema_valid": schema_valid,
        "rules_fallbacks": fallbacks,
        "total_cost_usd": round(cost, 6),
        "total_tokens": tokens,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "wall_clock_s": round(time.time() - started, 1),
        "stopped_reason": stopped_reason,
        "documents_completed": completed,
        "documents_completed_count": len(completed),
        "request_attempts": budget.attempts,
        "quota_remaining_at_end": budget.remaining,
        "quota_reads": budget.quota_reads,
        "quota_read_errors": budget.quota_errors,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        p = pathlib.Path(args.out)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
