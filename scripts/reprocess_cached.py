"""Rebuild the cached dev outputs from the stored raw model answers (no API calls).

The raw answer of every document is kept in evidence/debug_<doc_id>.json. After a
repair or schema change the stored answers can be re-validated offline, so the
cached outputs and the evaluation reflect the new code without spending anything.

  .venv\\Scripts\\python.exe scripts\\reprocess_cached.py --model google/gemma-4-31b-it:free
"""
import sys, json, pathlib, argparse, collections

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import postprocess, parse_llm_json, cache_key, save_cache, resolve_model
from dua.baseline import read_document
from dua.llm import redact_pii


def raw_answers(doc_id, model):
    """Raw answers stored for a document, oldest first, for the given model."""
    p = ROOT / "evidence" / f"debug_{doc_id}.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    out = []
    for a in data.get("attempts") or []:
        raw = a.get("raw_llm_output")
        if not raw:
            continue
        if model and a.get("model") not in (None, model):
            continue
        out.append((a.get("attempt"), raw, a.get("served_model")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true", help="report the changes without writing")
    args = ap.parse_args()
    model = args.model or resolve_model()

    root = ROOT / "data" / args.split
    labels = sorted((root / "labels").glob("*.json"))
    changed = []
    unchanged = 0
    no_raw = []
    reprocessed = 0
    errors = collections.Counter()
    type_before = collections.Counter()
    type_after = collections.Counter()

    for lp in labels:
        d = json.loads(lp.read_text())
        doc_id = d["doc_id"]
        text = read_document(root / "docs" / d["file"])
        rtext = redact_pii(text)
        cpath = ROOT / "cache" / (cache_key(rtext, model) + ".json")
        if not cpath.exists():
            continue
        reprocessed += 1
        cached = json.loads(cpath.read_text())
        answers = raw_answers(doc_id, model)
        if not answers:
            no_raw.append(doc_id)
            continue
        attempt, raw, served = answers[-1]
        try:
            obj = parse_llm_json(raw)
            out = postprocess(obj, text, model=model, served_model=served or cached.get("_served_model"),
                              latency=cached.get("_latency", 0.0), cost=cached.get("_cost"),
                              tokens=cached.get("_tokens", 0))
        except Exception as e:
            errors[f"{type(e).__name__}: {str(e)[:60]}"] += 1
            continue
        type_before[cached.get("doc_type")] += 1
        type_after[out.get("doc_type")] += 1
        if cached.get("doc_type") != out.get("doc_type") or cached.get("fields") != out.get("fields"):
            changed.append((doc_id, cached.get("doc_type"), out.get("doc_type"),
                            cached.get("fields"), out.get("fields")))
        else:
            unchanged += 1
        if not args.dry_run:
            save_cache(rtext, out, model)

    print(f"model: {model}")
    print(f"cached documents reprocessed: {reprocessed}   unchanged: {unchanged}   changed: {len(changed)}")
    print(f"documents without a stored raw answer: {len(no_raw)} {no_raw}")
    if errors:
        print("errors:", dict(errors))
    print("doc_type before:", dict(type_before))
    print("doc_type after :", dict(type_after))
    for doc_id, before, after, bf, af in changed:
        print(f"  {doc_id}: {before} -> {after}")
        bkeys = {k: v for k, v in (bf or {}).items() if v not in (None, "", [])}
        akeys = {k: v for k, v in (af or {}).items() if v not in (None, "", [])}
        print(f"      fields before: {sorted(bkeys)}")
        print(f"      fields after : {sorted(akeys)}")
        for k in sorted(set(bkeys) | set(akeys)):
            if bkeys.get(k) != akeys.get(k):
                print(f"        {k}: {json.dumps(bkeys.get(k), default=str)[:60]} -> {json.dumps(akeys.get(k), default=str)[:60]}")
    if not args.dry_run:
        print("\ncache entries rewritten")


if __name__ == "__main__":
    main()
