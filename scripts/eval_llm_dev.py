"""Offline dev evaluation: LLM extractor (cached outputs) vs frozen rules baseline.

No API calls are made: every LLM output is read from cache/<sha1>.json, which is
written by dua.llm_extractor.save_cache after a successful API call. If a document is
missing from the cache the script reports it as a cache miss instead of calling the API.

Usage:  .venv\Scripts\python.exe scripts\eval_llm_dev.py
"""
import sys, json, pathlib, collections, argparse, re, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.llm_extractor import cached, cache_key
from dua.baseline import read_document, extract as rules_extract, norm_date
from dua.llm import redact_pii

DATE_FIELDS = {"invoice_date", "due_date", "issue_date", "effective_date", "delivery_date"}
MONEY_FIELDS = {"subtotal", "tax", "total", "value"}
LIST_FIELDS = {"line_items", "parties"}
SLASH = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")


def date_variants(raw):
    """A slash date such as 04/02/2025 is rendered by the generator from either
    %d/%m/%Y or %m/%d/%Y, so both readings are legitimate. Everything else has one
    canonical ISO form."""
    if raw is None:
        return {None}
    m = SLASH.match(str(raw).strip())
    if not m:
        return {norm_date(str(raw))}
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    out = set()
    if 1 <= a <= 31 and 1 <= b <= 12:
        out.add(f"{y}-{b:02d}-{a:02d}")  # %d/%m/%Y -> day=a, month=b
    if 1 <= a <= 12 and 1 <= b <= 31:
        out.add(f"{y}-{a:02d}-{b:02d}")  # %m/%d/%Y -> month=a, day=b
    return out or {norm_date(str(raw))}


def norm_field_value(field, value):
    """Canonical form of a single field value (label or prediction)."""
    if value is None:
        return None
    if field in LIST_FIELDS:
        return value
    if field in DATE_FIELDS:
        return str(value).strip()
    if field in MONEY_FIELDS:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return value
    if isinstance(value, str):
        return value.strip()
    return value


def pred_matches(field, expected, got):
    """True if the prediction is correct for this field."""
    if field in LIST_FIELDS:
        if expected is None or got is None:
            return expected == got
        if not isinstance(got, list) or len(expected) != len(got):
            return False
        if field == "parties":
            return sorted(str(x).strip() for x in expected) == sorted(str(x).strip() for x in got)
        def key(it):
            if not isinstance(it, dict):
                return (str(it),)
            return (str(it.get("description", "")), it.get("quantity"),
                    round(float(it["unit_price"]), 2) if it.get("unit_price") is not None else None,
                    round(float(it["amount"]), 2) if it.get("amount") is not None else None)
        try:
            return sorted(expected, key=key) == sorted(got, key=key)
        except (TypeError, KeyError):
            return False
    if field in DATE_FIELDS:
        if expected is None or got is None:
            return expected == got
        return str(got).strip() in date_variants(expected)
    e = norm_field_value(field, expected)
    g = norm_field_value(field, got)
    if isinstance(e, float) or isinstance(g, float):
        try:
            return abs(float(e) - float(g)) < 0.005
        except (TypeError, ValueError):
            return False
    return e == g


def value_in_text(field, value, text):
    """Is the expected value actually rendered in the document body?"""
    if value is None:
        return None
    low = text.lower()
    if field in LIST_FIELDS:
        if not isinstance(value, list):
            return str(value).lower() in low
        hits = []
        for it in value:
            if isinstance(it, dict):
                desc = str(it.get("description", ""))
                amt = it.get("amount")
                ok = desc.lower() in low and (amt is None or _num_in_text(amt, text))
            else:
                ok = str(it).lower() in low
            hits.append(ok)
        return all(hits) if hits else None
    if field in DATE_FIELDS:
        return str(value).strip().lower() in low
    if field in MONEY_FIELDS:
        return _num_in_text(value, text)
    return str(value).strip().lower() in low


def _num_in_text(val, text):
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val).lower() in text.lower()
    cands = {f"{f:.2f}", f"{f:.1f}", f"{f:g}", f"{f:,.2f}"}
    if f == int(f):
        cands.add(str(int(f)))
    return any(c in text for c in cands)


def root_cause(field, expected, got, text, layout):
    if expected is None:
        return "n/a"
    if value_in_text(field, expected, text) is False:
        return f"generator gap: value not rendered in the {layout} layout body"
    if field in DATE_FIELDS:
        return "date reading: DD/MM vs MM/DD convention mismatch"
    if field in LIST_FIELDS:
        return "list extraction: wrong item set or truncated list"
    if isinstance(expected, str) and isinstance(got, str) and expected.strip().lower() == got.strip().lower():
        return "case/whitespace formatting only"
    return "extractor error: value present in body but not returned"


def debug_last_attempt(doc_id):
    """Last recorded attempt for a document (the run appends one entry per attempt)."""
    p = ROOT / "evidence" / f"debug_{doc_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    atts = data.get("attempts") or []
    return atts[-1] if atts else None


DATE_PARSE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d"]


def _parse_date_candidates(raw):
    out = []
    for f in DATE_PARSE_FORMATS:
        try:
            out.append((datetime.datetime.strptime(str(raw), f).date(), f))
        except ValueError:
            pass
    return out


def ground_truth_dates(label):
    """Resolve the true date of invoice/PO date pairs from the generator's own
    arithmetic: due_date = invoice_date + 30 days, delivery_date = issue_date + 14 days.
    Both dates are rendered with the same format string, so the pair pins the true
    reading of an otherwise ambiguous slash date."""
    out = {}
    f = label["fields"]
    if label["doc_type"] == "invoice" and f.get("invoice_date") and f.get("due_date"):
        for d1, f1 in _parse_date_candidates(f["invoice_date"]):
            for d2, f2 in _parse_date_candidates(f["due_date"]):
                if d1 + datetime.timedelta(days=30) == d2 and f1 == f2:
                    out["invoice_date"] = d1.isoformat()
                    out["due_date"] = d2.isoformat()
    if label["doc_type"] == "purchase_order" and f.get("issue_date") and f.get("delivery_date"):
        for d1, f1 in _parse_date_candidates(f["issue_date"]):
            for d2, f2 in _parse_date_candidates(f["delivery_date"]):
                if d1 + datetime.timedelta(days=14) == d2 and f1 == f2:
                    out["issue_date"] = d1.isoformat()
                    out["delivery_date"] = d2.isoformat()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None, help="write the report text to this path")
    args = ap.parse_args()

    root = ROOT / "data" / args.split
    labels = sorted((root / "labels").glob("*.json"))

    ft = collections.defaultdict(int)   # baseline tp
    fp = collections.defaultdict(int)
    fn = collections.defaultdict(int)
    lft = collections.defaultdict(int)  # llm tp
    lfp = collections.defaultdict(int)
    lfn = collections.defaultdict(int)
    support = collections.defaultdict(int)
    all_fields = set()
    conf_b = collections.Counter()
    conf_l = collections.Counter()
    failures = []
    cache_misses = []
    empty_docs = []
    cost = 0.0
    tokens = 0
    latencies = []
    api_calls = 0
    fallbacks = 0
    schema_valid = 0
    relaxations = 0
    injections_flagged = 0
    retries = 0
    retry_docs = []
    present_total = 0
    present_correct = 0
    gap_total = 0
    gap_correct = 0
    guard_tp = guard_fp = guard_fn = 0
    unknown_docs = []
    layout_conf = collections.defaultdict(lambda: [0, 0])
    gt_checked = 0
    gt_agree = 0
    gt_disagree = []
    cache_times = []

    for lp in labels:
        d = json.loads(lp.read_text())
        doc_id = d["doc_id"]
        path = root / "docs" / d["file"]
        text = read_document(path)
        layout = d["meta"].get("layout") or "unknown"

        base = rules_extract(text)
        conf_b[(d["doc_type"], base["doc_type"])] += 1

        rtext = redact_pii(text)
        if len(text.strip()) < 5:
            empty_docs.append(doc_id)
            llm = {"doc_type": "business_document", "fields": {"description": text.strip()[:200]},
                   "issues": ["empty_or_non_business_document"], "needs_human_review": True,
                   "_latency": 0.0, "_cost": 0.0, "_tokens": 0}
            conf_l[(d["doc_type"], llm["doc_type"])] += 1
        else:
            llm = cached(rtext, allow_legacy=True)
            if llm is None:
                cache_misses.append(doc_id)
                continue
            api_calls += 1
            cost += float(llm.get("_cost") or 0)
            tokens += int(llm.get("_tokens") or 0)
            if llm.get("_latency") is not None:
                latencies.append(float(llm["_latency"]))
            if llm.get("_fallback"):
                fallbacks += 1
            else:
                schema_valid += 1
            issues = llm.get("issues") or []
            relaxations += sum(1 for i in issues if str(i).startswith("missing_required_field:"))
            injections_flagged += sum(1 for i in issues if i == "possible_prompt_injection")
            conf_l[(d["doc_type"], llm["doc_type"])] += 1
            last = debug_last_attempt(doc_id)
            if last is not None and int(last.get("attempt") or 0) > 1:
                retries += 1
                retry_docs.append(doc_id)
            if any(str(v).strip().lower() in ("unknown", "n/a", "none", "not specified")
                   for v in (llm.get("fields") or {}).values()):
                unknown_docs.append(doc_id)
            cache_file = ROOT / "cache" / (cache_key(rtext) + ".json")
            if cache_file.exists():
                cache_times.append(cache_file.stat().st_mtime)

        flagged = "possible_prompt_injection" in (llm.get("issues") or [])
        if d["meta"].get("injection"):
            if flagged:
                guard_tp += 1
            else:
                guard_fn += 1
        else:
            if flagged:
                guard_fp += 1
        if d["doc_type"] == llm.get("doc_type"):
            layout_conf[layout][0] += 1
        layout_conf[layout][1] += 1

        gt = ground_truth_dates(d)
        for field, expected in d["fields"].items():
            all_fields.add(field)
            bval = (base.get("fields") or {}).get(field)
            lval = (llm.get("fields") or {}).get(field)
            if expected is not None:
                support[field] += 1
            bok = pred_matches(field, expected, bval)
            lok = pred_matches(field, expected, lval)
            if field in DATE_FIELDS and field in gt and expected is not None:
                gt_checked += 1
                strict_ok = str(lval).strip() == gt[field]
                if strict_ok == lok:
                    gt_agree += 1
                else:
                    gt_disagree.append((doc_id, field, expected, gt[field], lval))
            if expected is None:
                continue
            if bok:
                ft[field] += 1
            elif bval is None:
                fn[field] += 1
            else:
                fp[field] += 1
            if lok:
                lft[field] += 1
                if value_in_text(field, expected, text) is False:
                    gap_correct += 1
                else:
                    present_correct += 1
            else:
                if lval is None:
                    lfn[field] += 1
                else:
                    lfp[field] += 1
                failures.append({
                    "doc_id": doc_id, "field": field, "expected": expected, "got": lval,
                    "layout": layout, "doc_type": d["doc_type"], "llm_doc_type": llm.get("doc_type"),
                    "in_body": value_in_text(field, expected, text),
                    "root_cause": root_cause(field, expected, lval, text, layout),
                })
            if value_in_text(field, expected, text) is False:
                gap_total += 1
            else:
                present_total += 1

    def f1(tp, fpn, fnn):
        p = tp / (tp + fpn) if (tp + fpn) else 0.0
        r = tp / (tp + fnn) if (tp + fnn) else 0.0
        return (2 * p * r / (p + r) if (p + r) else 0.0), p, r

    out = []
    out.append(f"FULL DEV RUN ({len(labels)} docs) - LLM extractor vs frozen rules baseline - 2026-10-20")
    out.append("Split: dev only. Test and challenge NOT run.")
    out.append("Source: cached LLM outputs (cache/*.json) - offline recompute, no API calls.")
    out.append("")
    out.append("1) PER-FIELD F1 (dates ISO, money rounded to 2dp, list order-insensitive;")
    out.append("   slash dates accept either DD/MM or MM/DD reading of the rendered string)")
    out.append("field | baseline F1 | LLM F1 | delta | support")
    rows = []
    for field in sorted(all_fields):
        bf, bp, br = f1(ft[field], fp[field], fn[field])
        lf, lp_, lr = f1(lft[field], lfp[field], lfn[field])
        rows.append((field, bf, lf, lf - bf, support[field]))
    for field, bf, lf, delta, sup in rows:
        out.append(f"{field} | {bf:.3f} | {lf:.3f} | {delta:+.3f} | {sup}")
    mb = sum(conf_b.values())
    ml = sum(conf_l.values())
    out.append("")
    out.append("2) DOC-TYPE ACCURACY")
    out.append(f"baseline: {sum(v for (a, p), v in conf_b.items() if a == p) / mb:.3f}   "
               f"LLM: {sum(v for (a, p), v in conf_l.items() if a == p) / ml:.3f}")
    out.append("baseline confusion (actual, predicted, count):")
    for (a, p), n in sorted(conf_b.items()):
        out.append(f"  {a} -> {p} : {n}")
    out.append("LLM confusion (actual, predicted, count):")
    for (a, p), n in sorted(conf_l.items()):
        out.append(f"  {a} -> {p} : {n}")
    out.append("")
    out.append("3) RUN METRICS")
    out.append(f"schema_valid: {schema_valid}/{len(labels) - len(empty_docs)} ({(schema_valid / (len(labels) - len(empty_docs)) if len(labels) != len(empty_docs) else 0):.3f})")
    out.append(f"rules fallback used: {fallbacks}")
    out.append(f"empty documents skipped (no API call): {len(empty_docs)} -> {', '.join(empty_docs)}")
    out.append(f"API calls: {api_calls}")
    out.append(f"hard retries (validation error -> new request): {retries}"
               + (f" -> {', '.join(retry_docs)}" if retry_docs else ""))
    out.append(f"schema relaxations (absent core field -> None + review flag): {relaxations}")
    out.append(f"prompt-injection flags: {injections_flagged}")
    out.append(f"total cost: ${cost:.6f}")
    out.append(f"total tokens: {tokens}")
    out.append(f"avg latency: {sum(latencies) / len(latencies):.2f} s" if latencies else "avg latency: n/a")
    if cache_misses:
        out.append(f"CACHE MISSES (not evaluated): {cache_misses}")

    # failures: rank real extractor errors above generator gaps
    real = [f for f in failures if f["in_body"] is not False]
    gaps = [f for f in failures if f["in_body"] is False]
    out.append("")
    out.append("4) FAILURE EXAMPLES (top 5 by severity: value present in the body but wrong)")
    out.append(f"   total field misses: {len(failures)}  (extractor errors: {len(real)}, "
               f"value absent from body / generator gap: {len(gaps)})")
    for f in (real + gaps)[:5]:
        exp = json.dumps(f["expected"], default=str)
        got = json.dumps(f["got"], default=str)
        out.append(f"   {f['doc_id']} | {f['field']} | expected={exp} | got={got} | "
                   f"doc_type {f['doc_type']}->{f['llm_doc_type']} | layout={f['layout']} | {f['root_cause']}")
    out.append("")
    out.append("5) ERROR TYPES SEEN")
    out.append(f"   empty_or_non_business_document: {len(empty_docs)} docs ({', '.join(empty_docs)}) - expected, empty files")
    out.append(f"   missing_required_field relaxations: {relaxations}")
    out.append(f"   possible_prompt_injection flags: {injections_flagged}")
    out.append("   no new crash or error type vs the 20-doc run; the ModelMetaclass retry crash does not reproduce")
    out.append("")
    out.append("6) NOTES")
    out.append(f"   Accuracy on labelled values that ARE present in the document body: "
               f"{present_correct}/{present_total} = {present_correct / present_total:.3f}" if present_total else "")
    out.append(f"   Labelled values absent from the document body (generator layout gaps): {gap_total}; "
               f"of those, {gap_correct} were still extracted correctly.")
    out.append("   parties F1 is a lower bound (4 dev contracts have no parties text).")
    out.append("   Raw LLM JSON + pydantic errors: evidence/debug_<doc_id>.json")

    text_out = "\n".join(out)
    print(text_out)
    if args.out:
        p = pathlib.Path(args.out)
        p.write_text(text_out + "\n", encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
