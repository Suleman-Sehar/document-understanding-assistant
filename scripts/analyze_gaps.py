"""Generator gap analysis: labelled fields whose value never reaches the document body.

A gap is a data problem, not an extractor problem: the label carries a value that the
rendered document does not contain, so no extractor can recover it.

  .venv\\Scripts\\python.exe scripts\\analyze_gaps.py --split dev --explain

--explain regenerates the intended documents with the manifest seed (the generator is
deterministic) and classifies every gap by cause:

  layout_omission  the layout template never writes the field (by design)
  pdf_truncation   the PDF renderer cuts every line at 110 characters, so the tail of a
                   long line (narrative layout, long two_column rows) is dropped
  unexplained      the intended text contains the value but the document does not
"""
import sys, json, pathlib, argparse, collections, re, random, tempfile, importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua.baseline import read_document

PDF_LINE_LIMIT = 110

MONEY_FIELDS = {"subtotal", "tax", "total", "value"}
DATE_FIELDS = {"invoice_date", "due_date", "issue_date", "effective_date", "delivery_date"}


def num_in_text(val, text):
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val).lower() in text.lower()
    cands = {f"{f:.2f}", f"{f:.1f}", f"{f:g}", f"{f:,.2f}"}
    if f == int(f):
        cands.add(str(int(f)))
    return any(c in text for c in cands)


def value_in_text(field, value, text):
    if value is None:
        return None
    low = text.lower()
    if field == "line_items":
        if not isinstance(value, list):
            return str(value).lower() in low
        hits = []
        for it in value:
            if isinstance(it, dict):
                desc = str(it.get("description", ""))
                amt = it.get("amount")
                hits.append(desc.lower() in low and (amt is None or num_in_text(amt, text)))
            else:
                hits.append(str(it).lower() in low)
        return all(hits) if hits else None
    if field == "parties":
        return all(str(p).lower() in low for p in value) if isinstance(value, list) else str(value).lower() in low
    if field in DATE_FIELDS:
        return str(value).strip().lower() in low
    if field in MONEY_FIELDS:
        return num_in_text(value, text)
    return str(value).strip().lower() in low


def regenerate(split, count):
    """Rebuild the intended documents in a temp dir with the manifest seed.

    The renderers are patched to record the source text of every document, because the
    rendered file is what the extractor sees and the source text is what the label
    describes.
    """
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_synthetic.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gen_gap_"))
    gen.DATA = tmp
    source = {}

    def capture(factory):
        def wrapper(text, path):
            source[pathlib.Path(path).stem] = text
            return factory(text, path)
        return wrapper

    gen.render_pdf = capture(gen.render_pdf)
    gen.render_docx = capture(gen.render_docx)
    rng = random.Random(gen.SEED)
    for i in range(1, count + 1):
        gen.gen_one(rng, i, split, hard=False)
    return tmp, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--explain", action="store_true",
                    help="classify each gap by cause using a regenerated copy of the documents")
    args = ap.parse_args()

    root = ROOT / "data" / args.split
    labels = sorted((root / "labels").glob("*.json"))
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    count = manifest["counts"].get(args.split, len(labels))

    intended = {}
    if args.explain:
        tmp, source = regenerate(args.split, count)
        for lp in labels:
            d = json.loads(lp.read_text())
            intended[d["doc_id"]] = source[d["doc_id"]]

    gaps = []
    present = 0
    for lp in labels:
        d = json.loads(lp.read_text())
        text = read_document(root / "docs" / d["file"])
        layout = d["meta"].get("layout") or "unknown"
        is_pdf = d["file"].endswith(".pdf")
        for field, value in d["fields"].items():
            if value is None:
                continue
            if value_in_text(field, value, text) is False:
                cause = "unexplained"
                if args.explain:
                    itext = intended[d["doc_id"]]
                    if value_in_text(field, value, itext) is False:
                        cause = "layout_omission"
                    elif is_pdf and any(len(l) > PDF_LINE_LIMIT for l in itext.split("\n")):
                        cause = "pdf_truncation"
                gaps.append({"doc_id": d["doc_id"], "field": field, "layout": layout,
                             "doc_type": d["doc_type"], "cause": cause})
            else:
                present += 1

    by_layout_field = collections.Counter((g["layout"], g["field"]) for g in gaps)
    by_field = collections.Counter(g["field"] for g in gaps)
    by_layout = collections.Counter(g["layout"] for g in gaps)
    by_doc = collections.Counter(g["doc_id"] for g in gaps)

    lines = []
    lines.append(f"GENERATOR GAP ANALYSIS - {args.split} split ({len(labels)} documents)")
    lines.append("A gap is a labelled field whose value never appears in the rendered document body.")
    lines.append(f"labelled values present in the body : {present}")
    lines.append(f"labelled values absent from the body : {len(gaps)}")
    lines.append("")
    lines.append("by layout:")
    for lay, n in by_layout.most_common():
        lines.append(f"  {lay:12s} {n}")
    lines.append("by field:")
    for f, n in by_field.most_common():
        lines.append(f"  {f:18s} {n}")
    lines.append("by layout and field:")
    for (lay, f), n in sorted(by_layout_field.items()):
        lines.append(f"  {lay:12s} {f:18s} {n}")
    lines.append("documents with gaps:")
    for doc, n in by_doc.most_common():
        fields = sorted({g["field"] for g in gaps if g["doc_id"] == doc})
        lines.append(f"  {doc} {n:2d}  {', '.join(fields)}")

    if args.explain:
        by_cause = collections.Counter(g["cause"] for g in gaps)
        lines.append("")
        lines.append("by cause:")
        for cause, n in by_cause.most_common():
            lines.append(f"  {cause:16s} {n}")
        lines.append("")
        lines.append("cause by layout:")
        cl = collections.Counter((g["cause"], g["layout"]) for g in gaps)
        for (cause, lay), n in sorted(cl.items()):
            lines.append(f"  {cause:16s} {lay:12s} {n}")
        lines.append("")
        lines.append("cause by field:")
        cf = collections.Counter((g["cause"], g["field"]) for g in gaps)
        for (cause, f), n in sorted(cf.items()):
            lines.append(f"  {cause:16s} {f:18s} {n}")
        trunc_docs = sorted({g["doc_id"] for g in gaps if g["cause"] == "pdf_truncation"})
        lines.append("")
        lines.append(f"documents losing values to PDF line truncation ({len(trunc_docs)}):")
        for doc in trunc_docs:
            fields = sorted(g["field"] for g in gaps if g["doc_id"] == doc and g["cause"] == "pdf_truncation")
            lines.append(f"  {doc}  {', '.join(fields)}")
        unexp = [g for g in gaps if g["cause"] == "unexplained"]
        lines.append("")
        lines.append(f"unexplained gaps ({len(unexp)}):")
        for g in unexp:
            lines.append(f"  {g['doc_id']}  {g['field']}  layout={g['layout']}")

    out = "\n".join(lines)
    print(out)
    if args.out:
        p = pathlib.Path(args.out)
        p.write_text(out + "\n", encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
