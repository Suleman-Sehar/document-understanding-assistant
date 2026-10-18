import json, pathlib, sys
sys.path.insert(0,"src")
from dua.baseline import extract, read_document

root=pathlib.Path("data/dev")
labels=list((root/"labels").glob("*.json"))
tp=0; total=0
field_tp=collections.defaultdict(int) if False else {}
import collections
ft=collections.defaultdict(int); fp=collections.defaultdict(int); fn=collections.defaultdict(int)
layout_stats=collections.defaultdict(lambda: {"correct":0,"total":0})
for lp in labels:
    d=json.loads(lp.read_text())
    docpath=root/"docs"/d["file"]
    text=read_document(docpath)
    pred=extract(text)
    total+=1
    if pred["doc_type"]==d["doc_type"]:
        tp+=1
    layout=d["meta"].get("layout") or "unknown"
    layout_stats[layout]["total"]+=1
    if pred["doc_type"]==d["doc_type"]:
        layout_stats[layout]["correct"]+=1
    # per field
    for key,val in d["fields"].items():
        pval=pred["fields"].get(key)
        if val is None and pval is None:
            # true negative not counted in F1? Count as TP for empty? We'll skip
            continue
        if val==pval and val is not None:
            ft[key]+=1
        if val is not None and pval is None:
            fn[key]+=1
        if pval is not None and val != pval:
            fp[key]+=1
        if val is not None and pval is not None and val!=pval:
            fp[key]+=1  # already
print("doc_type_accuracy", tp/total)
print("\nper-field F1:")
for key in sorted(set().union(*[set(json.loads(p.read_text())["fields"].keys()) for p in labels])):
    tpv=ft[key]; fpv=fp[key]; fnv=fn[key]
    prec=tpv/(tpv+fpv) if tpv+fpv else 0
    rec=tpv/(tpv+fnv) if tpv+fnv else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    print(f"{key}: F1={f1:.3f} (tp={tpv}, fp={fpv}, fn={fnv})")
print("\nper-layout doc_type accuracy:")
for k,v in sorted(layout_stats.items()):
    print(k, v["correct"], "/", v["total"], round(v["correct"]/v["total"],3) if v["total"] else 0)
