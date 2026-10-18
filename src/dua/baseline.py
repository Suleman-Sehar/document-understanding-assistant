import re, pathlib
from datetime import datetime

def read_document(path):
    path=pathlib.Path(path)
    if path.suffix.lower()==".pdf":
        from pypdf import PdfReader
        r=PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in r.pages)
    if path.suffix.lower()==".docx":
        from docx import Document
        d=Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    return path.read_text(errors="ignore")

DATE_FORMATS=["%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%B %d, %Y","%d %B %Y","%d %b %Y","%Y/%m/%d"]

def norm_date(s):
    if s is None: return None
    s=s.strip().rstrip(".")
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s,f).date().isoformat()
        except ValueError:
            pass
    return s

def norm_num(s):
    if s is None: return None
    s=s.strip().replace("$","").replace("USD","").replace("EUR","").replace("GBP","").strip()
    if "," in s and "." in s:
        if s.rfind(",")>s.rfind("."):
            s=s.replace(".","").replace(",",".")
        else:
            s=s.replace(",","")
    elif "," in s:
        s=s.replace(",",".")
    try:
        return float(s)
    except ValueError:
        return None

def classify(text):
    t=text.lower()
    if "purchase order" in t or "po number" in t or "po=>" in t or "po number" in t: return "purchase_order"
    if "contract" in t or "agreement" in t or "parties" in t: return "short_contract"
    if "invoice" in t: return "invoice"
    # fallback by fields
    if "buyer" in t: return "purchase_order"
    return "invoice"

FIELD_KEYS={
 "invoice":["vendor_name","invoice_number","invoice_date","due_date","currency","subtotal","tax","total","po_reference"],
 "purchase_order":["buyer_name","po_number","issue_date","vendor_name","currency","total","delivery_date"],
 "short_contract":["parties","effective_date","term_months","value","currency","termination_clause","governing_law"],
}

def parse_kv(text):
    d={}
    for line in text.splitlines():
        line=line.strip()
        if "=>" in line:
            k,v=line.split("=>",1)
        elif ":" in line:
            k,v=line.split(":",1)
        else:
            continue
        k=k.strip().lower().replace(" ","_")
        v=v.strip()
        if k and v:
            d[k]=v
    return d

def extract(text):
    dt=classify(text)
    kv=parse_kv(text)
    fields={}
    def get(*names):
        for n in names:
            for k,v in kv.items():
                if k==n or k.endswith(n) or n in k:
                    return v
        return None
    if dt=="invoice":
        fields["vendor_name"]=get("vendor_name","vendor")
        fields["invoice_number"]=get("invoice_number","invoice_no")
        fields["invoice_date"]=norm_date(get("invoice_date","date"))
        fields["due_date"]=norm_date(get("due_date"))
        fields["currency"]=get("currency")
        fields["subtotal"]=norm_num(get("subtotal"))
        fields["tax"]=norm_num(get("tax"))
        fields["total"]=norm_num(get("total","grand_total","amount_payable","total_due","amount"))
        fields["po_reference"]=get("po_reference","po")
        items=[]
        for m in re.finditer(r"(\w+)\s*[|x]\s*(\d+)\s*[@|]\s*([\d.,]+)\s*[=|]\s*([\d.,]+)",text):
            items.append({"description":m.group(1),"quantity":int(m.group(2)),"unit_price":norm_num(m.group(3)),"amount":norm_num(m.group(4))})
        fields["line_items"]=items
    elif dt=="purchase_order":
        fields["buyer_name"]=get("buyer_name","buyer")
        fields["po_number"]=get("po_number","po_no","po")
        fields["issue_date"]=norm_date(get("issue_date","date"))
        fields["vendor_name"]=get("vendor_name","vendor")
        fields["currency"]=get("currency")
        fields["total"]=norm_num(get("total"))
        fields["delivery_date"]=norm_date(get("delivery_date"))
        items=[]
        for m in re.finditer(r"(\w+)\s+qty\s+(\d+)\s+price\s+([\d.,]+)\s+amount\s+([\d.,]+)",text,re.I):
            items.append({"description":m.group(1),"quantity":int(m.group(2)),"unit_price":norm_num(m.group(3)),"amount":norm_num(m.group(4))})
        fields["line_items"]=items
    else:
        p=get("parties")
        if p:
            try:
                import ast
                val=ast.literal_eval(p)
                if isinstance(val,list):
                    fields["parties"]=[str(x).strip() for x in val]
                else:
                    fields["parties"]=[x.strip() for x in p.split(",") if x.strip()]
            except Exception:
                fields["parties"]=[x.strip().strip("[]'\"") for x in p.split(",") if x.strip()]
        else:
            fields["parties"]=[]
        fields["effective_date"]=norm_date(get("effective_date","effective"))
        tm=get("term_months","term")
        if tm:
            m=re.search(r"(\d+)",str(tm)); fields["term_months"]=int(m.group(1)) if m else None
        else:
            fields["term_months"]=None
        fields["value"]=norm_num(get("value"))
        fields["currency"]=get("currency")
        fields["termination_clause"]=get("termination_clause","termination")
        fields["governing_law"]=get("governing_law")
    return {"doc_type":dt,"fields":fields,"summary":"","confidence":0.5,"needs_human_review":False,"issues":[]}
