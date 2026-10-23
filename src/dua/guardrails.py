import re
from typing import List, Dict, Any, Tuple


PII_PATTERNS = {
    'EMAIL': r'\S+@\S+',
    'PHONE': r'\b\d{3}-\d{3}-\d{4}\b',
    'CARD': r'\b\d{4}(?: \d{4}){3}\b',
    'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
    'IP': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'APIKEY': r'sk-[A-Za-z0-9]+',
}


def redact_pii(text: str) -> Tuple[str, Dict[str, List[str]]]:
    if text is None:
        return '', {}
    redacted = text
    found = {k: [] for k in PII_PATTERNS}
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            found[label] = list(matches)
        redacted = re.sub(pattern, f'[{label}]', redacted)
    return redacted, found


def check_injection(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    patterns = [
        r'ignore\s+all\s+previous\s+instructions',
        r'ignore\s+previous\s+instructions',
        r'system\s*:',
        r'you\s+are\s+now',
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def check_arithmetic(fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    subtotal = fields.get('subtotal')
    tax = fields.get('tax')
    total = fields.get('total')
    try:
        if subtotal is not None and tax is not None and total is not None:
            s = float(subtotal)
            tx = float(tax)
            tt = float(total)
            if abs((s + tx) - tt) > 0.01 + 1e-9:
                issues.append('arithmetic_inconsistency')
                return False, issues
    except Exception:
        pass
    return True, issues


def check_missing_core_fields(doc_type: str, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    needs_review = False
    if doc_type == 'invoice':
        if not fields.get('vendor_name'):
            issues.append('missing_required_field:vendor_name')
            needs_review = True
    elif doc_type == 'purchase_order':
        if not fields.get('buyer_name'):
            issues.append('missing_required_field:buyer_name')
            needs_review = True
    elif doc_type == 'short_contract':
        parties = fields.get('parties')
        if not parties or (isinstance(parties, list) and len(parties) == 0):
            issues.append('missing_required_field:parties')
            needs_review = True
    return needs_review, issues


def compute_confidence(result: Dict[str, Any]) -> Tuple[float, bool]:
    score = 0.0
    issues = result.get('issues', []) or []
    fields = result.get('fields', {}) or {}
    doc_type = result.get('doc_type')
    
    if doc_type in ['invoice', 'purchase_order', 'short_contract', 'business_document']:
        score += 0.3
    
    if len(issues) == 0:
        score += 0.3
    
    core_ok = True
    if doc_type == 'invoice':
        core_ok = bool(fields.get('vendor_name'))
    elif doc_type == 'purchase_order':
        core_ok = bool(fields.get('buyer_name'))
    elif doc_type == 'short_contract':
        parties = fields.get('parties')
        core_ok = bool(parties) and (not isinstance(parties, list) or len(parties) > 0)
    if core_ok and doc_type in ['invoice', 'purchase_order', 'short_contract']:
        score += 0.2
    elif doc_type == 'business_document':
        score += 0.2
    
    arith_ok, _ = check_arithmetic(fields)
    if arith_ok:
        score += 0.2
    
    needs_review = score < 0.6
    return round(score, 3), needs_review


def apply_guardrails(result: Dict[str, Any], text: str = '') -> Dict[str, Any]:
    out = dict(result)
    out['issues'] = list(out.get('issues') or [])
    out['fields'] = dict(out.get('fields') or {})
    # normalize free-text truncation/incomplete-data warnings to a standard label
    norm = []
    seen = set()
    for i in out['issues']:
        low = str(i).lower()
        if 'truncat' in low or 'incomplete line items' in low or 'incomplete data' in low or 'incomplete text' in low or 'truncated text' in low or 'truncated list' in low:
            if 'source_truncated' not in seen:
                norm.append('source_truncated')
                seen.add('source_truncated')
        else:
            if i not in seen:
                norm.append(i)
                seen.add(i)
    out['issues'] = norm
    
    arith_ok, arith_issues = check_arithmetic(out['fields'])
    for ai in arith_issues:
        if ai not in out['issues']:
            out['issues'].append(ai)
    
    needs_miss, miss_issues = check_missing_core_fields(out.get('doc_type'), out['fields'])
    for mi in miss_issues:
        if mi not in out['issues']:
            out['issues'].append(mi)
    if needs_miss:
        out['needs_human_review'] = True
    
    conf, needs_conf = compute_confidence(out)
    out['confidence'] = conf
    if needs_conf:
        out['needs_human_review'] = True
    
    return out


def build_review_queue_entry(doc_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    issues = result.get('issues', []) or []
    confidence = result.get('confidence', 0.0)
    if issues:
        reason = '; '.join(str(i) for i in issues)
    elif result.get('needs_human_review'):
        reason = 'needs_human_review'
    else:
        reason = 'ok'
    return {
        'doc_id': doc_id,
        'issues': issues,
        'confidence': confidence,
        'reason': reason,
    }
