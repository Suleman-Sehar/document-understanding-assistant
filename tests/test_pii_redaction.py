import re, pathlib, sys
sys.path.insert(0,"src")
from dua.llm import redact_pii

def test_no_pii_in_llm_input():
    text = "Contact: john.doe@example.com, 555-123-4567, 4111 1111 1111 1111, SSN 123-45-6789, IP 192.168.1.1, key sk-abc123"
    out = redact_pii(text)
    for pat in [r"\S+@\S+", r"\d{3}-\d{2}-\d{4}", r"\d{4} \d{4} \d{4} \d{4}", r"\d{3}-\d{3}-\d{4}", r"\d+\.\d+\.\d+\.\d+", r"sk-[A-Za-z0-9]+"]:
        assert not re.search(pat, out), pat
