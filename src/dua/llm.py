import re

def redact_pii(text):
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    text = re.sub(r"\b\d{4}(?: \d{4}){3}\b", "[CARD]", text)
    text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", text)
    text = re.sub(r"sk-[A-Za-z0-9]+", "[APIKEY]", text)
    return text
