import json, pathlib, sys, shutil
sys.path.insert(0,"src")
from unittest.mock import patch, MagicMock
from dua.llm_extractor import llm_extract

def test_raw_output_is_saved_on_validation_failure(tmp_path):
    shutil.rmtree("cache",ignore_errors=True)
    p=pathlib.Path("evidence/debug_doc_9999.json")
    p.unlink(missing_ok=True)
    doc=tmp_path/"doc_9999.txt"; doc.write_text("Invoice Number: 1")
    bad={"doc_type":"invoice","fields":{}}
    with patch("dua.llm_extractor.OpenAI") as M:
        client=M.return_value
        client.chat.completions.create.return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(bad)))])
        out=llm_extract(doc, client=client, base_url="x")
    assert p.exists()
    data=json.loads(p.read_text())
    assert any(a["raw_llm_output"] for a in data["attempts"])
    assert any(a["validation_error"] for a in data["attempts"])
    p.unlink()
