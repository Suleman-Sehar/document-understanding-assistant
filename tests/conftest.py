import pathlib, sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def isolated_llm_state(tmp_path, monkeypatch):
    """Keep every test away from the repository cache/ and evidence/ directories.

    cache/ holds the paid LLM responses of the 100-doc dev run; deleting it from a
    test would destroy the run evidence and force a re-run. evidence/ holds the
    per-document debug JSON of that run. Both are redirected to a temp directory
    for the duration of each test.
    """
    import dua.llm_extractor as LX

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    debug_dir = tmp_path / "evidence"
    debug_dir.mkdir()
    monkeypatch.setattr(LX, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(LX, "DEBUG_DIR", debug_dir)
    yield
