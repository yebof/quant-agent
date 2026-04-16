import sys
from pathlib import Path

import pytest

# Add src to path so tests can import from src.*
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Every test runs in its own tmp cwd so stores with relative data_dir defaults
    (NewsStore/MacroStore → 'data/news', 'data/macro') don't write to the real repo
    during the test suite.

    Agent PROMPT_PATH values are computed from __file__ (absolute), so they're
    unaffected. Tests that need real data/config paths should use explicit absolute
    paths or dedicated fixtures.
    """
    monkeypatch.chdir(tmp_path)
