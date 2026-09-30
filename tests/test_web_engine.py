"""The browser inference engine (web/sutra.js) must reproduce the Python reference outputs."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.skipif(not (WEB / "assets" / "demo.json").exists(), reason="run web/export.py first")
def test_js_engine_matches_python_reference():
    r = subprocess.run(["node", str(WEB / "test_engine.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["maxCnn"] < 1e-3 and res["maxGpt"] < 1e-3 and res["greedyMatches"]
