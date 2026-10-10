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


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("arch", ["gpt2", "llama", "attention-only"])
def test_js_engine_runs_every_architecture(arch, tmp_path):
    """Export a small random model of each architecture and require sutra.js to reproduce Python's logits."""
    import sys

    import numpy as np

    sys.path.insert(0, str(WEB))
    import export

    import sutra as st
    from sutra.models import GPT, GPTConfig
    st.set_default_dtype(np.float32)
    kw = {"gpt2": {}, "llama": dict(norm="rms", pos="rope", mlp="swiglu", bias=False), "attention-only": dict(mlp="none")}[arch]
    st.manual_seed(0)
    cfg = GPTConfig(vocab_size=23, block_size=16, n_layer=2, n_head=2, n_embd=32, **kw)
    model = GPT(cfg)
    for p in model.parameters():  # larger weights than the default init, so the check is sensitive
        if p.ndim == 2:
            p.data = p.data * 10
    deq = export.write_bundle("m", model.state_dict(), {"config": cfg.to_dict(), "tokenizer": {"type": "char", "chars": "abcdefghijklmnopqrstuvw"}}, tmp_path)
    model.load_state_dict(deq)  # the browser runs the int8 weights; so must the reference
    model.eval()
    prompt = [3, 1, 4, 1, 5, 9]
    logits, _ = model(np.array([prompt]))
    greedy = model.generate(np.array([prompt]), 12, temperature=0)[0, len(prompt):].tolist()  # crosses block_size
    (tmp_path / "m.golden.json").write_text(json.dumps({"prompt_ids": prompt, "step_logits": logits.data[0].tolist(),
                                                        "greedy_ids": greedy}))
    r = subprocess.run(["node", str(WEB / "check_gpt.js"), str(tmp_path), "m"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
