"""
Export trained Sutra models + experiment results for the browser demo.

  * CNN: BatchNorm is folded into the preceding convolutions (w' = w*g/s, b' = (b-mu)*g/s + beta),
    verified numerically, then weights are int8-quantised.
  * GPT: 2-D weights int8-quantised per row; LayerNorm params and biases stay float32.
  * Golden outputs computed in Python with the SAME dequantised weights, so the JS engine can
    self-test on page load.

    python web/export.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sutra as st  # noqa: E402
from sutra import quantize, serialize  # noqa: E402
from sutra.data import mnist  # noqa: E402
from sutra.models import GPT, DigitCNN, GPTConfig  # noqa: E402
from sutra.tokenizer import tokenizer_from_dict  # noqa: E402

WEB = ROOT / "web"
OUT = WEB / "assets"


def write_bundle(name: str, tensors: dict, meta: dict, out_dir: Path | None = None) -> dict:
    """One binary blob + a JSON manifest. int8 tensors carry float32 per-row scales. 4-byte aligned."""
    out_dir = out_dir or OUT
    blob, manifest = bytearray(), []

    def put(arr):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        return off

    deq = {}
    for key, arr in tensors.items():
        arr = np.asarray(arr, dtype=np.float32)
        if quantize.should_quantize(key, arr):
            q, s = quantize.quantize_int8(arr)
            manifest.append({"name": key, "shape": list(arr.shape), "dtype": "int8",
                             "offset": put(q), "scale_offset": put(s)})
            deq[key] = quantize.dequantize_int8(q, s)
        else:
            manifest.append({"name": key, "shape": list(arr.shape), "dtype": "float32", "offset": put(arr)})
            deq[key] = arr
    (out_dir / f"{name}.bin").write_bytes(bytes(blob))
    (out_dir / f"{name}.json").write_text(json.dumps({"tensors": manifest, **meta}))
    print(f"{name}: {len(blob) / 1e6:.2f} MB ({len(manifest)} tensors)")
    return deq


def fold_bn(conv_w, conv_b, bn: dict, eps=1e-5):
    s = bn["weight"] / np.sqrt(bn["running_var"] + eps)
    return conv_w * s[:, None, None, None], (conv_b - bn["running_mean"]) * s + bn["bias"]


def export_cnn():
    sd, meta = serialize.load_state(ROOT / "runs" / "mnist" / "model.npz")
    bn = lambda i: {k.split(".")[-1]: v for k, v in sd.items() if k.startswith(f"features.{i}.")}  # noqa: E731
    w1, b1 = fold_bn(sd["features.0.weight"], sd["features.0.bias"], bn(1))
    w2, b2 = fold_bn(sd["features.4.weight"], sd["features.4.bias"], bn(5))
    tensors = {"conv1.weight": w1, "conv1.bias": b1, "conv2.weight": w2, "conv2.bias": b2,
               "fc1.weight": sd["classifier.1.weight"], "fc1.bias": sd["classifier.1.bias"],
               "fc2.weight": sd["classifier.4.weight"], "fc2.bias": sd["classifier.4.bias"]}

    # 1) folding must be exact (float): compare against the real model in eval mode
    x_tr, _, x_te, y_te = mnist()
    x = ((x_te[:256].astype(np.float32) / 255 - meta["mean"]) / meta["std"])[:, None]
    model = DigitCNN()
    model.load_state_dict(sd)
    model.eval()
    with st.no_grad():
        ref = model(st.Tensor(x)).data
    got = cnn_forward(tensors, x)
    assert np.abs(ref - got).max() < 1e-3, "BN folding mismatch"
    print(f"BN folding verified: max |Δlogit| = {np.abs(ref - got).max():.2e}")

    deq = write_bundle("cnn", tensors, {"mean": meta["mean"], "std": meta["std"]})
    # 2) golden logits for the JS self-test, from the int8 weights the browser will use
    idx = [int(i) for i in np.random.default_rng(7).choice(len(x_te), 40, replace=False)]
    xs = ((x_te[idx].astype(np.float32) / 255 - meta["mean"]) / meta["std"])[:, None]
    golden = cnn_forward(deq, xs)
    return {"samples": [{"label": int(y_te[i]), "pixels": x_te[i].reshape(-1).tolist()} for i in idx],
            "golden_logits": golden[:4].tolist()}


def cnn_forward(t, x):
    F = st.F
    with st.no_grad():
        h = st.Tensor(x.astype(np.float32))
        h = F.max_pool2d(F.conv2d(h, st.Tensor(t["conv1.weight"]), st.Tensor(t["conv1.bias"]), 1, 1).relu(), 2)
        h = F.max_pool2d(F.conv2d(h, st.Tensor(t["conv2.weight"]), st.Tensor(t["conv2.bias"]), 1, 1).relu(), 2)
        h = F.linear(h.flatten(1), st.Tensor(t["fc1.weight"]), st.Tensor(t["fc1.bias"])).relu()
        return F.linear(h, st.Tensor(t["fc2.weight"]), st.Tensor(t["fc2.bias"])).data


def export_gpt():
    sd, meta = serialize.load_state(ROOT / "runs" / "gpt-char" / "model.npz")
    cfg = GPTConfig(**meta["config"])
    deq = write_bundle("gpt", sd, {"config": cfg.to_dict(), "tokenizer": meta["tokenizer"]})
    model = GPT(cfg)
    model.load_state_dict(deq)
    model.eval()
    tok = tokenizer_from_dict(meta["tokenizer"])
    prompt = "ROMEO:\n"
    ids = np.array([tok.encode(prompt)])
    with st.no_grad():
        logits, _ = model(ids)
    greedy = model.generate(ids, 60, temperature=0)
    return {"prompt": prompt, "last_logits": logits.data[0, -1].tolist(),
            "greedy": tok.decode(greedy[0, len(prompt):].tolist())}


def finite(obj):
    """JSON for browsers: Infinity/NaN are not valid JSON, so clamp overflow and null out NaN."""
    if isinstance(obj, float):
        return None if obj != obj else max(min(obj, 1e30), -1e30)
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


def copy_results():
    data = {}
    for p in sorted((ROOT / "experiments" / "results").glob("*.json")):
        data[p.stem] = json.loads(p.read_text())
    for name, path in [("mnist_log", ROOT / "runs" / "mnist" / "log.json"),
                       ("gpt_log", ROOT / "runs" / "gpt-char" / "log.json")]:
        if path.exists():
            log = json.loads(path.read_text())
            if name == "mnist_log":
                log.pop("misclassified", None)
            data[name] = log
    return data


def count_tests() -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True)
    per_file = re.findall(r"^tests/\S+\.py: (\d+)$", r.stdout, re.M)  # -qq output: "tests/x.py: N"
    if per_file:
        return sum(int(n) for n in per_file)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    return int(m.group(1)) if m else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cnn = export_cnn()
    gpt = export_gpt()
    results = copy_results()
    payload = {"cnn": cnn, "gpt": gpt, "results": results, "tests": count_tests(),
               "engine_lines": len((WEB / "sutra.js").read_text().splitlines())}
    (OUT / "demo.json").write_text(json.dumps(finite(payload), allow_nan=False))
    print(f"demo.json: {(OUT / 'demo.json').stat().st_size / 1e6:.2f} MB; results: {sorted(results)}")
    print(f"tests: {payload['tests']}, engine lines: {payload['engine_lines']}")
    print(f"GPT greedy continuation of {gpt['prompt']!r}: {gpt['greedy']!r}")


if __name__ == "__main__":
    main()
