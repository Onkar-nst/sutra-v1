"""Checkpointing: models (and any extra metadata) to a single .npz file."""
from __future__ import annotations

import json

import numpy as np


def save(module, path, **meta) -> None:
    arrays = {f"p:{k}": v for k, v in module.state_dict().items()}
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **arrays)


def load_state(path) -> tuple[dict, dict]:
    with np.load(path) as z:
        sd = {k[2:]: z[k] for k in z.files if k.startswith("p:")}
        meta = json.loads(bytes(z["__meta__"]).decode()) if "__meta__" in z.files else {}
    return sd, meta


def load(module, path) -> dict:
    sd, meta = load_state(path)
    module.load_state_dict(sd)
    return meta
