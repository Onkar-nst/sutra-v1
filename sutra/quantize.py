"""
Post-training weight quantisation: symmetric int8 with one scale per output row/channel.

    w ≈ scale[row] * q,   q ∈ [-127, 127],   scale[row] = max|w[row]| / 127

Per-row scales keep a single outlier from destroying the resolution of every other row.
Used to ship models to the browser at ~1/4 of their float32 size.
"""
from __future__ import annotations

import numpy as np


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = w.reshape(w.shape[0], -1).astype(np.float32)
    scale = np.abs(rows).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale.astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.reshape(q.shape[0], -1).astype(np.float32) * scale[:, None]).reshape(q.shape)


def should_quantize(name: str, arr: np.ndarray, min_size: int = 1024) -> bool:
    """Quantise matrices / conv kernels; keep biases, norms and running stats in float32."""
    return arr.ndim >= 2 and arr.size >= min_size


def quantize_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        out[k] = quantize_int8(v) if should_quantize(k, v) else v.astype(np.float32)
    return out


def simulate_int8(module) -> dict:
    """Replace a module's weights by their int8 round-trip (in place). Returns error statistics."""
    sd = module.state_dict()
    stats, new = {}, {}
    for k, v in sd.items():
        if should_quantize(k, v):
            deq = dequantize_int8(*quantize_int8(v))
            stats[k] = float(np.abs(deq - v).max() / (np.abs(v).max() + 1e-12))
            new[k] = deq
        else:
            new[k] = v
    module.load_state_dict(new)
    return stats


def size_bytes(sd: dict, int8: bool) -> int:
    total = 0
    for k, v in sd.items():
        if int8 and should_quantize(k, v):
            total += v.size + 4 * v.shape[0]
        else:
            total += 4 * v.size
    return total
