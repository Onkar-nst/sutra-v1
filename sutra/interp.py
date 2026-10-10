"""
Mechanistic-interpretability tools for Sutra GPTs.

  * attention patterns, and per-head scores for three well-known head types:
      previous-token heads   attend from position t to t-1
      duplicate-token heads  attend from a repeated token to its earlier occurrence
      induction heads        attend from a repeated token to the token that FOLLOWED its earlier
                             occurrence, then copy it (Elhage et al., 2021; Olsson et al., 2022)
  * a repeated-random-sequence probe: on [prefix, r, r] a model that can copy in context has much
    lower loss on the second copy than on the first
  * mean-ablation of any set of heads, measured on any loss
  * the logit lens (nostalgebraist, 2020): decode the residual stream after every layer
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .tensor import no_grad


def _attn(model):
    return [b.attn for b in model.blocks]


@contextmanager
def _eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


@no_grad()
def attention_patterns(model, idx) -> np.ndarray:
    """Attention probabilities, shape (n_layer, B, n_head, T, T)."""
    mods = _attn(model)
    with _eval_mode(model):
        for m in mods:
            m.keep_attention = True
        try:
            model(idx)
            return np.stack([m.last_attention for m in mods])
        finally:
            for m in mods:
                m.keep_attention, m.last_attention = False, None


def repeated_sequences(token_ids, n_seqs: int, length: int, prefix_ids=(), seed: int = 0) -> np.ndarray:
    """Random strings repeated twice: [prefix, r_1..r_L, r_1..r_L], shape (n_seqs, len(prefix) + 2L)."""
    rng = np.random.default_rng(seed)
    r = rng.choice(np.asarray(token_ids), size=(n_seqs, length))
    pre = np.tile(np.asarray(prefix_ids, dtype=np.int64).reshape(1, -1), (n_seqs, 1))
    return np.concatenate([pre, r, r], axis=1).astype(np.int64)


@no_grad()
def per_position_loss(model, seqs) -> np.ndarray:
    """Loss for predicting token t+1 from position t, averaged over the batch. Shape (T-1,)."""
    with _eval_mode(model):
        logits, _ = model(seqs[:, :-1])
    z = logits.data.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return -np.take_along_axis(logp, seqs[:, 1:, None], axis=-1)[..., 0].mean(axis=0)


def copy_losses(model, seqs, prefix_len: int, length: int) -> tuple[float, float]:
    """Mean loss on the first copy and on the second copy of repeated sequences."""
    loss = per_position_loss(model, seqs)  # loss[t] predicts token t+1
    first = loss[prefix_len:prefix_len + length - 1].mean()          # predicting r_2..r_L of copy 1
    second = loss[prefix_len + length:prefix_len + 2 * length - 1].mean()  # predicting r_2..r_L of copy 2
    return float(first), float(second)


def head_scores(pats: np.ndarray, prefix_len: int, length: int) -> dict:
    """
    Per-head scores (n_layer, n_head) from attention patterns on repeated sequences, averaged over every
    query in the second copy and over the batch.
    """
    q = np.arange(prefix_len + length, prefix_len + 2 * length)
    q = q[q < pats.shape[-1]]
    ind = q - length + 1
    ok = ind < pats.shape[-1]
    return {
        "previous_token": pats[..., q, q - 1].mean(axis=(1, 3)),
        "duplicate_token": pats[..., q, q - length].mean(axis=(1, 3)),
        "induction": pats[..., q[ok], ind[ok]].mean(axis=(1, 3)),
    }


@no_grad()
def head_means(model, batches) -> list[np.ndarray]:
    """Mean output of every head over natural text, shape (n_head, head_dim) per layer (for mean-ablation)."""
    mods = _attn(model)
    sums = [0.0] * len(mods)
    count = 0
    with _eval_mode(model):
        for m in mods:
            m.keep_head_output = True
        try:
            for x, _ in batches:
                model(x)
                for i, m in enumerate(mods):
                    sums[i] = sums[i] + m.last_head_output.sum(axis=(0, 2))
                count += x.shape[0] * x.shape[1]
        finally:
            for m in mods:
                m.keep_head_output, m.last_head_output = False, None
    return [s / count for s in sums]


@contextmanager
def ablated(model, heads, means):
    """Mean-ablate the given (layer, head) pairs inside the `with` block."""
    mods = _attn(model)
    for layer, head in heads:
        m = mods[layer]
        if m.head_mask is None:
            m.head_mask = np.ones(m.n_heads)
            m.head_mean = means[layer]
        m.head_mask[head] = 0.0
    try:
        yield model
    finally:
        for m in mods:
            m.head_mask, m.head_mean = None, None


def ablation_matrix(model, loss_fns: dict, means) -> dict:
    """For every single head: the change in each loss when that head alone is mean-ablated."""
    L, H = len(model.blocks), model.blocks[0].attn.n_heads
    base = {k: fn(model) for k, fn in loss_fns.items()}
    out = {k: np.zeros((L, H)) for k in loss_fns}
    for layer in range(L):
        for head in range(H):
            with ablated(model, [(layer, head)], means):
                for k, fn in loss_fns.items():
                    out[k][layer, head] = fn(model) - base[k]
    return {"baseline": base, "delta": out}


@no_grad()
def logit_lens(model, batches) -> list[float]:
    """Loss when the residual stream after the embedding and after each block is decoded directly."""
    from . import functional as F
    totals = None
    with _eval_mode(model):
        for x, y in batches:
            hidden = []
            model(x, hidden=hidden)
            losses = [F.cross_entropy(model.unembed(h), y).item() for h in hidden]
            totals = losses if totals is None else [a + b for a, b in zip(totals, losses)]
    return [t / len(batches) for t in totals]
