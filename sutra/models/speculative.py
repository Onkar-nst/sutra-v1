"""
Speculative decoding (Leviathan et al., 2023; Chen et al., 2023).

A small draft model proposes k tokens one at a time; the large target model scores all k in a single
forward pass; each proposal x_i is accepted with probability min(1, p_i(x_i) / q_i(x_i)). On the first
rejection a replacement is drawn from the residual distribution norm(max(0, p_i - q_i)), and if all k are
accepted one bonus token is drawn from p_{k+1}. The emitted tokens are distributed exactly as if sampled
from the target alone, while the target runs once per round instead of once per token.

Both models decode through a KV cache that supports rollback, so rejected proposals cost nothing later.
"""
from __future__ import annotations

import time

import numpy as np

from ..tensor import no_grad
from .gpt import next_token_probs


class CachedStream:
    """Incremental decoding for one sequence: feed tokens, get logits, roll the KV cache back."""

    def __init__(self, model):
        self.model = model
        self.block = model.cfg.block_size
        self.reset()

    def reset(self):
        self.caches = [{} for _ in self.model.blocks]
        self.pos = 0

    @no_grad()
    def feed(self, tokens) -> np.ndarray:
        """Append tokens to the context; returns logits (len(tokens), V) for the positions just fed."""
        tokens = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
        logits, _ = self.model(tokens, kv_caches=self.caches, pos_offset=self.pos)
        self.pos += tokens.shape[1]
        return logits.data[0]

    def rollback(self, n: int):
        """Forget everything after the first n positions."""
        for c in self.caches:
            if "k" in c:
                c["k"], c["v"] = c["k"][:, :, :n], c["v"][:, :, :n]
        self.pos = n

    def prime(self, context) -> np.ndarray:
        """Restart from a context; returns next-token logits."""
        self.reset()
        return self.feed(context)[-1]


def _sample(p: np.ndarray, rng) -> int:
    return int(rng.choice(len(p), p=p))


def _probs(logits, temperature, top_k):
    if temperature <= 0:  # greedy: a one-hot distribution on the argmax
        p = np.zeros(logits.shape[-1])
        p[int(np.argmax(logits))] = 1.0
        return p
    return next_token_probs(logits[None], temperature, top_k)[0]


def _window(out, block):
    """The context kept when the cache is full: the most recent half window."""
    return out[-(block // 2):]


@no_grad()
def generate_target_only(target, prompt, n_new, temperature=1.0, top_k=None, seed=0):
    """Baseline: plain autoregressive sampling from the target through the same cached-stream machinery."""
    target.eval()
    rng = np.random.default_rng(seed)
    t = CachedStream(target)
    out = list(prompt)
    logits = t.prime(out)
    while len(out) - len(prompt) < n_new:
        x = _sample(_probs(logits, temperature, top_k), rng)
        out.append(x)
        if t.pos + 1 > t.block:
            logits = t.prime(_window(out, t.block))
        else:
            logits = t.feed([x])[-1]
    return out, {"target_calls": None}


@no_grad()
def generate_speculative(target, draft, prompt, n_new, k=4, temperature=1.0, top_k=None, seed=0):
    """Speculative sampling. Returns (tokens, stats) with acceptance statistics and forward-pass counts."""
    target.eval()
    draft.eval()
    rng = np.random.default_rng(seed)
    T, D = CachedStream(target), CachedStream(draft)
    out = list(prompt)
    p_last, q_last = T.prime(out), D.prime(out)
    proposed = examined = accepted_total = rounds = 0
    block = min(T.block, D.block)
    while len(out) - len(prompt) < n_new:
        if T.pos + k + 1 > block:  # make room: restart both models from the recent context
            ctx = _window(out, block)
            p_last, q_last = T.prime(ctx), D.prime(ctx)
        n0 = T.pos
        qs, props, ql = [], [], q_last
        for _ in range(k):  # draft proposes k tokens
            q = _probs(ql, temperature, top_k)
            x = _sample(q, rng)
            qs.append(q)
            props.append(x)
            ql = D.feed([x])[-1]
        tl = T.feed(props)  # target scores all proposals in ONE forward pass
        ps = [_probs(p_last, temperature, top_k)] + [_probs(tl[i], temperature, top_k) for i in range(k)]
        n_acc, nxt = 0, None
        for i, x in enumerate(props):
            if rng.random() < min(1.0, ps[i][x] / max(qs[i][x], 1e-300)):
                n_acc += 1
                out.append(x)
                continue
            resid = np.maximum(ps[i] - qs[i], 0.0)
            nxt = _sample(resid / resid.sum() if resid.sum() > 0 else ps[i], rng)
            break
        if nxt is None:
            nxt = _sample(ps[k], rng)  # every proposal accepted: one bonus token from the target
        out.append(nxt)
        proposed += k
        examined += n_acc + (n_acc < k)  # proposals after the first rejection are never examined
        accepted_total += n_acc
        rounds += 1
        T.rollback(n0 + n_acc)
        D.rollback(n0 + n_acc)
        p_last, q_last = T.feed([nxt])[-1], D.feed([nxt])[-1]
    out = out[:len(prompt) + n_new]
    # acceptance_rate estimates alpha = E[min(1, p/q)] = sum(min(p, q)), the per-proposal acceptance
    # probability of Leviathan et al.; tokens_per_target_call should approach (1 - alpha^(k+1)) / (1 - alpha)
    return out, {"rounds": rounds, "proposed": proposed, "examined": examined, "accepted": accepted_total,
                 "acceptance_rate": accepted_total / max(examined, 1),
                 "tokens_per_target_call": (len(out) - len(prompt)) / max(rounds, 1)}


def benchmark(target, draft, prompt, n_new, ks=(1, 2, 3, 4, 6, 8), temperature=1.0, top_k=None, seed=0):
    """Wall-clock tokens/s of target-only decoding vs speculative decoding for several draft lengths."""
    t0 = time.perf_counter()
    generate_target_only(target, prompt, n_new, temperature, top_k, seed)
    base = n_new / (time.perf_counter() - t0)
    rows = []
    for k in ks:
        t0 = time.perf_counter()
        _, s = generate_speculative(target, draft, prompt, n_new, k, temperature, top_k, seed)
        tps = n_new / (time.perf_counter() - t0)
        rows.append({"k": k, "tokens_per_s": tps, "speedup": tps / base, **s})
    return {"baseline_tokens_per_s": base, "rows": rows}
