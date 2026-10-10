"""Speculative decoding: greedy equivalence, full acceptance with an identical draft, and distributional exactness."""

import numpy as np

import sutra as st
from sutra.models import GPT, GPTConfig
from sutra.models.gpt import next_token_probs
from sutra.models.speculative import CachedStream, generate_speculative, generate_target_only


def peaked_model(seed, vocab=11, block=32, layers=2):
    """Random GPT whose weights are scaled up so its next-token distributions are far from uniform."""
    st.manual_seed(seed)
    m = GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=layers, n_head=2, n_embd=16))
    for p in m.parameters():
        if p.ndim == 2:
            p.data = p.data * 25
    return m.eval()


def test_cached_stream_rollback_matches_fresh_forward():
    m = peaked_model(0)
    s = CachedStream(m)
    s.prime([1, 2, 3])
    s.feed([4, 5, 6])
    s.rollback(4)
    a = s.feed([7])[-1]
    b = CachedStream(m).prime([1, 2, 3, 4, 7])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_greedy_speculative_equals_greedy_target():
    target, draft = peaked_model(1), peaked_model(2, layers=1)
    prompt = [3, 1, 4]
    ref, _ = generate_target_only(target, prompt, 24, temperature=0)
    for k in (1, 3, 5):
        out, stats = generate_speculative(target, draft, prompt, 24, k=k, temperature=0)
        assert out == ref
        assert stats["rounds"] <= 24


def test_identical_draft_accepts_everything():
    m = peaked_model(3)
    _, stats = generate_speculative(m, m, [1, 2], 20, k=4, temperature=1.0, seed=5)
    assert stats["acceptance_rate"] == 1.0
    assert stats["tokens_per_target_call"] == 5.0  # k accepted + 1 bonus per round


def test_output_distribution_is_exactly_the_targets():
    """Joint law of the first two sampled tokens must match p_target(a) * p_target(b | a)."""
    V = 3
    target, draft = peaked_model(4, vocab=V, block=8), peaked_model(5, vocab=V, block=8, layers=1)
    prompt = [0, 1]

    def joint(model):
        s = CachedStream(model)
        pa = next_token_probs(s.prime(prompt)[None])[0]
        out = np.zeros((V, V))
        for a in range(V):
            pb = next_token_probs(CachedStream(model).prime(prompt + [a])[None])[0]
            out[a] = pa[a] * pb
        return out

    exact, draft_law = joint(target), joint(draft)
    assert 0.5 * np.abs(exact - draft_law).sum() > 0.2  # the draft really is different: the test has power
    n = 2500
    counts = np.zeros((V, V))
    for seed in range(n):
        out, _ = generate_speculative(target, draft, prompt, 2, k=2, temperature=1.0, seed=seed)
        counts[out[2], out[3]] += 1
    tv = 0.5 * np.abs(counts / n - exact).sum()
    assert tv < 0.05, (tv, counts / n, exact)


class UnigramLM:
    """A context-free 'language model' with fixed next-token logits, so the exact answer is known."""

    def __init__(self, probs, block=64):
        self.logits = np.log(np.asarray(probs, dtype=float))
        self.blocks = [None]
        self.cfg = GPTConfig(vocab_size=len(probs), block_size=block)

    def eval(self):
        return self

    def __call__(self, tokens, kv_caches=None, pos_offset=0):
        T = tokens.shape[1]
        for c in kv_caches:  # a stand-in cache of the right length, so rollback is exercised
            c["k"] = c["v"] = np.zeros((1, 1, pos_offset + T, 1))
        return st.Tensor(np.tile(self.logits, (1, T, 1))), None


def test_speculative_sampling_is_exact_and_accepts_at_the_theoretical_rate():
    """p and q chosen so that a wrong residual distribution would shift the output by 0.2 in TV."""
    p, q = np.array([0.6, 0.4, 1e-12]), np.array([0.5, 1e-12, 0.5])
    target, draft = UnigramLM(p), UnigramLM(q)
    counts, acc, examined = np.zeros(3), 0, 0
    for seed in range(4000):
        out, s = generate_speculative(target, draft, [0], 4, k=3, temperature=1.0, seed=seed)
        np.add.at(counts, out[1:], 1)
        acc, examined = acc + s["accepted"], examined + s["examined"]
    freq = counts / counts.sum()
    alpha = np.minimum(p, q).sum()                               # = 1 - TV(p, q) = 0.5
    assert 0.5 * np.abs(freq - p).sum() < 0.015, freq           # every emitted token ~ p, never q
    assert abs(acc / examined - alpha) < 0.02                   # per-proposal acceptance rate
    _, s = generate_speculative(target, draft, [0], 6000, k=3, temperature=1.0, seed=1)
    assert abs(s["tokens_per_target_call"] - (1 - alpha ** 4) / (1 - alpha)) < 0.05  # = 1.875 tokens per call
