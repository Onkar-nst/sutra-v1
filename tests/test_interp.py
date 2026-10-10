"""Interpretability tools: head scores on hand-built attention patterns, ablation bookkeeping, logit lens."""
import numpy as np
import pytest

import sutra as st
from sutra import interp
from sutra.models import GPT, GPTConfig


def synthetic_patterns(prefix, L, kinds):
    """One layer, one sequence; head i attends exactly where `kinds[i]` says for every query."""
    T = prefix + 2 * L
    pats = np.zeros((1, 1, len(kinds), T, T))
    for h, kind in enumerate(kinds):
        for t in range(T):
            if kind == "prev":
                k = max(t - 1, 0)
            elif kind == "dup":
                k = t - L if t >= prefix + L else t
            elif kind == "induction":
                k = t - L + 1 if t >= prefix + L else t
            else:  # attend to self
                k = t
            pats[0, 0, h, t, k] = 1.0
    return pats


def test_head_scores_identify_each_head_type():
    s = interp.head_scores(synthetic_patterns(1, 10, ["prev", "dup", "induction", "self"]), 1, 10)
    np.testing.assert_allclose(s["previous_token"][0], [1, 0, 0, 0])
    np.testing.assert_allclose(s["duplicate_token"][0], [0, 1, 0, 0])
    np.testing.assert_allclose(s["induction"][0], [0, 0, 1, 0])


def test_repeated_sequences_repeat():
    seqs = interp.repeated_sequences(range(5, 30), 4, 12, prefix_ids=[0], seed=1)
    assert seqs.shape == (4, 25)
    np.testing.assert_array_equal(seqs[:, 1:13], seqs[:, 13:])
    assert (seqs[:, 0] == 0).all()


CFG = GPTConfig(vocab_size=23, block_size=32, n_layer=2, n_head=2, n_embd=16)


def test_ablation_context_restores_model_and_matrix_shape():
    m = GPT(CFG).eval()
    x = np.random.default_rng(0).integers(0, 23, (2, 16))
    batches = [(x, np.roll(x, -1, axis=1))]
    means = interp.head_means(m, batches)
    assert len(means) == 2 and means[0].shape == (2, 8)
    before = m(x)[0].data
    with interp.ablated(m, [(0, 1), (1, 0)], means):
        during = m(x)[0].data
    after = m(x)[0].data
    assert not np.allclose(before, during)
    np.testing.assert_array_equal(before, after)
    assert all(b.attn.head_mask is None for b in m.blocks)
    res = interp.ablation_matrix(m, {"loss": lambda mm: st.F.cross_entropy(mm(x)[0], batches[0][1]).item()}, means)
    assert res["delta"]["loss"].shape == (2, 2)


def test_logit_lens_last_layer_is_model_loss():
    m = GPT(CFG).eval()
    x = np.random.default_rng(1).integers(0, 23, (2, 16))
    y = np.roll(x, -1, axis=1)
    lens = interp.logit_lens(m, [(x, y)])
    assert len(lens) == 3
    assert lens[-1] == pytest.approx(m(x, y)[1].item(), rel=1e-6)


def test_copy_losses_on_a_model_that_copies():
    """A 'model' that predicts the second copy perfectly must score a much lower second-copy loss."""
    class Copier:
        training = False

        def train(self, mode=True):
            return self

        def eval(self):
            return self

        def __call__(self, idx):
            L = (idx.shape[1] + 1 - 1) // 2
            logits = np.zeros(idx.shape + (30,))
            for t in range(idx.shape[1]):
                if t >= 1 + L - 1 and t - L + 1 >= 1:
                    logits[:, t, :] = -20.0
                    logits[np.arange(idx.shape[0]), t, idx[:, t - L + 1]] = 20.0
            return st.Tensor(logits), None

    seqs = interp.repeated_sequences(range(30), 3, 10, prefix_ids=[0], seed=2)
    first, second = interp.copy_losses(Copier(), seqs, 1, 10)
    assert second < 1e-3 < first
