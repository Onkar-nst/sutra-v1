"""Neural-network ops (fused backward passes) vs PyTorch."""
import numpy as np
import pytest
import torch
import torch.nn.functional as TF
from conftest import check, randn

import sutra as st
from sutra import functional as F


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_softmax_and_log_softmax(axis):
    x = randn(3, 5, 4) * 3
    check(lambda t: F.softmax(t, axis), lambda t: torch.softmax(t, axis), x)
    check(lambda t: F.log_softmax(t, axis), lambda t: torch.log_softmax(t, axis), x)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_cross_entropy(reduction, smoothing):
    logits = randn(8, 7) * 2
    target = np.array([0, 3, 6, 2, -100, 1, 1, 5])
    check(lambda t: F.cross_entropy(t, target, label_smoothing=smoothing, reduction=reduction),
          lambda t: TF.cross_entropy(t, torch.tensor(target), label_smoothing=smoothing, reduction=reduction),
          logits)


def test_cross_entropy_class_dim_last_for_sequences():
    logits, target = randn(2, 5, 11), np.random.default_rng(0).integers(0, 11, (2, 5))
    check(lambda t: F.cross_entropy(t, target),
          lambda t: TF.cross_entropy(t.reshape(-1, 11), torch.tensor(target).reshape(-1)), logits)


def test_cross_entropy_is_stable_for_huge_logits():
    x = st.Tensor(np.array([[1e4, -1e4, 0.0]]), requires_grad=True)
    loss = F.cross_entropy(x, np.array([1]))
    loss.backward()
    assert np.isfinite(loss.item()) and np.all(np.isfinite(x.grad))


def test_bce_with_logits_and_mse():
    x, y = randn(6, 3) * 3, (randn(6, 3, seed=5) > 0).astype(float)
    check(lambda t: F.binary_cross_entropy_with_logits(t, y),
          lambda t: TF.binary_cross_entropy_with_logits(t, torch.tensor(y)), x)
    check(lambda t: F.mse_loss(t, y), lambda t: TF.mse_loss(t, torch.tensor(y)), x)


@pytest.mark.parametrize("xshape", [(5, 4), (2, 3, 4)])
def test_linear(xshape):
    check(lambda x, w, b: F.linear(x, w, b), TF.linear, randn(*xshape), randn(6, 4, seed=1), randn(6, seed=2))
    check(lambda x, w: F.linear(x, w), TF.linear, randn(*xshape), randn(6, 4, seed=1))


def test_embedding_with_repeated_indices():
    idx = np.array([[0, 2, 2], [4, 0, 2]])
    check(lambda w: F.embedding(idx, w), lambda w: TF.embedding(torch.tensor(idx), w), randn(5, 3))


def test_layer_norm():
    check(lambda x, w, b: F.layer_norm(x, w, b), lambda x, w, b: TF.layer_norm(x, (6,), w, b),
          randn(2, 3, 6) * 2 + 1, randn(6, seed=1), randn(6, seed=2))
    check(lambda x: F.layer_norm(x), lambda x: TF.layer_norm(x, (6,)), randn(4, 6))


@pytest.mark.parametrize("shape", [(8, 5), (4, 3, 5, 5)])
@pytest.mark.parametrize("training", [True, False])
def test_batch_norm(shape, training):
    C = shape[1]
    rm_s, rv_s = np.full(C, 0.3), np.full(C, 1.7)
    rm_t, rv_t = torch.tensor(rm_s.copy()), torch.tensor(rv_s.copy())
    check(lambda x, w, b: F.batch_norm(x, rm_s, rv_s, w, b, training=training),
          lambda x, w, b: TF.batch_norm(x, rm_t, rv_t, w, b, training=training),
          randn(*shape) * 2 + 0.5, randn(C, seed=1), randn(C, seed=2))
    np.testing.assert_allclose(rm_s, rm_t.numpy(), atol=1e-10)
    np.testing.assert_allclose(rv_s, rv_t.numpy(), atol=1e-10)


@pytest.mark.parametrize("stride,padding,k", [(1, 0, 3), (1, 1, 3), (2, 1, 3), (2, 0, 2), (3, 2, 5), (1, 0, 1)])
def test_conv2d(stride, padding, k):
    check(lambda x, w, b: F.conv2d(x, w, b, stride, padding),
          lambda x, w, b: TF.conv2d(x, w, b, stride, padding),
          randn(2, 3, 9, 8), randn(4, 3, k, k, seed=1), randn(4, seed=2))


@pytest.mark.parametrize("k,s", [(2, None), (3, 1), (3, 2)])
def test_pooling(k, s):
    x = randn(2, 3, 8, 7)
    check(lambda t: F.max_pool2d(t, k, s), lambda t: TF.max_pool2d(t, k, s), x)
    check(lambda t: F.avg_pool2d(t, k, s), lambda t: TF.avg_pool2d(t, k, s), x)


@pytest.mark.parametrize("causal", [False, True])
def test_attention(causal):
    q, k, v = randn(2, 3, 5, 4, seed=0), randn(2, 3, 5, 4, seed=1), randn(2, 3, 5, 4, seed=2)
    check(lambda a, b, c: F.scaled_dot_product_attention(a, b, c, causal=causal),
          lambda a, b, c: TF.scaled_dot_product_attention(a, b, c, is_causal=causal), q, k, v)


def test_attention_with_padding_mask_and_kv_cache_shape():
    q, k, v = randn(1, 2, 3, 4, seed=0), randn(1, 2, 6, 4, seed=1), randn(1, 2, 6, 4, seed=2)
    blocked = np.zeros((3, 6), dtype=bool)
    blocked[:, 0] = True
    # queries are the LAST 3 of 6 positions (kv-cache situation): causal must offset the diagonal
    allowed = ~(F.causal_mask(3, 6) | blocked)
    check(lambda a, b, c: F.scaled_dot_product_attention(a, b, c, causal=True, mask=blocked),
          lambda a, b, c: TF.scaled_dot_product_attention(a, b, c, attn_mask=torch.tensor(allowed)), q, k, v)


def test_dropout_statistics_and_gradient():
    st.manual_seed(1)
    x = st.Tensor(np.ones((200, 200)), requires_grad=True)
    y = F.dropout(x, 0.3, training=True)
    kept = y.data != 0
    assert abs(kept.mean() - 0.7) < 0.01
    assert np.allclose(y.data[kept], 1 / 0.7)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, y.data)
    assert F.dropout(x, 0.3, training=False) is x
