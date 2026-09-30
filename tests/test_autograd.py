"""Properties of the autograd engine itself, plus finite-difference checks that don't rely on PyTorch."""
import numpy as np
import pytest

import sutra as st
from sutra import functional as F


def numerical_grad(f, x: np.ndarray, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


@pytest.mark.parametrize("name,fn", [
    ("mlp", lambda x: ((x @ x.T).tanh() * x.sum(axis=1, keepdims=True)).mean()),
    ("softmax-attn", lambda x: F.scaled_dot_product_attention(x[None], x[None], x[None], causal=True).sum()),
    ("norms", lambda x: (F.layer_norm(x) ** 2).sum() + F.log_softmax(x, 0).max()),
    ("ce", lambda x: F.cross_entropy(x * 3, np.array([0, 1, 2, 3]), label_smoothing=0.2)),
    ("conv", lambda x: F.max_pool2d(F.conv2d(x.reshape(1, 1, 4, 5), x.reshape(4, 5)[None, None, :3, :3], padding=1), 2).sum()),
])
def test_finite_differences(name, fn):
    x0 = np.random.default_rng(3).standard_normal((4, 5))
    x = st.Tensor(x0.copy(), requires_grad=True)
    fn(x).backward()
    num = numerical_grad(lambda a: fn(st.Tensor(a)).item(), x0.copy())
    np.testing.assert_allclose(x.grad, num, atol=1e-6, rtol=1e-5)


def test_gradients_accumulate_across_backward_calls():
    x = st.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 3).sum().backward()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 3 + 2 * x.data)


def test_diamond_graph_sums_both_paths():
    x = st.Tensor(np.array(3.0), requires_grad=True)
    a = x * 2
    y = a * a + a  # dy/dx = (2a + 1) * 2
    y.backward()
    assert x.grad == pytest.approx((2 * 6 + 1) * 2)


def test_same_tensor_used_twice_in_one_op():
    x = st.Tensor(np.array([1.5, -2.0]), requires_grad=True)
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data)


def test_no_grad_builds_no_graph():
    x = st.Tensor(np.ones(3), requires_grad=True)
    with st.no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad and y.is_leaf

    @st.no_grad()
    def f(t):
        return t * 2
    assert not f(x).requires_grad
    assert st.is_grad_enabled()


def test_retain_grad_and_graph_release():
    x = st.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    h = (x * 2).retain_grad()
    y = (h * h).sum()
    y.backward(retain_graph=True)
    np.testing.assert_allclose(h.grad, 2 * h.data)
    y.backward()  # second pass works because the graph was retained
    np.testing.assert_allclose(x.grad, 2 * (2 * 2 * h.data))


def test_very_deep_graph_has_no_recursion_limit():
    x = st.Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(20_000):
        y = y * 1.0001
    y.backward()
    assert x.grad == pytest.approx(1.0001 ** 20_000, rel=1e-6)


def test_detach_blocks_gradient():
    x = st.Tensor(np.array(2.0), requires_grad=True)
    y = x * x.detach()
    y.backward()
    assert x.grad == pytest.approx(2.0)


def test_non_scalar_backward_requires_gradient():
    x = st.Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()


def test_integer_tensors_cannot_require_grad():
    with pytest.raises(TypeError):
        st.Tensor(np.arange(3), requires_grad=True)


def test_float32_stays_float32():
    st.set_default_dtype(np.float32)
    x = st.randn(3, 3, requires_grad=True)
    y = (x.gelu() * 0.5 + 2 ** x).sigmoid().logsumexp(axis=1).mean()
    y.backward()
    assert y.dtype == np.float32 and x.grad.dtype == np.float32
