"""Every Tensor primitive vs PyTorch: forward values and vector-Jacobian products."""
import numpy as np
import pytest
import torch
from conftest import check, randn

import sutra as st

BROADCAST_PAIRS = [((3, 4), (3, 4)), ((3, 4), (4,)), ((2, 3, 4), (3, 1)), ((1, 4), (5, 1)), ((2, 1, 4), (1, 3, 1)), ((), (3,))]


@pytest.mark.parametrize("sa,sb", BROADCAST_PAIRS)
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_binary_broadcast(op, sa, sb):
    f = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
         "div": lambda a, b: a / b}[op]
    b = randn(*sb, seed=1)
    if op == "div":
        b = np.abs(b) + 0.5
    check(f, f, randn(*sa, seed=0), b)


@pytest.mark.parametrize("sa,sb", [((3, 4), (4, 5)), ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)),
                                   ((2, 1, 3, 4), (5, 4, 2)), ((4,), (4, 5)), ((3, 4), (4,)), ((4,), (4,)),
                                   ((2, 3, 4), (4,))])
def test_matmul(sa, sb):
    check(lambda a, b: a @ b, lambda a, b: a @ b, randn(*sa, seed=0), randn(*sb, seed=1))


def test_scalar_ops_and_reflected():
    f = lambda a: (2.0 - a) * 3.0 / (1.5 + a * a) + 0.5 ** a - a ** 3
    check(f, f, randn(3, 4))


def test_tensor_power():
    check(lambda a, b: a ** b, lambda a, b: a ** b, np.abs(randn(3, 4)) + 0.2, randn(3, 4, seed=2))


POS = lambda *s: np.abs(randn(*s)) + 0.1  # noqa: E731


@pytest.mark.parametrize("name,st_f,th_f,pos", [
    ("exp", lambda x: x.exp(), torch.exp, False),
    ("log", lambda x: x.log(), torch.log, True),
    ("sqrt", lambda x: x.sqrt(), torch.sqrt, True),
    ("tanh", lambda x: x.tanh(), torch.tanh, False),
    ("sigmoid", lambda x: x.sigmoid(), torch.sigmoid, False),
    ("relu", lambda x: x.relu(), torch.relu, False),
    ("gelu", lambda x: x.gelu(), lambda x: torch.nn.functional.gelu(x, approximate="tanh"), False),
    ("abs", lambda x: x.abs(), torch.abs, False),
    ("neg", lambda x: -x, lambda x: -x, False),
    ("clamp", lambda x: x.clamp(-0.5, 0.7), lambda x: x.clamp(-0.5, 0.7), False),
    ("leaky_relu", lambda x: st.F.leaky_relu(x, 0.1), lambda x: torch.nn.functional.leaky_relu(x, 0.1), False),
])
def test_unary(name, st_f, th_f, pos):
    check(st_f, th_f, POS(4, 5) if pos else randn(4, 5))


def test_sigmoid_is_stable_for_extreme_inputs():
    x = st.Tensor(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]), requires_grad=True)
    y = x.sigmoid()
    y.sum().backward()
    assert np.all(np.isfinite(y.data)) and np.all(np.isfinite(x.grad))


AXES = [None, 0, 1, -1, (0, 2), (1, 2)]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("op", ["sum", "mean", "max", "min", "logsumexp", "var"])
def test_reductions(op, axis, keepdims):
    if op == "logsumexp" and axis is None:
        axis = (0, 1, 2)
    th_axis = axis if axis is not None else tuple(range(3))

    def st_f(x):
        if op == "var":
            return x.var(axis=axis, keepdims=keepdims)
        return getattr(x, op)(axis=axis, keepdims=keepdims)

    def th_f(x):
        if op in ("sum", "mean", "logsumexp"):
            return getattr(torch, op)(x, dim=th_axis, keepdim=keepdims)
        if op == "var":
            return torch.var(x, dim=th_axis, keepdim=keepdims, unbiased=False)
        return (torch.amax if op == "max" else torch.amin)(x, dim=th_axis, keepdim=keepdims)
    check(st_f, th_f, randn(2, 3, 4))


def test_max_ties_split_gradient_like_torch():
    a = np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
    check(lambda x: x.max(axis=1), lambda x: torch.amax(x, dim=1), a)


def test_shape_ops():
    x = randn(2, 3, 4)
    check(lambda t: t.reshape(4, 6), lambda t: t.reshape(4, 6), x)
    check(lambda t: t.transpose(2, 0, 1), lambda t: t.permute(2, 0, 1), x)
    check(lambda t: t.T, lambda t: t.permute(2, 1, 0), x)
    check(lambda t: t.swapaxes(0, 2), lambda t: t.transpose(0, 2), x)
    check(lambda t: t.flatten(1), lambda t: t.flatten(1), x)
    check(lambda t: t.unsqueeze(1).squeeze(1), lambda t: t.unsqueeze(1).squeeze(1), x)
    check(lambda t: t.reshape(2, 1, 12).expand(2, 5, 12), lambda t: t.reshape(2, 1, 12).expand(2, 5, 12), x)


@pytest.mark.parametrize("idx", [
    (0,), (slice(1, None),), (Ellipsis, 1), (slice(None), slice(None, None, 2)), (None, 0, slice(1, 3)),
    (np.array([0, 1, 1, 0]),), (slice(None), np.array([2, 2, 0])), (np.array([1, 0]), np.array([2, 2])),
])
def test_getitem(idx):
    th_idx = tuple(torch.tensor(i) if isinstance(i, np.ndarray) else i for i in idx)
    check(lambda t: t[idx], lambda t: t[th_idx], randn(2, 3, 4))


def test_getitem_boolean_mask():
    x = randn(3, 4)
    m = x > 0.1
    check(lambda t: t[m], lambda t: t[torch.tensor(m)], x)


def test_cat_stack_split_pad():
    a, b, c = randn(2, 3, seed=0), randn(2, 5, seed=1), randn(2, 3, seed=2)
    check(lambda x, y: st.cat([x, y, x], axis=1), lambda x, y: torch.cat([x, y, x], dim=1), a, b)
    check(lambda x, y: st.stack([x, y], axis=1), lambda x, y: torch.stack([x, y], dim=1), a, c)
    check(lambda x: st.cat(x.split([1, 3, 1], axis=1)[::-1], axis=1),
          lambda x: torch.cat(x.split([1, 3, 1], dim=1)[::-1], dim=1), b)
    check(lambda x: st.pad(x, ((1, 2), (0, 3))), lambda x: torch.nn.functional.pad(x, (0, 3, 1, 2)), a)


def test_where_maximum_minimum_masked_fill():
    a, b = randn(3, 4, seed=0), randn(4, seed=1)
    cond = a > 0
    check(lambda x, y: st.where(cond, x, y), lambda x, y: torch.where(torch.tensor(cond), x, y), a, b)
    check(lambda x, y: st.maximum(x, y), torch.maximum, a, b)
    check(lambda x, y: st.minimum(x, y), torch.minimum, a, b)
    check(lambda x: x.masked_fill(cond, -2.0), lambda x: x.masked_fill(torch.tensor(cond), -2.0), a)
