"""Gradient checkpointing must give bit-for-bit the same forward pass and gradients, including with dropout."""
import numpy as np

import sutra as st
from sutra.models import GPT, GPTConfig


def run(checkpointing, dropout):
    st.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=17, block_size=16, n_layer=3, n_head=2, n_embd=16, dropout=dropout))
    m.grad_checkpointing = checkpointing
    idx = np.random.default_rng(1).integers(0, 17, (2, 12))
    st.manual_seed(42)  # same dropout masks in both runs, if the replay is correct
    _, loss = m(idx, np.roll(idx, -1, axis=1))
    loss.backward()
    return loss.item(), {n: p.grad.copy() for n, p in m.named_parameters()}


def test_checkpointed_gradients_equal_plain_gradients():
    for dropout in (0.0, 0.3):
        l1, g1 = run(False, dropout)
        l2, g2 = run(True, dropout)
        assert l1 == l2
        for name in g1:
            np.testing.assert_allclose(g2[name], g1[name], atol=1e-12, err_msg=f"{name} (dropout={dropout})")


def test_checkpoint_function_on_inputs_and_closure_params():
    w = st.Tensor(np.random.default_rng(2).standard_normal((4, 4)), requires_grad=True)
    x1 = st.Tensor(np.random.default_rng(3).standard_normal((3, 4)), requires_grad=True)
    x2 = st.Tensor(x1.data.copy(), requires_grad=True)
    f = lambda t: ((t @ w).tanh() * 2).sum(axis=1)  # noqa: E731
    f(x1).sum().backward()
    gw, gx = w.grad.copy(), x1.grad.copy()
    w.grad = None
    st.checkpoint(f, x2).sum().backward()
    np.testing.assert_allclose(w.grad, gw, atol=1e-12)
    np.testing.assert_allclose(x2.grad, gx, atol=1e-12)


def test_no_graph_is_stored_inside_the_checkpoint():
    x = st.Tensor(np.ones((2, 3)), requires_grad=True)
    y = st.checkpoint(lambda t: (t * 2).exp() + 1, x)
    assert y._parents == (x,)  # one node instead of the three inside the function
