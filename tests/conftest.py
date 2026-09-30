import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sutra as st  # noqa: E402

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def float64_everywhere():
    """Compare in double precision so tolerances measure correctness, not rounding."""
    prev = st.get_default_dtype()
    st.set_default_dtype(np.float64)
    st.manual_seed(0)
    torch.manual_seed(0)
    yield
    st.set_default_dtype(prev)


def check(fn_st, fn_th, *arrays, atol=1e-7, rtol=1e-6, seed=0):
    """
    Run the same computation in Sutra and PyTorch on identical float64 inputs, backprop the SAME
    random upstream gradient through both, and require outputs and every input gradient to match.
    """
    xs_st = [st.Tensor(a.copy(), requires_grad=True) for a in arrays]
    xs_th = [torch.tensor(a.copy(), requires_grad=True) for a in arrays]
    out_st, out_th = fn_st(*xs_st), fn_th(*xs_th)
    np.testing.assert_allclose(out_st.data, out_th.detach().numpy(), atol=atol, rtol=rtol, err_msg="forward")
    g = np.random.default_rng(seed).standard_normal(out_st.shape)
    out_st.backward(g)
    out_th.backward(torch.tensor(g))
    for i, (a, b) in enumerate(zip(xs_st, xs_th)):
        if b.grad is None:
            assert a.grad is None or np.allclose(a.grad, 0), f"input {i}: unexpected grad"
            continue
        assert a.grad is not None, f"input {i}: missing grad"
        np.testing.assert_allclose(a.grad, b.grad.numpy(), atol=atol, rtol=rtol, err_msg=f"grad of input {i}")
    return out_st


def randn(*shape, seed=None):
    return np.random.default_rng(seed if seed is not None else hash(shape) % 2**32).standard_normal(shape)
