"""
Sutra — a deep-learning framework written from scratch on NumPy.

    import sutra as st
    x = st.randn(4, 3, requires_grad=True)
    y = (x @ x.T).relu().sum()
    y.backward()
    x.grad  # dy/dx
"""
from . import functional, init, nn, optim
from .tensor import (Tensor, arange, cat, get_default_dtype, is_grad_enabled, manual_seed, matmul, maximum,
                     minimum, no_grad, ones, pad, rand, randn, rng, set_default_dtype, stack, tensor, where, zeros)

F = functional

__version__ = "1.0.0"

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn", "rand", "arange", "cat", "stack", "where", "maximum", "minimum",
    "matmul", "pad", "no_grad", "is_grad_enabled", "manual_seed", "rng", "set_default_dtype", "get_default_dtype",
    "nn", "optim", "functional", "F", "init",
]
