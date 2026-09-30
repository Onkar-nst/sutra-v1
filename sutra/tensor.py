"""
Sutra's core: an n-dimensional Tensor with reverse-mode automatic differentiation.

Every differentiable operation produces a new Tensor that remembers its parents and
a closure `_backward(g) -> tuple[grad_for_each_parent]`. Calling `.backward()` on a
scalar walks the graph in reverse topological order and accumulates gradients
(the vector-Jacobian products) into the leaves' `.grad`.

Gradients are stored as plain NumPy arrays; data lives in `.data`.
"""
from __future__ import annotations

import functools
import numpy as np

_DEFAULT_DTYPE = np.float32
_GRAD_ENABLED = True


# --------------------------------------------------------------------------- config

def set_default_dtype(dtype) -> None:
    """Set the dtype used for tensors built from Python floats / factory functions."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


def get_default_dtype():
    return _DEFAULT_DTYPE


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class no_grad:
    """Context manager / decorator that disables graph construction."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper


# --------------------------------------------------------------------------- helpers

def _to_array(x, dtype=None) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.data
    if isinstance(x, np.ndarray):
        return x if dtype is None else x.astype(dtype, copy=False)
    arr = np.asarray(x)
    if dtype is not None:
        return arr.astype(dtype)
    if arr.dtype.kind == "f":
        return arr.astype(_DEFAULT_DTYPE)
    return arr


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `g` down to `shape`, undoing NumPy broadcasting."""
    if g.shape == shape:
        return g
    ndiff = g.ndim - len(shape)
    if ndiff > 0:
        g = g.sum(axis=tuple(range(ndiff)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g.reshape(shape)


def _norm_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def _expand_reduced(g: np.ndarray, axes: tuple, keepdims: bool) -> np.ndarray:
    """Re-insert reduced axes so `g` broadcasts against the reduction's input."""
    if keepdims:
        return g
    for a in axes:
        g = np.expand_dims(g, a)
    return g


def _is_basic_index(idx) -> bool:
    """True when indexing can't hit the same element twice (so plain assignment is a valid adjoint)."""
    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        return True
    if not isinstance(idx, tuple):
        idx = (idx,)
    return all(isinstance(i, (int, np.integer, slice)) or i is None or i is Ellipsis for i in idx)


def _result(data, parents, backward, op) -> "Tensor":
    """Wrap `data` as the output of an op, recording the graph only when needed."""
    out = Tensor(data)
    if _GRAD_ENABLED and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
        out._op = op
    return out


# --------------------------------------------------------------------------- Tensor

class Tensor:
    __array_priority__ = 1000  # make `ndarray op Tensor` dispatch to Tensor's reflected ops

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        self.data = _to_array(data, dtype)
        if requires_grad and self.data.dtype.kind != "f":
            raise TypeError("only floating point tensors can require gradients")
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: tuple = ()
        self._backward = None
        self._op = ""
        self._retain = False

    # ---------------------------------------------------------------- basics
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def is_leaf(self) -> bool:
        return self._backward is None

    def numel(self) -> int:
        return self.data.size

    def item(self):
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self):
        return self.data.tolist()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        body = np.array2string(self.data, precision=4, separator=", ", prefix="tensor(")
        extra = f", grad_fn=<{self._op}>" if self._op else (", requires_grad=True" if self.requires_grad else "")
        return f"tensor({body}{extra})"

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def retain_grad(self) -> "Tensor":
        """Keep `.grad` on this non-leaf tensor after backward (like torch)."""
        self._retain = True
        return self

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        self.requires_grad = flag
        return self

    def zero_grad(self) -> None:
        self.grad = None

    def astype(self, dtype) -> "Tensor":
        a = self
        src = a.dtype

        def bw(g):
            return (g.astype(src),)
        return _result(a.data.astype(dtype), (a,), bw, "astype")

    def _lift(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        if isinstance(other, np.ndarray):
            return Tensor(other)
        dtype = self.dtype if self.dtype.kind == "f" else None
        return Tensor(np.asarray(other, dtype=dtype))

    # ---------------------------------------------------------------- autograd engine
    def backward(self, grad=None, retain_graph: bool = False) -> None:
        """Reverse-mode differentiation from this tensor into all reachable leaves."""
        if not self.requires_grad:
            raise RuntimeError("tensor does not require grad and has no grad_fn")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = np.ones_like(self.data)
        grad = _to_array(grad, self.dtype)

        # iterative DFS topological sort (deep graphs would blow Python's recursion limit)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in visited:
                    stack.append((p, False))

        grads = {id(self): grad}
        for node in reversed(topo):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf: accumulate
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            if node._retain:
                node.grad = g.copy() if node.grad is None else node.grad + g
            parent_grads = node._backward(g)
            for p, pg in zip(node._parents, parent_grads):
                if pg is None or not p.requires_grad:
                    continue
                prev = grads.get(id(p))
                grads[id(p)] = pg if prev is None else prev + pg
            if not retain_graph:
                node._backward = None
                node._parents = ()

    # ---------------------------------------------------------------- arithmetic
    def __add__(self, other):
        a, b = self, self._lift(other)

        def bw(g):
            return (_unbroadcast(g, a.shape) if a.requires_grad else None,
                    _unbroadcast(g, b.shape) if b.requires_grad else None)
        return _result(a.data + b.data, (a, b), bw, "add")

    def __radd__(self, other):
        return self._lift(other) + self

    def __sub__(self, other):
        a, b = self, self._lift(other)

        def bw(g):
            return (_unbroadcast(g, a.shape) if a.requires_grad else None,
                    _unbroadcast(-g, b.shape) if b.requires_grad else None)
        return _result(a.data - b.data, (a, b), bw, "sub")

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        a, b = self, self._lift(other)

        def bw(g):
            return (_unbroadcast(g * b.data, a.shape) if a.requires_grad else None,
                    _unbroadcast(g * a.data, b.shape) if b.requires_grad else None)
        return _result(a.data * b.data, (a, b), bw, "mul")

    def __rmul__(self, other):
        return self._lift(other) * self

    def __truediv__(self, other):
        a, b = self, self._lift(other)

        def bw(g):
            return (_unbroadcast(g / b.data, a.shape) if a.requires_grad else None,
                    _unbroadcast(-g * a.data / (b.data * b.data), b.shape) if b.requires_grad else None)
        return _result(a.data / b.data, (a, b), bw, "div")

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __neg__(self):
        a = self
        return _result(-a.data, (a,), lambda g: (-g,), "neg")

    def __pow__(self, p):
        if isinstance(p, Tensor):
            return (p * self.log()).exp()
        a = self

        def bw(g):
            return (g * p * a.data ** (p - 1),)
        return _result(a.data ** p, (a,), bw, "pow")

    def __rpow__(self, base):
        a = self
        out = np.asarray(base, dtype=a.dtype) ** a.data

        log_base = float(np.log(base))

        def bw(g):
            return (g * out * log_base,)
        return _result(out, (a,), bw, "rpow")

    def __matmul__(self, other):
        return matmul(self, self._lift(other))

    def __rmatmul__(self, other):
        return matmul(self._lift(other), self)

    # comparisons produce non-differentiable boolean tensors
    def __lt__(self, other): return Tensor(self.data < _to_array(other))
    def __le__(self, other): return Tensor(self.data <= _to_array(other))
    def __gt__(self, other): return Tensor(self.data > _to_array(other))
    def __ge__(self, other): return Tensor(self.data >= _to_array(other))
    def eq(self, other): return Tensor(self.data == _to_array(other))
    def ne(self, other): return Tensor(self.data != _to_array(other))
    __hash__ = object.__hash__

    # ---------------------------------------------------------------- elementwise
    def exp(self):
        a = self
        out = np.exp(a.data)
        return _result(out, (a,), lambda g: (g * out,), "exp")

    def log(self):
        a = self
        return _result(np.log(a.data), (a,), lambda g: (g / a.data,), "log")

    def sqrt(self):
        a = self
        out = np.sqrt(a.data)
        return _result(out, (a,), lambda g: (g * 0.5 / out,), "sqrt")

    def tanh(self):
        a = self
        out = np.tanh(a.data)
        return _result(out, (a,), lambda g: (g * (1 - out * out),), "tanh")

    def sigmoid(self):
        a = self
        out = np.exp(-np.logaddexp(0, -a.data)).astype(a.dtype, copy=False)
        return _result(out, (a,), lambda g: (g * out * (1 - out),), "sigmoid")

    def relu(self):
        a = self
        mask = a.data > 0
        return _result(a.data * mask, (a,), lambda g: (g * mask,), "relu")

    def gelu(self):
        """GELU, tanh approximation (as used in GPT-2)."""
        a = self
        x = a.data
        c = float(np.sqrt(2.0 / np.pi))
        x2 = x * x  # x*x*x instead of x**3: generic pow is ~10x slower in NumPy
        t = x2 * 0.044715
        t += 1
        t *= x
        t *= c
        np.tanh(t, out=t)
        out = t + 1
        out *= x
        out *= 0.5

        def bw(g):
            # d/dx = 0.5(1+t) + 0.5 x (1-t^2) c (1 + 3*0.044715 x^2)
            d = 1 - t * t
            d *= x
            d *= (x2 * (3 * 0.044715) + 1) * (0.5 * c)
            d += 0.5 * (1 + t)
            d *= g
            return (d,)
        return _result(out, (a,), bw, "gelu")

    def abs(self):
        a = self
        return _result(np.abs(a.data), (a,), lambda g: (g * np.sign(a.data),), "abs")

    def clamp(self, min=None, max=None):
        a = self
        out = np.clip(a.data, min, max)
        mask = np.ones_like(a.data, dtype=bool)
        if min is not None:
            mask &= a.data >= min
        if max is not None:
            mask &= a.data <= max
        return _result(out, (a,), lambda g: (g * mask,), "clamp")

    def masked_fill(self, mask, value):
        a = self
        m = _to_array(mask).astype(bool)
        out = np.where(m, np.asarray(value, dtype=a.dtype), a.data)
        return _result(out, (a,), lambda g: (_unbroadcast(np.where(m, 0, g), a.shape),), "masked_fill")

    # ---------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        a = self
        axes = _norm_axes(axis, a.ndim)

        def bw(g):
            return (np.broadcast_to(_expand_reduced(g, axes, keepdims), a.shape),)
        return _result(a.data.sum(axis=axes, keepdims=keepdims), (a,), bw, "sum")

    def mean(self, axis=None, keepdims=False):
        axes = _norm_axes(axis, self.ndim)
        n = int(np.prod([self.shape[i] for i in axes])) if axes else 1
        return self.sum(axis=axes, keepdims=keepdims) / n

    def var(self, axis=None, keepdims=False, unbiased=False):
        axes = _norm_axes(axis, self.ndim)
        n = int(np.prod([self.shape[i] for i in axes]))
        d = self - self.mean(axis=axes, keepdims=True)
        return (d * d).sum(axis=axes, keepdims=keepdims) / (n - 1 if unbiased else n)

    def _extremum(self, axis, keepdims, fn, name):
        a = self
        axes = _norm_axes(axis, a.ndim)
        full = fn(a.data, axis=axes, keepdims=True)

        def bw(g):
            mask = (a.data == full)
            # ties share the gradient equally (matches torch.amax/amin)
            return (mask * _expand_reduced(g, axes, keepdims) / mask.sum(axis=axes, keepdims=True),)
        out = full if keepdims else full.reshape([s for i, s in enumerate(a.shape) if i not in axes])
        return _result(out, (a,), bw, name)

    def max(self, axis=None, keepdims=False):
        return self._extremum(axis, keepdims, np.max, "max")

    def min(self, axis=None, keepdims=False):
        return self._extremum(axis, keepdims, np.min, "min")

    def logsumexp(self, axis=-1, keepdims=False):
        a = self
        axes = _norm_axes(axis, a.ndim)
        m = a.data.max(axis=axes, keepdims=True)
        m = np.where(np.isfinite(m), m, 0)
        s = np.exp(a.data - m).sum(axis=axes, keepdims=True)
        full = m + np.log(s)

        def bw(g):
            return (_expand_reduced(g, axes, keepdims) * np.exp(a.data - full),)
        out = full if keepdims else np.squeeze(full, axis=axes)
        return _result(out, (a,), bw, "logsumexp")

    def argmax(self, axis=None, keepdims=False):
        return Tensor(np.argmax(self.data, axis=axis, keepdims=keepdims))

    def argmin(self, axis=None, keepdims=False):
        return Tensor(np.argmin(self.data, axis=axis, keepdims=keepdims))

    # ---------------------------------------------------------------- shape ops
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return _result(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),), "reshape")

    view = reshape

    def flatten(self, start_dim=0, end_dim=-1):
        s = self.shape
        start, end = start_dim % self.ndim, end_dim % self.ndim
        new = s[:start] + (int(np.prod(s[start:end + 1])),) + s[end + 1:]
        return self.reshape(new)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        a = self
        axes = axes or tuple(reversed(range(a.ndim)))
        inv = np.argsort(axes)
        return _result(a.data.transpose(axes), (a,), lambda g: (g.transpose(inv),), "transpose")

    permute = transpose

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, a1, a2):
        axes = list(range(self.ndim))
        axes[a1], axes[a2] = axes[a2], axes[a1]
        return self.transpose(tuple(axes))

    def squeeze(self, axis=None):
        if axis is None:
            new = tuple(s for s in self.shape if s != 1)
        else:
            axes = _norm_axes(axis, self.ndim)
            new = tuple(s for i, s in enumerate(self.shape) if i not in axes)
        return self.reshape(new)

    def unsqueeze(self, axis):
        axis = axis % (self.ndim + 1)
        return self.reshape(self.shape[:axis] + (1,) + self.shape[axis:])

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return _result(np.broadcast_to(a.data, shape), (a,), lambda g: (_unbroadcast(g, a.shape),), "expand")

    broadcast_to = expand

    def __getitem__(self, idx):
        a = self
        if isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        basic = _is_basic_index(idx)

        def bw(g):
            grad = np.zeros_like(a.data)
            if basic:
                grad[idx] = g
            else:
                np.add.at(grad, idx, g)
            return (grad,)
        return _result(a.data[idx], (a,), bw, "getitem")

    def split(self, sizes, axis=-1):
        """Split into chunks of the given sizes (int -> equal chunks of that size)."""
        axis = axis % self.ndim
        n = self.shape[axis]
        if isinstance(sizes, int):
            sizes = [sizes] * (n // sizes)
        out, start = [], 0
        for s in sizes:
            sl = [slice(None)] * self.ndim
            sl[axis] = slice(start, start + s)
            out.append(self[tuple(sl)])
            start += s
        return out

    def chunk(self, n, axis=-1):
        return self.split(self.shape[axis] // n, axis)


# --------------------------------------------------------------------------- free functions

def tensor(data, requires_grad=False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    ad, bd = a.data, b.data
    a1, b1 = ad.ndim == 1, bd.ndim == 1
    A = ad[None, :] if a1 else ad
    B = bd[:, None] if b1 else bd
    out = A @ B
    if a1:
        out = np.squeeze(out, axis=-2)
    if b1:
        out = np.squeeze(out, axis=-1)

    def bw(g):
        G = g
        if b1:
            G = G[..., None]
        if a1:
            G = np.expand_dims(G, -2)
        ga = gb = None
        if a.requires_grad:
            ga = _unbroadcast(G @ np.swapaxes(B, -1, -2), A.shape).reshape(ad.shape)
        if b.requires_grad:
            gb = _unbroadcast(np.swapaxes(A, -1, -2) @ G, B.shape).reshape(bd.shape)
        return ga, gb
    return _result(out, (a, b), bw, "matmul")


def where(cond, a, b) -> Tensor:
    c = _to_array(cond).astype(bool)
    a = a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=_DEFAULT_DTYPE))
    b = b if isinstance(b, Tensor) else Tensor(np.asarray(b, dtype=a.dtype))

    def bw(g):
        return (_unbroadcast(np.where(c, g, 0), a.shape) if a.requires_grad else None,
                _unbroadcast(np.where(c, 0, g), b.shape) if b.requires_grad else None)
    return _result(np.where(c, a.data, b.data), (a, b), bw, "where")


def maximum(a, b) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = a._lift(b)
    gt, eq = a.data > b.data, a.data == b.data

    def bw(g):
        wa = gt + 0.5 * eq
        return (_unbroadcast(g * wa, a.shape) if a.requires_grad else None,
                _unbroadcast(g * (1 - wa), b.shape) if b.requires_grad else None)
    return _result(np.maximum(a.data, b.data), (a, b), bw, "maximum")


def minimum(a, b) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor(a)
    return -maximum(-a, -a._lift(b))


def cat(tensors, axis=0) -> Tensor:
    tensors = list(tensors)
    axis = axis % tensors[0].ndim
    sizes = [t.shape[axis] for t in tensors]
    cuts = np.cumsum(sizes)[:-1]

    def bw(g):
        parts = np.split(g, cuts, axis=axis)
        return tuple(p if t.requires_grad else None for p, t in zip(parts, tensors))
    return _result(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), bw, "cat")


def stack(tensors, axis=0) -> Tensor:
    tensors = list(tensors)
    axis = axis % (tensors[0].ndim + 1)

    def bw(g):
        return tuple(np.take(g, i, axis=axis) if t.requires_grad else None for i, t in enumerate(tensors))
    return _result(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), bw, "stack")


def pad(x: Tensor, pad_width, value=0.0) -> Tensor:
    """Constant padding, NumPy `pad_width` convention: ((before, after), ...)."""
    pw = [(p, p) if isinstance(p, int) else tuple(p) for p in pad_width]
    sl = tuple(slice(b, x.shape[i] + b) for i, (b, _) in enumerate(pw))
    return _result(np.pad(x.data, pw, constant_values=value), (x,), lambda g: (g[sl],), "pad")


# factories ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)


def manual_seed(seed: int) -> None:
    """Seed Sutra's global random generator (initialisation, dropout, shuffling)."""
    global _rng
    _rng = np.random.default_rng(seed)


def rng() -> np.random.Generator:
    return _rng


def zeros(*shape, requires_grad=False, dtype=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(np.zeros(shape, dtype=dtype or _DEFAULT_DTYPE), requires_grad)


def ones(*shape, requires_grad=False, dtype=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(np.ones(shape, dtype=dtype or _DEFAULT_DTYPE), requires_grad)


def randn(*shape, requires_grad=False, dtype=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.standard_normal(shape).astype(dtype or _DEFAULT_DTYPE), requires_grad)


def rand(*shape, requires_grad=False, dtype=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.random(shape).astype(dtype or _DEFAULT_DTYPE), requires_grad)


def arange(*args, dtype=None):
    return Tensor(np.arange(*args, dtype=dtype))
