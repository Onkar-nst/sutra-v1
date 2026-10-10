"""
Neural-network operations.

Most ops here carry a hand-derived, fused backward pass instead of being composed
from primitives: fewer graph nodes, fewer temporaries, and numerically stabler
(softmax / cross-entropy / normalisation). Every one of them is verified against
PyTorch in tests/.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tensor import Tensor, _result, _to_array, rng

# ------------------------------------------------------------------ activations

def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    slope = np.where(x.data > 0, 1.0, negative_slope).astype(x.dtype)
    return _result(x.data * slope, (x,), lambda g: (g * slope,), "leaky_relu")


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    d = x.data
    e = np.exp(d - d.max(axis=axis, keepdims=True))
    out = e / e.sum(axis=axis, keepdims=True)

    def bw(g):
        return (out * (g - (g * out).sum(axis=axis, keepdims=True)),)
    return _result(out, (x,), bw, "softmax")


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    d = x.data
    z = d - d.max(axis=axis, keepdims=True)
    out = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))

    def bw(g):
        return (g - np.exp(out) * g.sum(axis=axis, keepdims=True),)
    return _result(out, (x,), bw, "log_softmax")


# ------------------------------------------------------------------ losses

def cross_entropy(logits: Tensor, target, ignore_index: int = -100,
                  label_smoothing: float = 0.0, reduction: str = "mean") -> Tensor:
    """Softmax cross-entropy. Class dimension is LAST: logits (..., C), target (...)."""
    x = logits.data
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    t = _to_array(target).astype(np.int64).reshape(-1)
    valid = t != ignore_index
    tt = np.where(valid, t, 0)
    rows = np.arange(len(t))

    z = x2 - x2.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss_i = -logp[rows, tt]
    eps = label_smoothing
    if eps > 0:
        loss_i = (1 - eps) * loss_i - eps * logp.mean(axis=1)
    loss_i = loss_i * valid
    n_valid = max(int(valid.sum()), 1)

    if reduction == "mean":
        out = loss_i.sum() / n_valid
    elif reduction == "sum":
        out = loss_i.sum()
    elif reduction == "none":
        out = loss_i.reshape(_to_array(target).shape)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    def bw(g):
        grad = np.exp(logp)
        if eps > 0:
            grad -= eps / C
            grad[rows, tt] -= 1 - eps
        else:
            grad[rows, tt] -= 1
        grad *= valid[:, None]
        if reduction == "mean":
            grad *= g / n_valid
        elif reduction == "sum":
            grad *= g
        else:
            grad *= g.reshape(-1, 1)
        return (grad.reshape(x.shape),)
    return _result(np.asarray(out, dtype=x.dtype), (logits,), bw, "cross_entropy")


def binary_cross_entropy_with_logits(logits: Tensor, target, reduction: str = "mean") -> Tensor:
    x = logits.data
    y = _to_array(target).astype(x.dtype)
    loss = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
    n = loss.size
    out = loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss

    def bw(g):
        p = np.exp(-np.logaddexp(0, -x))
        grad = p - y
        return (grad * (g / n if reduction == "mean" else g),)
    return _result(np.asarray(out, dtype=x.dtype), (logits,), bw, "bce_with_logits")


def mse_loss(pred: Tensor, target, reduction: str = "mean") -> Tensor:
    d = pred - (target if isinstance(target, Tensor) else Tensor(_to_array(target, pred.dtype)))
    sq = d * d
    return sq.mean() if reduction == "mean" else sq.sum() if reduction == "sum" else sq


# ------------------------------------------------------------------ layers

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """y = x W^T + b with W stored (out, in) like PyTorch. Batch dims are flattened for one GEMM."""
    xd, W = x.data, weight.data
    out = xd @ W.T
    if bias is not None:
        out = out + bias.data
    parents = (x, weight) if bias is None else (x, weight, bias)

    def bw(g):
        g2 = g.reshape(-1, g.shape[-1])
        gx = (g @ W) if x.requires_grad else None
        gw = (g2.T @ xd.reshape(-1, xd.shape[-1])) if weight.requires_grad else None
        if bias is None:
            return gx, gw
        return gx, gw, (g2.sum(axis=0) if bias.requires_grad else None)
    return _result(out, parents, bw, "linear")


def embedding(idx, weight: Tensor) -> Tensor:
    ids = _to_array(idx).astype(np.int64)
    D = weight.shape[1]

    def bw(g):
        gw = np.zeros_like(weight.data)
        np.add.at(gw, ids.reshape(-1), g.reshape(-1, D))
        return (gw,)
    return _result(weight.data[ids], (weight,), bw, "embedding")


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    mask = (rng().random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return _result(x.data * mask, (x,), lambda g: (g * mask,), "dropout")


def _norm_backward(g, xhat, rstd, w, axes):
    gxhat = g * w if w is not None else g
    return rstd * (gxhat - gxhat.mean(axis=axes, keepdims=True)
                   - xhat * (gxhat * xhat).mean(axis=axes, keepdims=True))


def layer_norm(x: Tensor, weight: Tensor | None = None, bias: Tensor | None = None,
               eps: float = 1e-5) -> Tensor:
    """LayerNorm over the last dimension, fused forward + backward."""
    d = x.data
    D = d.shape[-1]
    mu = d.mean(axis=-1, keepdims=True)
    xc = d - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
    xhat = xc * rstd
    out = xhat
    if weight is not None:
        out = out * weight.data
    if bias is not None:
        out = out + bias.data
    parents = tuple(t for t in (x, weight, bias) if t is not None)

    def bw(g):
        grads = [_norm_backward(g, xhat, rstd, None if weight is None else weight.data, -1)
                 if x.requires_grad else None]
        g2 = g.reshape(-1, D)
        if weight is not None:
            grads.append((g2 * xhat.reshape(-1, D)).sum(axis=0))
        if bias is not None:
            grads.append(g2.sum(axis=0))
        return tuple(grads)
    return _result(out, parents, bw, "layer_norm")


def batch_norm(x: Tensor, running_mean: np.ndarray | None, running_var: np.ndarray | None,
               weight: Tensor | None = None, bias: Tensor | None = None, training: bool = True,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """BatchNorm over all axes except the channel axis 1. Running stats are updated in place."""
    d = x.data
    axes = (0,) + tuple(range(2, d.ndim))
    bshape = [1] * d.ndim
    bshape[1] = d.shape[1]

    if training:
        mu = d.mean(axis=axes, keepdims=True)
        xc = d - mu
        var = (xc * xc).mean(axis=axes, keepdims=True)
        if running_mean is not None:
            n = d.size // d.shape[1]
            running_mean *= 1 - momentum
            running_mean += momentum * mu.reshape(-1)
            running_var *= 1 - momentum
            running_var += momentum * var.reshape(-1) * n / max(n - 1, 1)
    else:
        mu = running_mean.reshape(bshape)
        var = running_var.reshape(bshape)
        xc = d - mu
    rstd = (1.0 / np.sqrt(var + eps)).astype(d.dtype)
    xhat = xc * rstd
    w = None if weight is None else weight.data.reshape(bshape)
    out = xhat
    if w is not None:
        out = out * w
    if bias is not None:
        out = out + bias.data.reshape(bshape)
    parents = tuple(t for t in (x, weight, bias) if t is not None)

    def bw(g):
        if not x.requires_grad:
            gx = None
        elif training:
            gx = _norm_backward(g, xhat, rstd, w, axes)
        else:
            gx = g * rstd * (w if w is not None else 1)
        grads = [gx]
        if weight is not None:
            grads.append((g * xhat).sum(axis=axes))
        if bias is not None:
            grads.append(g.sum(axis=axes))
        return tuple(grads)
    return _result(out, parents, bw, "batch_norm")


# ------------------------------------------------------------------ convolution & pooling

def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None,
           stride: int = 1, padding: int = 0) -> Tensor:
    """2-D convolution (cross-correlation) via an im2col view + one tensordot GEMM."""
    N, C, H, W = x.shape
    O, _, kh, kw = weight.shape
    s, p = stride, padding
    xp = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p))) if p else x.data
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::s, ::s]  # N,C,OH,OW,kh,kw
    OH, OW = win.shape[2], win.shape[3]
    out = np.tensordot(win, weight.data, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
    if bias is not None:
        out = out + bias.data[None, :, None, None]
    out = np.ascontiguousarray(out)
    parents = (x, weight) if bias is None else (x, weight, bias)

    def bw(g):
        gx = gw = None
        if weight.requires_grad:
            gw = np.tensordot(g, win, axes=([0, 2, 3], [0, 2, 3]))  # O,C,kh,kw
        if x.requires_grad:
            gcols = np.tensordot(g, weight.data, axes=([1], [0]))  # N,OH,OW,C,kh,kw
            gxp = np.zeros(xp.shape, dtype=g.dtype)
            for i in range(kh):  # col2im: scatter each kernel tap back
                for j in range(kw):
                    gxp[:, :, i:i + s * OH:s, j:j + s * OW:s] += gcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
            gx = gxp[:, :, p:p + H, p:p + W] if p else gxp
        if bias is None:
            return gx, gw
        return gx, gw, (g.sum(axis=(0, 2, 3)) if bias.requires_grad else None)
    return _result(out, parents, bw, "conv2d")


def max_pool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    k, s = kernel_size, stride or kernel_size
    N, C = x.shape[:2]
    win = sliding_window_view(x.data, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    OH, OW = win.shape[2], win.shape[3]
    flat = win.reshape(N, C, OH, OW, k * k)
    am = flat.argmax(axis=-1)
    out = np.take_along_axis(flat, am[..., None], axis=-1)[..., 0]

    def bw(g):
        gx = np.zeros_like(x.data)
        for t in range(k * k):
            i, j = divmod(t, k)
            gx[:, :, i:i + s * OH:s, j:j + s * OW:s] += g * (am == t)
        return (gx,)
    return _result(out, (x,), bw, "max_pool2d")


def avg_pool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    k, s = kernel_size, stride or kernel_size
    win = sliding_window_view(x.data, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    OH, OW = win.shape[2], win.shape[3]
    out = win.mean(axis=(-2, -1))

    def bw(g):
        gx = np.zeros_like(x.data)
        for i in range(k):
            for j in range(k):
                gx[:, :, i:i + s * OH:s, j:j + s * OW:s] += g / (k * k)
        return (gx,)
    return _result(out, (x,), bw, "avg_pool2d")


# ------------------------------------------------------------------ attention

def causal_mask(Tq: int, Tk: int) -> np.ndarray:
    """True where attention is NOT allowed. Supports Tq < Tk (queries at the end, e.g. KV cache)."""
    return np.triu(np.ones((Tq, Tk), dtype=bool), k=1 + Tk - Tq)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
                                 mask=None, dropout_p: float = 0.0, training: bool = True,
                                 return_weights: bool = False):
    """
    softmax(q k^T / sqrt(d) + mask) v  with a fused, hand-derived backward:
        dV = P^T dO,  dP = dO V^T,  dS = P * (dP - rowsum(dP * P)),  dQ = dS K / sqrt(d),  dK = dS^T Q / sqrt(d)
    `mask` is boolean, True = blocked.
    """
    qd, kd, vd = q.data, k.data, v.data
    scale = np.asarray(1.0 / np.sqrt(qd.shape[-1]), dtype=qd.dtype)
    S = (qd @ np.swapaxes(kd, -1, -2)) * scale
    blocked = None
    if causal:
        blocked = causal_mask(S.shape[-2], S.shape[-1])
    if mask is not None:
        m = _to_array(mask).astype(bool)
        blocked = m if blocked is None else (blocked | m)
    if blocked is not None:
        S = np.where(blocked, -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    drop = None
    Pd = P
    if training and dropout_p > 0:
        drop = (rng().random(P.shape) >= dropout_p).astype(P.dtype) / (1 - dropout_p)
        Pd = P * drop
    out = Pd @ vd

    def bw(g):
        gv = np.swapaxes(Pd, -1, -2) @ g
        dP = g @ np.swapaxes(vd, -1, -2)
        if drop is not None:
            dP = dP * drop
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True)) * scale
        return (dS @ kd if q.requires_grad else None,
                np.swapaxes(dS, -1, -2) @ qd if k.requires_grad else None,
                gv if v.requires_grad else None)
    res = _result(out, (q, k, v), bw, "sdpa")
    return (res, P) if return_weights else res


# ------------------------------------------------------------------ modern (LLaMA-style) components

def silu(x: Tensor) -> Tensor:
    """SiLU / swish: x * sigmoid(x)."""
    d = x.data
    s = np.exp(-np.logaddexp(0, -d)).astype(d.dtype, copy=False)

    def bw(g):
        return (g * s * (1 + d * (1 - s)),)
    return _result(d * s, (x,), bw, "silu")


def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    """RMSNorm over the last dimension (Zhang & Sennrich, 2019): no mean-centring, no bias."""
    d = x.data
    D = d.shape[-1]
    r = 1.0 / np.sqrt((d * d).mean(axis=-1, keepdims=True) + eps)
    xhat = d * r
    out = xhat if weight is None else xhat * weight.data
    parents = (x,) if weight is None else (x, weight)

    def bw(g):
        gxhat = g if weight is None else g * weight.data
        gx = r * (gxhat - xhat * (gxhat * xhat).mean(axis=-1, keepdims=True)) if x.requires_grad else None
        if weight is None:
            return (gx,)
        return gx, (g.reshape(-1, D) * xhat.reshape(-1, D)).sum(axis=0)
    return _result(out, parents, bw, "rms_norm")


def rope_cache(head_dim: int, positions, base: float = 10000.0, dtype=np.float32):
    """cos/sin tables for rotary embeddings, shape (T, head_dim) in the 'rotate-half' layout."""
    inv = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=-1)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def _rotate_half(a: np.ndarray) -> np.ndarray:
    h = a.shape[-1] // 2
    return np.concatenate([-a[..., h:], a[..., :h]], axis=-1)


def _rotate_half_T(a: np.ndarray) -> np.ndarray:  # adjoint of _rotate_half
    h = a.shape[-1] // 2
    return np.concatenate([a[..., h:], -a[..., :h]], axis=-1)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """
    Rotary position embedding (Su et al., 2021) on x (..., T, head_dim):  y = x*cos + rotate_half(x)*sin.
    The map is a rotation, so the backward pass is the inverse rotation:  gx = g*cos + rotate_half^T(g*sin).
    """
    d = x.data
    out = d * cos + _rotate_half(d) * sin

    def bw(g):
        return (g * cos + _rotate_half_T(g * sin),)
    return _result(out, (x,), bw, "rope")
