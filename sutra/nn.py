"""
Module system: parameter registration, train/eval modes, state dicts, and standard layers.
The API deliberately mirrors torch.nn so models port over line-for-line.
"""
from __future__ import annotations

import numpy as np

from . import functional as F
from . import init
from .tensor import Tensor, get_default_dtype


class Parameter(Tensor):
    """A Tensor that a Module registers as trainable."""

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return "Parameter(" + super().__repr__() + ")"


class Module:
    def __init__(self):
        object.__setattr__(self, "_params", [])
        object.__setattr__(self, "_modules", [])
        object.__setattr__(self, "_buffers", [])
        object.__setattr__(self, "training", True)

    # registration happens transparently on attribute assignment
    def __setattr__(self, name, value):
        if "_params" not in self.__dict__:
            raise RuntimeError("call super().__init__() before assigning attributes")
        for reg in (self._params, self._modules):
            if name in reg:
                reg.remove(name)
        if isinstance(value, Parameter):
            self._params.append(name)
        elif isinstance(value, Module):
            self._modules.append(name)
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        """Non-trainable state that belongs in the state dict (e.g. BatchNorm running stats)."""
        if name not in self._buffers:
            self._buffers.append(name)
        object.__setattr__(self, name, value)

    # ---------------------------------------------------------------- traversal
    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name in self._modules:
            yield from getattr(self, name).named_modules(f"{prefix}.{name}" if prefix else name)

    def children(self):
        return [getattr(self, n) for n in self._modules]

    def modules(self):
        return [m for _, m in self.named_modules()]

    def named_parameters(self):
        seen = set()
        for mname, m in self.named_modules():
            for pname in m._params:
                p = getattr(m, pname)
                if id(p) in seen:  # tied weights appear once
                    continue
                seen.add(id(p))
                yield (f"{mname}.{pname}" if mname else pname), p

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True):
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self):
        return self.train(False)

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict:
        sd = {}
        for mname, m in self.named_modules():
            pre = f"{mname}." if mname else ""
            for n in m._params:
                sd[pre + n] = getattr(m, n).data.copy()
            for n in m._buffers:
                sd[pre + n] = np.array(getattr(m, n), copy=True)
        return sd

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        own = {}
        for mname, m in self.named_modules():
            pre = f"{mname}." if mname else ""
            for n in m._params + m._buffers:
                own[pre + n] = (m, n)
        if strict:
            missing, unexpected = set(own) - set(sd), set(sd) - set(own)
            if missing or unexpected:
                raise KeyError(f"state_dict mismatch. missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for key, arr in sd.items():
            if key not in own:
                continue
            m, n = own[key]
            cur = getattr(m, n)
            arr = np.asarray(arr)
            if isinstance(cur, Tensor):
                if cur.shape != arr.shape:
                    raise ValueError(f"shape mismatch for {key}: {cur.shape} vs {arr.shape}")
                cur.data = arr.astype(cur.dtype).copy()
            else:
                cur[...] = arr

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        lines = [f"{type(self).__name__}({self.extra_repr()}"]
        for name in self._modules:
            sub = repr(getattr(self, name)).replace("\n", "\n  ")
            lines.append(f"  ({name}): {sub}")
        return lines[0] + ")" if len(lines) == 1 else "\n".join(lines) + "\n)"


# ------------------------------------------------------------------ containers

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def __iter__(self):
        return iter(self.children())

    def __getitem__(self, i):
        return self.children()[i]

    def __len__(self):
        return len(self._modules)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for m in modules:
            self.append(m)

    def append(self, m: Module):
        setattr(self, str(len(self._modules)), m)

    def __iter__(self):
        return iter(self.children())

    def __getitem__(self, i):
        return self.children()[i]

    def __len__(self):
        return len(self._modules)


# ------------------------------------------------------------------ layers

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        bound = 1.0 / np.sqrt(in_features)  # PyTorch's default init
        self.weight = Parameter(init.uniform((out_features, in_features), -bound, bound))
        self.bias = Parameter(init.uniform((out_features,), -bound, bound)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        self.weight = Parameter(init.uniform((out_channels, in_channels, kernel_size, kernel_size), -bound, bound))
        self.bias = Parameter(init.uniform((out_channels,), -bound, bound)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class _BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        dt = get_default_dtype()
        self.weight = Parameter(np.ones(num_features, dtype=dt)) if affine else None
        self.bias = Parameter(np.zeros(num_features, dtype=dt)) if affine else None
        self.register_buffer("running_mean", np.zeros(num_features, dtype=dt))
        self.register_buffer("running_var", np.ones(num_features, dtype=dt))

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training, self.momentum, self.eps)

    def extra_repr(self):
        return f"{self.num_features}"


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.dim, self.eps = dim, eps
        dt = get_default_dtype()
        self.weight = Parameter(np.ones(dim, dtype=dt))
        self.bias = Parameter(np.zeros(dim, dtype=dt)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f"{self.dim}"


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim = num_embeddings, dim
        self.weight = Parameter(init.normal((num_embeddings, dim), 0.0, 1.0))

    def forward(self, idx):
        return F.embedding(idx, self.weight)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.dim}"


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p}"


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride)


class AvgPool2d(MaxPool2d):
    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride)


class Flatten(Module):
    def forward(self, x):
        return x.flatten(1)


class Identity(Module):
    def forward(self, x):
        return x


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class MultiHeadAttention(Module):
    """Self-attention with a fused QKV projection, optional causal mask, and KV cache."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, causal: bool = True, bias: bool = True):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim, self.n_heads, self.causal, self.dropout = dim, n_heads, causal, dropout
        self.qkv = Linear(dim, 3 * dim, bias=bias)
        self.proj = Linear(dim, dim, bias=bias)
        self.resid_drop = Dropout(dropout)
        self.last_attention = None  # set when `keep_attention` is on (for visualisation)
        self.keep_attention = False

    def forward(self, x, kv_cache: dict | None = None):
        B, T, C = x.shape
        H, hd = self.n_heads, C // self.n_heads
        q, k, v = self.qkv(x).split(C, axis=-1)
        q, k, v = (t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        if kv_cache is not None:  # inference only: append to and read from the cache
            if "k" in kv_cache:
                k = Tensor(np.concatenate([kv_cache["k"], k.data], axis=2))
                v = Tensor(np.concatenate([kv_cache["v"], v.data], axis=2))
            kv_cache["k"], kv_cache["v"] = k.data, v.data
        y, att = F.scaled_dot_product_attention(q, k, v, causal=self.causal, dropout_p=self.dropout,
                                                training=self.training, return_weights=True)
        if self.keep_attention:
            self.last_attention = att
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_drop(self.proj(y))

    def extra_repr(self):
        return f"dim={self.dim}, heads={self.n_heads}, causal={self.causal}"
