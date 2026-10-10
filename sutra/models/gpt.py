"""
A decoder-only transformer built entirely on Sutra.

Default configuration is GPT-2 style: pre-LayerNorm blocks, fused-QKV causal self-attention, GELU MLP,
learned positional embeddings, weight-tied LM head and GPT-2 residual-scaled init. Three switches turn
it into a LLaMA-style model piece by piece: RMSNorm (`norm="rms"`), rotary embeddings (`pos="rope"`)
and a SwiGLU MLP (`mlp="swiglu"`). Sampling uses a KV cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import functional as F
from .. import init
from ..nn import Dropout, Embedding, LayerNorm, Linear, Module, ModuleList, MultiHeadAttention, RMSNorm
from ..checkpoint import checkpoint
from ..tensor import Tensor, is_grad_enabled, no_grad, rng


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    norm: str = "layer"     # "layer" | "rms"
    pos: str = "learned"    # "learned" | "rope"
    mlp: str = "gelu"       # "gelu" | "swiglu" | "none" (attention-only, as in interpretability studies)
    bias: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def llama(cls, **kw):
        """RMSNorm + RoPE + SwiGLU, no biases."""
        return cls(norm="rms", pos="rope", mlp="swiglu", bias=False, **kw)


def make_norm(cfg: GPTConfig):
    return RMSNorm(cfg.n_embd) if cfg.norm == "rms" else LayerNorm(cfg.n_embd, bias=cfg.bias)


class MLP(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(self.fc(x).gelu()))


class SwiGLU(Module):
    """proj(silu(gate(x)) * up(x)) with hidden = 8d/3, which matches the GELU MLP's 8d^2 weights."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        self.gate = Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up = Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


class Block(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = MultiHeadAttention(cfg.n_embd, cfg.n_head, cfg.dropout, causal=True, bias=cfg.bias,
                                       rope=cfg.pos == "rope")
        self.has_mlp = cfg.mlp != "none"
        if self.has_mlp:
            self.ln2 = make_norm(cfg)
            self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else MLP(cfg)

    def forward(self, x, kv_cache=None, pos_offset: int = 0):
        x = x + self.attn(self.ln1(x), kv_cache, pos_offset)
        return x + self.mlp(self.ln2(x)) if self.has_mlp else x


class GPT(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos == "learned":
            self.wpe = Embedding(cfg.block_size, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)
        self.blocks = ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.grad_checkpointing = False  # recompute each block in backward instead of storing activations
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                p.data = init.zeros(p.shape)
            elif p.ndim == 2:
                std = 0.02
                if name.endswith("proj.weight"):  # residual projections, scaled per GPT-2
                    std = 0.02 / np.sqrt(2 * self.cfg.n_layer)
                p.data = init.normal(p.shape, 0.0, std)

    def forward(self, idx, targets=None, kv_caches=None, pos_offset: int = 0, hidden: list | None = None):
        """Returns (logits, loss). If `hidden` is a list, the residual stream after the embedding and after
        every block is appended to it (used by the logit lens)."""
        idx = idx.data if isinstance(idx, Tensor) else np.asarray(idx)
        B, T = idx.shape
        assert pos_offset + T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.wte(idx)
        if self.cfg.pos == "learned":
            x = x + self.wpe(np.arange(pos_offset, pos_offset + T))
        x = self.drop(x)
        if hidden is not None:
            hidden.append(x)
        for i, block in enumerate(self.blocks):
            if self.grad_checkpointing and kv_caches is None and is_grad_enabled():
                x = checkpoint(lambda h, b=block: b(h, None, pos_offset), x)
            else:
                x = block(x, None if kv_caches is None else kv_caches[i], pos_offset)
            if hidden is not None:
                hidden.append(x)
        logits = self.unembed(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)

    def unembed(self, x):
        """Final norm + weight-tied LM head (the LM head reuses the token embedding matrix)."""
        return F.linear(self.ln_f(x), self.wte.weight)

    def param_groups(self, weight_decay: float):
        """Decay 2-D weight matrices only; never biases, norm gains or positional embeddings."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (decay if p.ndim >= 2 and "wpe" not in name else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    @no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, use_cache: bool = True, seed: int | None = None):
        """Autoregressive sampling. With `use_cache`, each step feeds one token and reuses cached keys/values."""
        was_training = self.training
        self.eval()
        gen = np.random.default_rng(seed) if seed is not None else rng()
        idx = np.array(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        caches, pos = None, 0
        for _ in range(max_new_tokens):
            if use_cache:
                if caches is None or pos >= self.cfg.block_size:
                    # (re)fill the cache from the most recent context window
                    ctx = idx[:, -self.cfg.block_size + 1:] if idx.shape[1] >= self.cfg.block_size else idx
                    caches = [{} for _ in self.blocks]
                    logits, _ = self(ctx, kv_caches=caches)
                    pos = ctx.shape[1]
                else:
                    logits, _ = self(idx[:, -1:], kv_caches=caches, pos_offset=pos)
                    pos += 1
            else:
                logits, _ = self(idx[:, -self.cfg.block_size:])
            nxt = sample_next(logits.data[:, -1, :], temperature, top_k, top_p, gen)
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
        self.train(was_training)
        return idx


def next_token_probs(logits: np.ndarray, temperature=1.0, top_k=None, top_p=None) -> np.ndarray:
    """The sampling distribution after temperature, top-k and nucleus filtering. logits: (B, V)."""
    z = logits.astype(np.float64) / temperature
    if top_k is not None:
        kth = np.sort(z, axis=-1)[:, -min(top_k, z.shape[-1])][:, None]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    if top_p is not None:  # nucleus sampling
        order = np.argsort(-p, axis=-1)
        sp = np.take_along_axis(p, order, axis=-1)
        keep = np.cumsum(sp, axis=-1) - sp < top_p
        sp = np.where(keep, sp, 0)
        p = np.zeros_like(p)
        np.put_along_axis(p, order, sp, axis=-1)
        p /= p.sum(axis=-1, keepdims=True)
    return p


def sample_next(logits: np.ndarray, temperature=1.0, top_k=None, top_p=None, gen=None) -> np.ndarray:
    gen = gen or rng()
    if temperature <= 0:
        return logits.argmax(axis=-1)
    p = next_token_probs(logits, temperature, top_k, top_p)
    return np.array([gen.choice(p.shape[-1], p=row) for row in p], dtype=np.int64)
