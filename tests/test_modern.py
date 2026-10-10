"""LLaMA-style components (RMSNorm, RoPE, SiLU/SwiGLU), head ablation hooks and the logit-lens hidden states."""
import numpy as np
import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as TF
from conftest import check, randn

import sutra as st
from sutra import functional as F
from sutra.models import GPT, GPTConfig


def test_silu():
    check(F.silu, TF.silu, randn(4, 7) * 3)


def test_rms_norm():
    check(lambda x, w: F.rms_norm(x, w), lambda x, w: TF.rms_norm(x, (6,), w, eps=1e-5),
          randn(2, 3, 6) * 2 + 0.5, randn(6, seed=1))
    check(lambda x: F.rms_norm(x), lambda x: TF.rms_norm(x, (6,), eps=1e-5), randn(5, 6))


def torch_rope(x, positions, base=10000.0):
    """Independent reference: rotate each (x_i, x_{i+d/2}) pair by angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = torch.tensor(positions, dtype=torch.float64)[:, None] * inv[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


@pytest.mark.parametrize("offset", [0, 5])
def test_rope_matches_reference(offset):
    pos = np.arange(offset, offset + 7)
    cos, sin = F.rope_cache(8, pos, dtype=np.float64)
    check(lambda x: F.apply_rope(x, cos, sin), lambda x: torch_rope(x, pos), randn(2, 3, 7, 8))


def test_rope_scores_depend_only_on_relative_position():
    rng = np.random.default_rng(0)
    q, k = rng.standard_normal(16), rng.standard_normal(16)

    def score(m, n):
        cq, sq = F.rope_cache(16, [m], dtype=np.float64)
        ck, sk = F.rope_cache(16, [n], dtype=np.float64)
        qr = F.apply_rope(st.Tensor(q[None]), cq, sq).data[0]
        kr = F.apply_rope(st.Tensor(k[None]), ck, sk).data[0]
        return qr @ kr
    assert score(3, 1) == pytest.approx(score(10, 8), abs=1e-10)
    assert score(3, 1) == pytest.approx(score(50, 48), abs=1e-10)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-6)


CFG = GPTConfig.llama(vocab_size=19, block_size=12, n_layer=2, n_head=2, n_embd=24)


class TorchLlama(tnn.Module):
    """Reference LLaMA-style decoder in plain PyTorch with the same parameter names."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        h = int(8 * c.n_embd / 3)
        self.wte = tnn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = tnn.ModuleList()
        for _ in range(c.n_layer):
            b = tnn.Module()
            b.ln1, b.ln2 = tnn.RMSNorm(c.n_embd, eps=1e-5), tnn.RMSNorm(c.n_embd, eps=1e-5)
            b.attn = tnn.Module()
            b.attn.qkv = tnn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
            b.attn.proj = tnn.Linear(c.n_embd, c.n_embd, bias=False)
            b.mlp = tnn.Module()
            b.mlp.gate, b.mlp.up = tnn.Linear(c.n_embd, h, bias=False), tnn.Linear(c.n_embd, h, bias=False)
            b.mlp.proj = tnn.Linear(h, c.n_embd, bias=False)
            self.blocks.append(b)
        self.ln_f = tnn.RMSNorm(c.n_embd, eps=1e-5)

    def forward(self, idx):
        B, T = idx.shape
        H, C = self.c.n_head, self.c.n_embd
        x = self.wte(idx)
        for b in self.blocks:
            q, k, v = b.attn.qkv(b.ln1(x)).split(C, dim=-1)
            q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
            q, k = torch_rope(q, list(range(T))), torch_rope(k, list(range(T)))
            y = TF.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
            x = x + b.attn.proj(y)
            h = b.ln2(x)
            x = x + b.mlp.proj(TF.silu(b.mlp.gate(h)) * b.mlp.up(h))
        return self.ln_f(x) @ self.wte.weight.T


def test_llama_style_gpt_matches_pytorch_reference():
    ours = GPT(CFG)
    ref = TorchLlama(CFG)
    ref.load_state_dict({k: torch.tensor(v) for k, v in ours.state_dict().items()})
    rng = np.random.default_rng(0)
    idx, tgt = rng.integers(0, CFG.vocab_size, (3, 11)), rng.integers(0, CFG.vocab_size, (3, 11))
    logits, loss = ours(idx, tgt)
    ref_logits = ref(torch.tensor(idx))
    np.testing.assert_allclose(logits.data, ref_logits.detach().numpy(), atol=1e-10)
    loss.backward()
    TF.cross_entropy(ref_logits.reshape(-1, CFG.vocab_size), torch.tensor(tgt).reshape(-1)).backward()
    ref_params = dict(ref.named_parameters())
    for name, p in ours.named_parameters():
        np.testing.assert_allclose(p.grad, ref_params[name].grad.numpy(), atol=1e-10, err_msg=name)


def test_llama_style_kv_cache_matches_full_forward():
    m = GPT(CFG).eval()
    seq = np.random.default_rng(1).integers(0, CFG.vocab_size, (2, 10))
    full, _ = m(seq)
    caches = [{} for _ in m.blocks]
    steps = [m(seq[:, :3], kv_caches=caches)[0].data]
    for t in range(3, 10):
        steps.append(m(seq[:, t:t + 1], kv_caches=caches, pos_offset=t)[0].data)
    np.testing.assert_allclose(np.concatenate(steps, axis=1), full.data, atol=1e-10)


def test_swiglu_parameter_count_matches_gelu_mlp():
    d = 96
    gelu = GPT(GPTConfig(n_embd=d, n_layer=1, n_head=4, bias=False))
    swi = GPT(GPTConfig(n_embd=d, n_layer=1, n_head=4, bias=False, mlp="swiglu"))
    g = sum(p.size for n, p in gelu.named_parameters() if ".mlp." in n)
    s = sum(p.size for n, p in swi.named_parameters() if ".mlp." in n)
    assert abs(g - s) / g < 0.01


def test_head_mask_ablates_exactly_one_head():
    m = GPT(GPTConfig(vocab_size=19, block_size=12, n_layer=1, n_head=3, n_embd=24)).eval()
    idx = np.random.default_rng(2).integers(0, 19, (2, 8))
    attn = m.blocks[0].attn
    base, _ = m(idx)
    attn.head_mask = np.ones(3)
    same, _ = m(idx)
    np.testing.assert_allclose(same.data, base.data, atol=1e-12)
    attn.head_mask = np.array([1.0, 0.0, 1.0])
    ablated, _ = m(idx)
    assert not np.allclose(ablated.data, base.data)
    # by hand: zero head 1's slice of the attention output before the output projection
    attn.head_mask = None
    x = m.wte(idx) + m.wpe(np.arange(8))
    h = m.blocks[0].ln1(x)
    q, k, v = attn.qkv(h).split(24, axis=-1)
    q, k, v = (t.reshape(2, 8, 3, 8).transpose(0, 2, 1, 3) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, causal=True).data
    y[:, 1] = 0
    y = attn.proj(st.Tensor(y.transpose(0, 2, 1, 3).reshape(2, 8, 24)))
    x2 = x + y
    x2 = x2 + m.blocks[0].mlp(m.blocks[0].ln2(x2))
    np.testing.assert_allclose(m.unembed(x2).data, ablated.data, atol=1e-10)


def test_hidden_states_for_logit_lens():
    m = GPT(GPTConfig(vocab_size=19, block_size=12, n_layer=3, n_head=2, n_embd=16)).eval()
    idx = np.random.default_rng(3).integers(0, 19, (1, 6))
    hidden = []
    logits, _ = m(idx, hidden=hidden)
    assert len(hidden) == 4
    np.testing.assert_allclose(m.unembed(hidden[-1]).data, logits.data, atol=1e-12)
