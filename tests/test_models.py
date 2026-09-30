"""The GPT against an independent PyTorch implementation, KV-cache correctness, tokenizer, sampling."""
import numpy as np
import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as TF

import sutra as st
from sutra.models import GPT, GPTConfig, sample_next
from sutra.tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_dict

CFG = GPTConfig(vocab_size=23, block_size=12, n_layer=2, n_head=2, n_embd=16)


class TorchGPT(tnn.Module):
    """Reference implementation written directly in PyTorch with the same parameter names."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte, self.wpe = tnn.Embedding(c.vocab_size, c.n_embd), tnn.Embedding(c.block_size, c.n_embd)
        self.blocks = tnn.ModuleList()
        for _ in range(c.n_layer):
            b = tnn.Module()
            b.ln1, b.ln2 = tnn.LayerNorm(c.n_embd), tnn.LayerNorm(c.n_embd)
            b.attn = tnn.Module()
            b.attn.qkv, b.attn.proj = tnn.Linear(c.n_embd, 3 * c.n_embd), tnn.Linear(c.n_embd, c.n_embd)
            b.mlp = tnn.Module()
            b.mlp.fc, b.mlp.proj = tnn.Linear(c.n_embd, 4 * c.n_embd), tnn.Linear(4 * c.n_embd, c.n_embd)
            self.blocks.append(b)
        self.ln_f = tnn.LayerNorm(c.n_embd)

    def forward(self, idx):
        B, T = idx.shape
        H, C = self.c.n_head, self.c.n_embd
        x = self.wte(idx) + self.wpe(torch.arange(T))
        for b in self.blocks:
            q, k, v = b.attn.qkv(b.ln1(x)).split(C, dim=-1)
            q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
            y = TF.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
            x = x + b.attn.proj(y)
            x = x + b.mlp.proj(TF.gelu(b.mlp.fc(b.ln2(x)), approximate="tanh"))
        return self.ln_f(x) @ self.wte.weight.T


def make_pair():
    ours = GPT(CFG)
    ref = TorchGPT(CFG)
    ref.load_state_dict({k: torch.tensor(v) for k, v in ours.state_dict().items()})
    return ours, ref


def test_gpt_matches_pytorch_reference_logits_loss_and_every_gradient():
    ours, ref = make_pair()
    rng = np.random.default_rng(0)
    idx, tgt = rng.integers(0, CFG.vocab_size, (3, 10)), rng.integers(0, CFG.vocab_size, (3, 10))
    logits, loss = ours(idx, tgt)
    ref_logits = ref(torch.tensor(idx))
    ref_loss = TF.cross_entropy(ref_logits.reshape(-1, CFG.vocab_size), torch.tensor(tgt).reshape(-1))
    np.testing.assert_allclose(logits.data, ref_logits.detach().numpy(), atol=1e-10)
    loss.backward()
    ref_loss.backward()
    ref_params = dict(ref.named_parameters())
    for name, p in ours.named_parameters():
        np.testing.assert_allclose(p.grad, ref_params[name].grad.numpy(), atol=1e-10, err_msg=name)


def test_weight_tying_counts_embedding_once():
    m = GPT(CFG)
    names = [n for n, _ in m.named_parameters()]
    assert len(names) == len(set(names))
    assert not any("head" in n for n in names)


def test_kv_cache_gives_identical_logits_to_full_recompute():
    m = GPT(CFG).eval()
    seq = np.random.default_rng(1).integers(0, CFG.vocab_size, (2, 9))
    full, _ = m(seq)
    caches = [{} for _ in m.blocks]
    first, _ = m(seq[:, :4], kv_caches=caches)
    steps = [first.data]
    for t in range(4, 9):
        lg, _ = m(seq[:, t:t + 1], kv_caches=caches, pos_offset=t)
        steps.append(lg.data)
    np.testing.assert_allclose(np.concatenate(steps, axis=1), full.data, atol=1e-10)


def test_generate_cached_equals_uncached_and_is_deterministic():
    m = GPT(CFG)
    prompt = np.array([[1, 2, 3]])
    a = m.generate(prompt, 20, temperature=0.8, top_k=5, use_cache=True, seed=7)  # crosses block_size
    b = m.generate(prompt, 8, temperature=0.8, top_k=5, use_cache=False, seed=7)
    assert a.shape == (1, 23)
    np.testing.assert_array_equal(a[:, :11], b)
    np.testing.assert_array_equal(a, m.generate(prompt, 20, temperature=0.8, top_k=5, seed=7))


def test_sampling_filters():
    logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    assert sample_next(logits, temperature=0).item() == 0
    gen = np.random.default_rng(0)
    draws = [sample_next(logits, top_k=2, gen=gen).item() for _ in range(300)]
    assert set(draws) == {0, 1}
    draws = [sample_next(logits, top_p=0.7, gen=gen).item() for _ in range(300)]
    assert set(draws) == {0, 1}  # 0.5 < 0.7, so token 1 is needed to cross the threshold; token 2 is not


def test_gpt_can_memorise_a_sequence():
    """End-to-end learning check: a tiny GPT must drive loss on a fixed batch near zero."""
    st.manual_seed(0)
    m = GPT(CFG)
    opt = st.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    seq = np.random.default_rng(2).integers(0, CFG.vocab_size, (2, 13))
    for _ in range(150):
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


# ------------------------------------------------------------------ tokenizers

TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n"
        "First Citizen:\nYou are all resolved rather to die than to famish?\n") * 20


def test_bpe_roundtrip_including_unicode_and_unseen_text():
    tok = BPETokenizer.train(TEXT, 300)
    for s in [TEXT, "unseen wörds — ñ 日本語 🙂 123456", "", "   spaces\n\n\ttabs"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses_and_merges_are_meaningful():
    tok = BPETokenizer.train(TEXT, 320)
    ids = tok.encode(TEXT)
    assert len(ids) < 0.4 * len(TEXT.encode())
    assert "Citizen" in "".join(tok.token_strings()[256:])
    assert tok.vocab_size <= 320


def test_bpe_classic_example():
    # Wikipedia's example: "aaabdaaabac" -> first merge is "aa"
    tok = BPETokenizer.train("aaabdaaabac", 259)
    assert tok.merges[0] == (ord("a"), ord("a"))


def test_tokenizer_serialisation(tmp_path):
    tok = BPETokenizer.train(TEXT, 280)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    ct = CharTokenizer.from_text(TEXT)
    ct2 = tokenizer_from_dict(ct.to_dict())
    assert ct2.decode(ct2.encode("Speak, speak.")) == "Speak, speak."
