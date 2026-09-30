"""
Wall-clock comparison: one full training step (forward + backward + optimizer) in Sutra vs PyTorch,
same architecture, same batch, float32, CPU. Reports the median over timed repetitions.

    python experiments/benchmark.py
"""
import platform
import time

import numpy as np
from common import save, st

from sutra import optim
from sutra.models import GPT, MLP, DigitCNN, GPTConfig


def timeit(step, reps=20, warmup=3):
    for _ in range(warmup):
        step()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def sutra_step(model, x, y, loss_fn):
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        loss = loss_fn(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return step


def torch_models():
    import torch
    import torch.nn as tnn
    import torch.nn.functional as TF

    class TorchGPT(tnn.Module):
        def __init__(self, V, T, L, D, H):
            super().__init__()
            self.wte, self.wpe, self.H = tnn.Embedding(V, D), tnn.Embedding(T, D), H
            self.blocks = tnn.ModuleList([tnn.ModuleDict(dict(
                ln1=tnn.LayerNorm(D), qkv=tnn.Linear(D, 3 * D), proj=tnn.Linear(D, D), ln2=tnn.LayerNorm(D),
                fc=tnn.Linear(D, 4 * D), out=tnn.Linear(4 * D, D))) for _ in range(L)])
            self.ln_f = tnn.LayerNorm(D)

        def forward(self, idx):
            B, T = idx.shape
            x = self.wte(idx) + self.wpe(torch.arange(T))
            for b in self.blocks:
                q, k, v = b["qkv"](b["ln1"](x)).chunk(3, dim=-1)
                q, k, v = (t.view(B, T, self.H, -1).transpose(1, 2) for t in (q, k, v))
                y = TF.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
                x = x + b["proj"](y)
                x = x + b["out"](TF.gelu(b["fc"](b["ln2"](x)), approximate="tanh"))
            return self.ln_f(x) @ self.wte.weight.T

    cnn = tnn.Sequential(
        tnn.Conv2d(1, 16, 3, padding=1), tnn.BatchNorm2d(16), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Conv2d(16, 32, 3, padding=1), tnn.BatchNorm2d(32), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Flatten(), tnn.Linear(1568, 128), tnn.ReLU(), tnn.Dropout(0.25), tnn.Linear(128, 10))
    mlp = tnn.Sequential(tnn.Flatten(), tnn.Linear(784, 256), tnn.ReLU(), tnn.Linear(256, 256), tnn.ReLU(),
                         tnn.Linear(256, 10))
    return torch, mlp, cnn, TorchGPT


def main():
    rng = np.random.default_rng(0)
    torch, t_mlp, t_cnn, TorchGPT = torch_models()
    torch.set_default_dtype(torch.float32)
    x_img = rng.standard_normal((128, 1, 28, 28)).astype(np.float32)
    y_img = rng.integers(0, 10, 128)
    idx = rng.integers(0, 65, (32, 128))
    ce = lambda m, x, y: st.F.cross_entropy(m(x), y)  # noqa: E731
    gpt_cfg = dict(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128)

    cases = {
        "MLP 784-256-256-10 (batch 128)": (MLP([784, 256, 256, 10]), t_mlp, st.Tensor(x_img), y_img, ce),
        "DigitCNN (batch 128)": (DigitCNN(), t_cnn, st.Tensor(x_img), y_img, ce),
        "GPT 4x128, ctx 128 (batch 32)": (GPT(GPTConfig(**gpt_cfg)), TorchGPT(65, 128, 4, 128, 4), idx, idx,
                                          lambda m, x, y: m(x, y)[1]),
    }
    rows = []
    for name, (ours, theirs, x, y, loss_fn) in cases.items():
        ms_ours = timeit(sutra_step(ours, x, y, loss_fn))
        tx = torch.tensor(x.data if isinstance(x, st.Tensor) else x)
        ty = torch.tensor(y)
        topt = torch.optim.AdamW(theirs.parameters(), lr=1e-3)
        is_lm = "GPT" in name

        def tstep():
            out = theirs(tx)
            loss = torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), ty.reshape(-1))
            topt.zero_grad()
            loss.backward()
            topt.step()
        ms_torch = timeit(tstep)
        rows.append({"model": name, "sutra_ms": ms_ours, "torch_ms": ms_torch, "slowdown": ms_ours / ms_torch,
                     "language_model": is_lm})
        print(f"  {name:34s} sutra {ms_ours:7.1f} ms   torch {ms_torch:7.1f} ms   ratio {ms_ours / ms_torch:4.1f}x")
    save("benchmark", {"machine": f"{platform.machine()} / {platform.processor() or platform.platform()}",
                       "numpy": np.__version__, "torch": torch.__version__, "threads": torch.get_num_threads(),
                       "rows": rows})


if __name__ == "__main__":
    main()
