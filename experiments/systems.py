"""
Systems experiments: speculative decoding and gradient checkpointing. Run on an otherwise idle machine,
since both measure wall-clock time.

  speculative    The 2.7M flagship GPT as the target; the 28k-800k models from the scaling study as drafts.
                 For each draft and draft length k: acceptance rate alpha, tokens per target forward pass,
                 and the wall-clock speedup over plain decoding (same KV-cache machinery, same sampling).
  checkpointing  Peak memory (tracemalloc) and step time of forward + backward with and without per-block
                 gradient checkpointing, for GPTs of increasing depth.

    python experiments/systems.py --study speculative|checkpointing|all
"""
import argparse
import gc
import time
import tracemalloc

import numpy as np
from common import ROOT, save, st

from sutra import serialize
from sutra.models import GPT, GPTConfig
from sutra.models.speculative import generate_speculative, generate_target_only
from sutra.tokenizer import tokenizer_from_dict


def load(path):
    sd, meta = serialize.load_state(path)
    m = GPT(GPTConfig(**meta["config"]))
    m.load_state_dict(sd)
    return m.eval(), meta


def speculative(n_new=240, seeds=3, temperature=0.8):
    target, meta = load(ROOT / "runs" / "gpt-char" / "model.npz")
    tok = tokenizer_from_dict(meta["tokenizer"])
    prompt = tok.encode("ROMEO:\n")
    drafts = {}
    for tag in ("scale-L1-D48", "scale-L2-D64", "scale-L3-D96", "scale-L4-D128"):
        d, dmeta = load(ROOT / "runs" / "studies" / tag / "model.npz")
        assert dmeta["tokenizer"] == meta["tokenizer"], "draft and target must share a tokenizer"
        drafts[tag] = d

    def timed(fn):
        t0 = time.perf_counter()
        out, stats = fn()
        return time.perf_counter() - t0, out, stats

    # cost of one single-token forward pass for each model (the draft/target cost ratio c)
    def step_cost(model):
        from sutra.models.speculative import CachedStream
        s = CachedStream(model)
        s.prime(prompt)
        t0 = time.perf_counter()
        for i in range(60):
            s.feed([prompt[i % len(prompt)]])
        return (time.perf_counter() - t0) / 60

    target_cost = step_cost(target)
    base = [timed(lambda s=s: generate_target_only(target, prompt, n_new, temperature, None, s))[0] for s in range(seeds)]
    base_tps = n_new / float(np.mean(base))
    print(f"target-only: {base_tps:.0f} tokens/s  (one target forward = {1000 * target_cost:.2f} ms)")
    rows = []
    for tag, draft in drafts.items():
        c = step_cost(draft) / target_cost
        for k in (1, 2, 3, 4, 6, 8):
            ts, alphas, tpc = [], [], []
            for s in range(seeds):
                dt, _, stats = timed(lambda s=s: generate_speculative(target, draft, prompt, n_new, k, temperature, None, s))
                ts.append(dt)
                alphas.append(stats["acceptance_rate"])
                tpc.append(stats["tokens_per_target_call"])
            a = float(np.mean(alphas))
            row = {"draft": tag, "draft_params": draft.num_parameters(), "cost_ratio": c, "k": k, "alpha": a,
                   "tokens_per_target_call": float(np.mean(tpc)),
                   "theory_tokens_per_call": (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1,
                   "tokens_per_s": n_new / float(np.mean(ts)), "speedup": (n_new / float(np.mean(ts))) / base_tps}
            rows.append(row)
            print(f"  {tag:14s} k={k}  alpha {a:.3f}  tokens/call {row['tokens_per_target_call']:.2f} "
                  f"(theory {row['theory_tokens_per_call']:.2f})  speedup {row['speedup']:.2f}x")
    save("speculative", {"target_params": target.num_parameters(), "n_new": n_new, "seeds": seeds,
                         "temperature": temperature, "baseline_tokens_per_s": base_tps,
                         "target_step_ms": 1000 * target_cost, "rows": rows})


def checkpointing(depths=(2, 4, 6, 8, 12), batch=32, block=128, width=192):
    rows = []
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 65, (batch, block))
    tgt = np.roll(idx, -1, axis=1)
    for L in depths:
        for ckpt in (False, True):
            st.manual_seed(0)
            m = GPT(GPTConfig(vocab_size=65, block_size=block, n_layer=L, n_head=6, n_embd=width))
            m.grad_checkpointing = ckpt

            def step():
                _, loss = m(idx, tgt)
                loss.backward()
                m.zero_grad()

            step()  # warm-up
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                step()
                times.append(time.perf_counter() - t0)
            gc.collect()
            tracemalloc.start()
            step()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"n_layer": L, "checkpointing": ckpt, "peak_mb": peak / 2 ** 20,
                         "step_ms": 1000 * float(np.median(times)), "params": m.num_parameters()})
            print(f"  {L:2d} layers  checkpointing={ckpt!s:5s}  peak {peak / 2 ** 20:7.1f} MB  "
                  f"step {rows[-1]['step_ms']:6.0f} ms")
    save("checkpointing", {"batch": batch, "block": block, "width": width, "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["speculative", "checkpointing", "all"], default="all")
    a = ap.parse_args()
    if a.study in ("speculative", "all"):
        speculative()
    if a.study in ("checkpointing", "all"):
        checkpointing()
