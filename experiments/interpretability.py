"""
Mechanistic interpretability of GPTs trained with Sutra.

  flagship   Dissects the 6-layer Shakespeare GPT: four in-context copying probes, previous-token /
             duplicate-token / induction scores for all 36 heads, mean-ablation of every head on text loss,
             and the logit lens. Result: strong previous-token heads, no induction heads, no copying.
  oracle     Data analysis explaining why: how often a verbatim repeat is even available to copy from,
             as a function of context length.
  synthetic  A controlled task where copying is the only way to reduce loss (a random segment reappears after
             a random-length prefix). 2-layer attention-only models acquire induction heads in a sudden phase
             change; 1-layer models, which cannot compose two attention layers, never do (Olsson et al., 2022).
             A 2-layer arm with MLPs is included for comparison. The trained circuit is then dissected by
             ablation, including a K-composition test. 3 seeds each.
  context    The hypothesis test on real text: the same 4-layer GPT trained on Shakespeare with 128- vs
             512-character context, probed for induction heads as it trains.

    python experiments/interpretability.py --study flagship|oracle|synthetic|context|all
"""
import argparse
import re
import string
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import ROOT, save, st

from sutra import interp, optim, serialize
from sutra.data import get_lm_batch, tiny_shakespeare
from sutra.models import GPT, GPTConfig
from sutra.tokenizer import tokenizer_from_dict

sys.path.insert(0, str(ROOT / "examples"))
from train_gpt import prepare_data  # noqa: E402


def val_batches(val_ids, n, batch=32, block=128, seed=1234):
    st.manual_seed(seed)
    return [get_lm_batch(val_ids, batch, block) for _ in range(n)]


def text_loss(batches):
    def fn(model):
        with st.no_grad():
            model.eval()
            return float(np.mean([model(x, y)[1].item() for x, y in batches]))
    return fn


def text_probe(tok, val_text, n, length, seed=0):
    """Real held-out Shakespeare chunks, each repeated twice after a newline."""
    rng = np.random.default_rng(seed)
    nl = tok.encode("\n")
    rows = []
    for _ in range(n):
        i = int(rng.integers(0, len(val_text) - length))
        c = tok.encode(val_text[i:i + length])
        rows.append(nl + c + c)
    return np.array(rows, dtype=np.int64)


# ------------------------------------------------------------------ flagship

def flagship():
    sd, meta = serialize.load_state(ROOT / "runs" / "gpt-char" / "model.npz")
    model = GPT(GPTConfig(**meta["config"]))
    model.load_state_dict(sd)
    model.eval()
    tok = tokenizer_from_dict(meta["tokenizer"])
    text = tiny_shakespeare()
    val_text = text[int(0.9 * len(text)):]
    _, _, val_ids, _ = prepare_data("char", 0)
    batches = val_batches(val_ids, 8)
    Lyr, H, L = model.cfg.n_layer, model.cfg.n_head, 60
    rng = np.random.default_rng(0)
    nl = tok.encode("\n")

    words = sorted(set(re.findall(r"[a-z]+", val_text.lower())))

    def word_probe(n):
        rows = []
        for _ in range(n):
            s = ""
            while len(s) < L:
                s += str(rng.choice(words)) + " "
            w = tok.encode(s[:L])
            rows.append(nl + w + w)
        return np.array(rows, dtype=np.int64)

    probes = {
        "random letters": interp.repeated_sequences([tok.stoi[c] for c in string.ascii_letters], 32, L, nl, seed=0),
        "random lowercase": interp.repeated_sequences([tok.stoi[c] for c in string.ascii_lowercase], 32, L, nl, seed=0),
        "random real words": word_probe(32),
        "real text, repeated": text_probe(tok, val_text, 32, L),
    }
    probe_rows, all_scores = [], None
    for name, seqs in probes.items():
        first, second = interp.copy_losses(model, seqs, 1, L)
        sc = interp.head_scores(interp.attention_patterns(model, seqs), 1, L)
        probe_rows.append({"probe": name, "first_copy_loss": first, "second_copy_loss": second,
                           "max_induction": float(sc["induction"].max())})
        print(f"  {name:22s} first copy {first:.3f}  second copy {second:.3f}  max induction {sc['induction'].max():.3f}")
        if name == "real text, repeated":
            all_scores = sc
            example_seq = seqs[0]

    means = interp.head_means(model, batches)
    abl = interp.ablation_matrix(model, {"text": text_loss(batches)}, means)
    lens = interp.logit_lens(model, batches)
    print(f"  baseline text loss {abl['baseline']['text']:.4f}; worst single-head ablation "
          f"+{abl['delta']['text'].max():.4f}; logit lens {[round(v, 3) for v in lens]}")

    prev = all_scores["previous_token"]
    lp, hp = np.unravel_index(prev.argmax(), prev.shape)
    pats = interp.attention_patterns(model, example_seq[None])
    save("interp_flagship", {
        "probe_length": L, "probes": probe_rows,
        "scores": {k: v.tolist() for k, v in all_scores.items()},
        "ablation": {"baseline": abl["baseline"], "delta": {k: v.tolist() for k, v in abl["delta"].items()}},
        "logit_lens": lens,
        "example": {"tokens": tok.decode(example_seq[:48].tolist()), "previous_head": [int(lp), int(hp)],
                    "pattern": np.round(pats[lp, 0, hp, :48, :48], 3).tolist()}})


# ------------------------------------------------------------------ data analysis

def oracle():
    """How often could copying even help? At position t, look for the most recent earlier occurrence (inside
    the context window) of the k characters before t, and predict the character that followed it."""
    text = tiny_shakespeare()
    val = text[int(0.9 * len(text)):]
    rng = np.random.default_rng(0)
    pos = rng.choice(np.arange(2048, len(val) - 1), 4000, replace=False)
    rows = []
    for k in (4, 8, 16):
        for W in (32, 64, 128, 256, 512, 1024, 2048):
            fire = hit = 0
            for t in pos:
                ctx = val[t - W:t]
                j = ctx.rfind(ctx[-k:], 0, len(ctx) - 1)
                if j != -1 and j + k < len(ctx):
                    fire += 1
                    hit += ctx[j + k] == val[t]
            rows.append({"k": k, "window": W, "coverage": fire / len(pos), "precision": hit / fire if fire else 0.0,
                         "useful": hit / len(pos)})
            print(f"  k={k:2d} window {W:5d}: available at {100 * fire / len(pos):5.1f}% of characters, "
                  f"right {100 * (hit / fire if fire else 0):5.1f}% of the time")
    save("interp_oracle", {"positions": len(pos), "rows": rows})


# ------------------------------------------------------------------ synthetic copying task

V, T_SYN = 64, 64
ARMS = {"1-layer, attention-only": (1, "none"), "2-layer, attention-only": (2, "none"), "2-layer, with MLPs": (2, "gelu")}


def copy_batch(rng, B):
    """[random prefix] + r + r with |r| drawn from 16-31, so the repeat offset differs in every sequence and
    no fixed positional shortcut exists. Returns inputs, targets and the mask of copyable target positions."""
    x = rng.integers(0, V, (B, T_SYN + 1))
    mask = np.zeros((B, T_SYN), dtype=bool)
    for b in range(B):
        n = int(rng.integers(16, 32))
        m = T_SYN + 1 - 2 * n
        x[b, m + n:m + 2 * n] = x[b, m:m + n]
        mask[b, m + n:m + 2 * n - 1] = True  # predicting r_2..r_n of the second copy
    return x[:, :-1], x[:, 1:], mask


def masked_loss(model, xb, yb, mask):
    with st.no_grad():
        model.eval()
        logits, _ = model(xb)
        model.train()
    z = logits.data.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    nll = -(np.take_along_axis(z, yb[..., None], -1)[..., 0] - np.log(np.exp(z).sum(-1)))
    return float(nll[mask].mean()), float(nll[~mask].mean())


def synthetic_model(n_layer, mlp):
    return GPT(GPTConfig(vocab_size=V, block_size=T_SYN, n_layer=n_layer, n_head=2, n_embd=64, mlp=mlp))


def synthetic_run(job):
    arm, seed, steps = job
    n_layer, mlp = ARMS[arm]
    st.manual_seed(seed)
    model = synthetic_model(n_layer, mlp)
    opt = optim.AdamW(model.param_groups(0.01), lr=1e-3)
    rng = np.random.default_rng(100 + seed)
    ev = copy_batch(np.random.default_rng(7), 256)
    probe_prefix = list(np.random.default_rng(3).integers(0, V, 8))
    probe = interp.repeated_sequences(range(V), 64, 24, probe_prefix, seed=4)
    hist = []
    for step in range(steps + 1):
        if step % 50 == 0:
            copy, rand = masked_loss(model, *ev)
            sc = interp.head_scores(interp.attention_patterns(model, probe), 8, 24)
            hist.append({"step": step, "copy_loss": copy, "random_loss": rand,
                         "induction_max": float(sc["induction"].max()), "prev_max": float(sc["previous_token"].max())})
        if step == steps:
            break
        xb, yb, _ = copy_batch(rng, 64)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  {arm} seed {seed}: copy loss {hist[-1]['copy_loss']:.3f} (chance {np.log(V):.3f}), "
          f"induction max {hist[-1]['induction_max']:.2f}", flush=True)
    circuit = None
    if arm == "2-layer, attention-only":
        circuit = dissect(model, probe, ev)
    return arm, seed, hist, circuit


def dissect(model, probe, ev):
    """Per-head scores, single-head ablations, and the K-composition test: removing the layer-1
    previous-token head should break the layer-2 induction head even though that head is left intact."""
    sc = interp.head_scores(interp.attention_patterns(model, probe), 8, 24)
    xb, yb, _ = copy_batch(np.random.default_rng(11), 128)
    means = interp.head_means(model, [(xb, yb)])
    loss_fn = lambda mm: masked_loss(mm, *ev)[0]  # noqa: E731
    abl = interp.ablation_matrix(model, {"copy": loss_fn}, means)
    li, hi = np.unravel_index(sc["induction"].argmax(), sc["induction"].shape)
    prev0 = sc["previous_token"][0]
    hp = int(prev0.argmax())
    with interp.ablated(model, [(0, hp)], means):
        after = interp.head_scores(interp.attention_patterns(model, probe), 8, 24)["induction"][li, hi]
        after_loss = loss_fn(model)
    pats = interp.attention_patterns(model, probe[:1])
    return {"scores": {k: v.tolist() for k, v in sc.items()}, "ablation_baseline": abl["baseline"]["copy"],
            "ablation_delta": abl["delta"]["copy"].tolist(),
            "induction_head": [int(li), int(hi)], "previous_head": [0, hp],
            "composition": {"induction_score_before": float(sc["induction"][li, hi]),
                            "induction_score_after_prev_ablation": float(after),
                            "copy_loss_after_prev_ablation": after_loss},
            "tokens": probe[0].tolist(),
            "induction_pattern": np.round(pats[li, 0, hi], 3).tolist(),
            "previous_pattern": np.round(pats[0, 0, hp], 3).tolist()}


def synthetic(steps, seeds, workers):
    jobs = [(arm, s, steps) for arm in ARMS for s in range(seeds)]
    runs, circuits = {arm: [] for arm in ARMS}, []
    with ProcessPoolExecutor(workers) as ex:
        for arm, s, hist, circuit in ex.map(synthetic_run, jobs):
            runs[arm].append({"seed": s, "history": hist})
            if circuit:
                circuits.append({"seed": s, **circuit})
    save("interp_synthetic", {"steps": steps, "seeds": seeds, "vocab": V, "context": T_SYN,
                              "model": "64 wide, 2 heads per layer", "chance_loss": float(np.log(V)),
                              "runs": runs, "circuits": circuits})


# ------------------------------------------------------------------ context length on real text

def context_run(job):
    block, steps = job
    tok, train_ids, val_ids, _ = prepare_data("char", 0)
    text = tiny_shakespeare()
    val_text = text[int(0.9 * len(text)):]
    L = (block - 1) // 2 - 4
    probe = text_probe(tok, val_text, 16, L, seed=5)
    batch = 4096 // block
    st.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=block, n_layer=4, n_head=4, n_embd=128))
    opt = optim.AdamW(model.param_groups(0.1), lr=1e-3, betas=(0.9, 0.99))
    sched = optim.CosineWarmup(opt, 100, steps, 0.1)
    batches = val_batches(val_ids, 4, batch=batch, block=block)
    hist = []
    for step in range(steps + 1):
        if step % 100 == 0:
            first, second = interp.copy_losses(model, probe, 1, L)
            sc = interp.head_scores(interp.attention_patterns(model, probe), 1, L)
            hist.append({"step": step, "val": text_loss(batches)(model), "first_copy": first, "second_copy": second,
                         "induction_max": float(sc["induction"].max()), "prev_max": float(sc["previous_token"].max())})
            model.train()
            print(f"  context {block}: step {step} val {hist[-1]['val']:.3f} copy {first:.3f}->{second:.3f} "
                  f"induction {hist[-1]['induction_max']:.3f}", flush=True)
        if step == steps:
            break
        x, y = get_lm_batch(train_ids, batch, block)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return block, hist


def context(steps):
    runs = {}
    with ProcessPoolExecutor(2) as ex:
        for block, hist in ex.map(context_run, [(128, steps), (512, steps)]):
            runs[str(block)] = hist
    save("interp_context", {"steps": steps, "tokens_per_step": 4096, "model": "4 layers x 128, 4 heads", "runs": runs})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["flagship", "oracle", "synthetic", "context", "all"], default="all")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.study in ("flagship", "all"):
        flagship()
    if a.study in ("oracle", "all"):
        oracle()
    if a.study in ("synthetic", "all"):
        synthetic(a.steps or 5000, a.seeds, a.workers)
    if a.study in ("context", "all"):
        context(a.steps or 3000)
