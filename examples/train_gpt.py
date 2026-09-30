"""
Train a GPT on Tiny Shakespeare with Sutra.

    python examples/train_gpt.py                                  # char-level flagship model
    python examples/train_gpt.py --tokenizer bpe --vocab 512      # byte-level BPE variant

Loss is reported in nats/token; because char and BPE tokens differ in length, models are
compared in bits per character (BPC), the tokenizer-independent metric.
Writes runs/<name>/{model.npz, log.json, samples.txt}.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sutra as st  # noqa: E402
from sutra import optim, serialize  # noqa: E402
from sutra.data import get_lm_batch, tiny_shakespeare  # noqa: E402
from sutra.models import GPT, GPTConfig  # noqa: E402
from sutra.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab", type=int, default=512, help="BPE vocabulary size")
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def prepare_data(kind: str, vocab: int):
    text = tiny_shakespeare()
    n = int(0.9 * len(text))
    train_text, val_text = text[:n], text[n:]
    if kind == "char":
        tok = CharTokenizer.from_text(text)
    else:
        tok = BPETokenizer.train(train_text, vocab)
    train_ids = np.array(tok.encode(train_text), dtype=np.int64)
    val_ids = np.array(tok.encode(val_text), dtype=np.int64)
    return tok, train_ids, val_ids, len(val_text) / len(val_ids)


@st.no_grad()
def evaluate(model, batches):
    model.eval()
    losses = [model(x, y)[1].item() for x, y in batches]
    model.train()
    return float(np.mean(losses))


def train(args) -> dict:
    out = Path(args.out or ROOT / "runs" / f"gpt-{args.tokenizer}")
    out.mkdir(parents=True, exist_ok=True)
    st.manual_seed(args.seed)
    say = (lambda *a, **k: None) if args.quiet else print

    tok, train_ids, val_ids, cpt = prepare_data(args.tokenizer, args.vocab)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg)
    n_params = model.num_parameters()
    say(f"tokenizer={args.tokenizer} vocab={tok.vocab_size} chars/token={cpt:.3f} "
        f"train tokens={len(train_ids):,} params={n_params:,}")

    # fixed evaluation batches => every eval measures the same data (low-variance comparisons)
    st.manual_seed(1234)
    eval_train = [get_lm_batch(train_ids, args.batch, args.block) for _ in range(args.eval_iters)]
    eval_val = [get_lm_batch(val_ids, args.batch, args.block) for _ in range(args.eval_iters)]
    st.manual_seed(args.seed)

    opt = optim.AdamW(model.param_groups(args.wd), lr=args.lr, betas=(0.9, 0.99))
    sched = optim.CosineWarmup(opt, args.warmup, args.iters, args.min_lr_ratio)
    log = {"config": vars(args), "model": cfg.to_dict(), "params": n_params, "chars_per_token": cpt,
           "steps": [], "evals": []}
    best, t0, window = float("inf"), time.time(), []
    tokens_per_step = args.batch * args.block

    for it in range(1, args.iters + 1):
        x, y = get_lm_batch(train_ids, args.batch, args.block)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        gnorm = optim.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        window.append(loss.item())

        if it % 10 == 0:
            el = time.time() - t0
            log["steps"].append({"step": it, "loss": float(np.mean(window)), "lr": opt.lr, "grad_norm": gnorm,
                                 "tok_per_s": it * tokens_per_step / el})
            window = []
        if it % args.eval_every == 0 or it == args.iters:
            tr, va = evaluate(model, eval_train), evaluate(model, eval_val)
            bpc = va / math.log(2) / cpt
            log["evals"].append({"step": it, "train": tr, "val": va, "val_bpc": bpc, "time": time.time() - t0})
            say(f"step {it:5d} | train {tr:.4f} | val {va:.4f} | val bpc {bpc:.3f} | "
                f"lr {opt.lr:.2e} | {it * tokens_per_step / (time.time() - t0):,.0f} tok/s", flush=True)
            if va < best:
                best = va
                serialize.save(model, out / "model.npz", config=cfg.to_dict(), tokenizer=tok.to_dict())

    serialize.load(model, out / "model.npz")
    best_eval = min(log["evals"], key=lambda e: e["val"])
    log.update(best_val=best_eval["val"], best_val_bpc=best_eval["val_bpc"], best_step=best_eval["step"],
               train_time_s=time.time() - t0)
    samples = []
    for s in range(args.samples):
        prompt = np.array(tok.encode("\n"), dtype=np.int64)
        ids = model.generate(prompt, 400 if args.tokenizer == "char" else 150, temperature=0.8, top_k=40, seed=s)
        samples.append(tok.decode(ids[0].tolist()))
    log["samples"] = samples
    (out / "samples.txt").write_text("\n\n=====\n\n".join(samples))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    say(f"best val {best_eval['val']:.4f} nats/token = {best_eval['val_bpc']:.3f} bits/char at step "
        f"{best_eval['step']} ({log['train_time_s'] / 60:.1f} min)")
    say("\n--- sample ---\n" + samples[0])
    return log


if __name__ == "__main__":
    train(build_parser().parse_args())
