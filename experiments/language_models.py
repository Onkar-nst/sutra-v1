"""
Language-model studies on Tiny Shakespeare, every model trained with Sutra.

  scaling    — six GPTs from 0.03M to 2.7M non-embedding parameters, identical data, steps and
               hyper-parameters. Fits a saturating power law  L(N) = E + A / N^alpha.
  tokenizers — one architecture trained with char-level, byte-level (BPE-256, no merges), BPE-512 and
               BPE-1024 tokenizers at equal compute (same steps x tokens). Compared in bits per character.

    python experiments/language_models.py --study scaling
    python experiments/language_models.py --study tokenizers
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from common import ROOT, save

sys.path.insert(0, str(ROOT / "examples"))
from train_gpt import build_parser, train  # noqa: E402

SIZES = [(1, 48, 2), (2, 64, 2), (3, 96, 3), (4, 128, 4), (5, 160, 5), (6, 192, 6)]


def run(tag, iters, **kw):
    """Train one model, or reuse its finished run (so an interrupted study resumes where it stopped)."""
    done = ROOT / "runs" / "studies" / tag / "log.json"
    if done.exists():
        log = json.loads(done.read_text())
        if log["config"]["iters"] == iters and all(str(log["config"].get(k)) == str(v) for k, v in kw.items()):
            return log
    argv = ["--iters", str(iters), "--dropout", "0", "--eval_every", str(iters // 3), "--eval_iters", "20",
            "--samples", "1", "--quiet", "--out", str(ROOT / "runs" / "studies" / tag)]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return train(build_parser().parse_args(argv))


def non_embedding_params(log):
    m = log["model"]
    return log["params"] - m["vocab_size"] * m["n_embd"] - m["block_size"] * m["n_embd"]


def study_scaling(iters):
    rows = []
    for L, D, H in SIZES:
        log = run(f"scale-L{L}-D{D}", iters, n_layer=L, n_embd=D, n_head=H)
        n = non_embedding_params(log)
        tokens = iters * 32 * 128
        rows.append({"n_layer": L, "n_embd": D, "params": log["params"], "non_embedding_params": n,
                     "val_loss": log["best_val"], "val_bpc": log["best_val_bpc"],
                     "train_flops": 6 * log["params"] * tokens, "minutes": log["train_time_s"] / 60,
                     "curve": [(e["step"], e["val"]) for e in log["evals"]]})
        print(f"  L{L} D{D}: {n:>9,} non-emb params  val {log['best_val']:.4f}  ({rows[-1]['minutes']:.1f} min)",
              flush=True)

    from scipy.optimize import curve_fit
    N = np.array([r["non_embedding_params"] for r in rows], dtype=float)
    Lv = np.array([r["val_loss"] for r in rows])
    law = lambda n, E, A, alpha: E + A * n ** (-alpha)  # noqa: E731
    # E is an irreducible loss, so it must be >= 0; unconstrained fits can wander to negative E
    (E, A, alpha), _ = curve_fit(law, N, Lv, p0=(1.3, 10.0, 0.3), bounds=([0, 0, 0], [np.inf, np.inf, 2]),
                                 maxfev=50000)
    pred = law(N, E, A, alpha)
    r2 = 1 - ((Lv - pred) ** 2).sum() / ((Lv - Lv.mean()) ** 2).sum()
    print(f"  fit: L(N) = {E:.3f} + {A:.2f} * N^-{alpha:.3f}   (R^2 = {r2:.4f})")
    residuals = (Lv - pred).tolist()
    return {"residuals": residuals, "iters": iters, "tokens": iters * 32 * 128, "rows": rows,
            "fit": {"E": float(E), "A": float(A), "alpha": float(alpha), "r2": float(r2)}}


def study_tokenizers(iters):
    rows = []
    for name, kw in [("char", {"tokenizer": "char"}), ("BPE-256 (bytes)", {"tokenizer": "bpe", "vocab": 256}),
                     ("BPE-512", {"tokenizer": "bpe", "vocab": 512}), ("BPE-1024", {"tokenizer": "bpe", "vocab": 1024})]:
        tag = "tok-" + name.split(" ")[0].lower()
        log = run(tag, iters, n_layer=4, n_embd=128, n_head=4, **kw)
        rows.append({"tokenizer": name, "vocab_size": log["model"]["vocab_size"], "params": log["params"],
                     "chars_per_token": log["chars_per_token"], "val_loss_per_token": log["best_val"],
                     "val_bpc": log["best_val_bpc"], "sample": log["samples"][0][:300]})
        print(f"  {name:16s} vocab {log['model']['vocab_size']:5d}  chars/token {log['chars_per_token']:.2f}  "
              f"val bpc {log['best_val_bpc']:.3f}", flush=True)
    return {"iters": iters, "architecture": "4 layers, 128 wide, 4 heads, context 128 tokens", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["scaling", "tokenizers", "all"], default="all")
    ap.add_argument("--iters", type=int, default=1500)
    args = ap.parse_args()
    if args.study in ("scaling", "all"):
        print("== scaling study")
        save("scaling", study_scaling(args.iters))
    if args.study in ("tokenizers", "all"):
        print("== tokenizer study")
        save("tokenizers", study_tokenizers(args.iters))


if __name__ == "__main__":
    Path(ROOT / "runs" / "studies").mkdir(parents=True, exist_ok=True)
    main()
