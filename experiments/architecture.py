"""
GPT-2 block vs LLaMA-style block, one component at a time, on Tiny Shakespeare (char level).

Five configurations × 3 seeds, all 4 layers × 128 wide, 1,500 steps × 4,096 tokens, identical
hyper-parameters and identical evaluation batches. Each variant is compared with the GPT-2 baseline
by Welch's t-test on best validation loss.

    python experiments/architecture.py            # ~25 min with 4 worker processes
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import ROOT, save, summarize, welch

sys.path.insert(0, str(ROOT / "examples"))

CONFIGS = {
    "GPT-2 baseline": {},
    "+ RMSNorm": {"norm": "rms"},
    "+ RoPE": {"pos": "rope"},
    "+ SwiGLU": {"mlp": "swiglu"},
    "LLaMA-style (all three, no biases)": {"norm": "rms", "pos": "rope", "mlp": "swiglu", "no_bias": True},
}


def run_one(job):
    name, kw, seed, iters = job
    from train_gpt import build_parser, train
    tag = "arch-" + "-".join(f"{k}{v}" for k, v in kw.items() if k != "no_bias") + ("-nobias" if kw.get("no_bias") else "")
    tag = (tag if tag != "arch-" else "arch-baseline") + f"-s{seed}"
    out = ROOT / "runs" / "studies" / tag
    done = out / "log.json"
    if done.exists():
        return name, seed, json.loads(done.read_text())
    argv = ["--iters", str(iters), "--n_layer", "4", "--n_embd", "128", "--n_head", "4", "--dropout", "0",
            "--eval_every", str(iters // 3), "--eval_iters", "20", "--samples", "1", "--quiet", "--seed", str(seed),
            "--out", str(out)]
    for k, v in kw.items():
        argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return name, seed, train(build_parser().parse_args(argv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(n, kw, s, args.iters) for n, kw in CONFIGS.items() for s in range(args.seeds)]
    logs = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for name, seed, log in ex.map(run_one, jobs):
            logs.setdefault(name, []).append(log)
            print(f"  {name:36s} seed {seed}  val {log['best_val']:.4f}  params {log['params']:,}", flush=True)
    base = [l["best_val"] for l in logs["GPT-2 baseline"]]
    rows = []
    for name in CONFIGS:
        vals = [l["best_val"] for l in logs[name]]
        row = {"config": name, "params": logs[name][0]["params"], "val_loss": summarize(vals),
               "minutes": float(np.mean([l["train_time_s"] for l in logs[name]]) / 60)}
        if name != "GPT-2 baseline":
            t, p = welch(vals, base)
            row["vs_baseline"] = {"delta": float(np.mean(vals) - np.mean(base)), "t": t, "p": p}
        rows.append(row)
        print(f"{name:36s} {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
              + (f"   Δ {row['vs_baseline']['delta']:+.4f}  p={row['vs_baseline']['p']:.3g}" if "vs_baseline" in row else ""))
    save("architecture", {"iters": args.iters, "seeds": args.seeds, "architecture": "4 layers x 128, 4 heads, context 128",
                          "rows": rows})


if __name__ == "__main__":
    main()
