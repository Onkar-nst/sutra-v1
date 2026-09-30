"""
Render the README figures (docs/figures/*.png) from the saved experiment results and training logs.

    python experiments/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round", "axes.axisbelow": True,
})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)


def gpt_training():
    p = ROOT / "runs" / "gpt-char" / "log.json"
    if not p.exists():
        return
    log = json.loads(p.read_text())
    ev = log["evals"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for key, color, name in [("train", S[0], "train"), ("val", S[1], "validation")]:
        xs, ys = [e["step"] for e in ev], [e[key] for e in ev]
        ax.plot(xs, ys, color=color, label=name)
        end_label(ax, xs[-1], ys[-1], f"{name} {ys[-1]:.3f}")
    ax.set(xlabel="training step", ylabel="loss (nats / char)",
           title=f"GPT on Tiny Shakespeare · {log['params'] / 1e6:.1f}M params · best val {log['best_val_bpc']:.2f} bits/char")
    ax.legend(loc="upper right")
    ax.set_xlim(0, ev[-1]["step"] * 1.18)
    save(fig, "gpt_training")


def mnist():
    p = ROOT / "runs" / "mnist" / "log.json"
    if not p.exists():
        return
    log = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs, ys = [s["step"] for s in log["steps"]], [s["loss"] for s in log["steps"]]
    ax.plot(xs, ys, color=S[0])
    ax.set_yscale("log")
    ax.set(xlabel="training step", ylabel="training loss (log)",
           title=f"DigitCNN on MNIST · {log['params']:,} params · test accuracy {100 * log['test_acc']:.2f}%")
    save(fig, "mnist_training")


def optimizers():
    d = load("ablation_optimizers")
    if not d:
        return
    rows = sorted(d["results"].items(), key=lambda kv: kv[1]["val_acc"]["mean"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (name, r) in enumerate(rows):
        m, ci = 100 * r["val_acc"]["mean"], 100 * r["val_acc"]["ci95"]
        ax.errorbar(m, i, xerr=ci, fmt="o", color=S[0], ecolor=INK2, elinewidth=1.4, capsize=4, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{m:.2f}%  (lr {r['best_lr']:g})", (m + ci, i), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    lo = min(100 * (r["val_acc"]["mean"] - r["val_acc"]["ci95"]) for _, r in rows)
    hi = max(100 * (r["val_acc"]["mean"] + r["val_acc"]["ci95"]) for _, r in rows)
    ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.set(xlabel="validation accuracy, % (mean and 95% CI)",
           title=f"Optimizers, each LR-tuned · {d['seeds']} seeds × {d['epochs']} epochs")
    save(fig, "optimizers")


def init_signal():
    d = load("ablation_init")
    if not d:
        return
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, act in zip(axes, ["tanh", "relu"]):
        for i, name in enumerate(["N(0, 0.01)", "N(0, 1)", "Xavier", "Kaiming"]):
            r = d["results"][f"{act} / {name}"]
            ys = np.array([v if v is not None else np.inf for v in r["act_std"]], dtype=float)
            xs = np.arange(1, len(ys) + 1)
            ok = np.isfinite(ys) & (ys < 1e30)
            ax.plot(xs[ok], np.maximum(ys[ok], 1e-30), color=S[i], label=f"{name} → {100 * r['val_acc']['mean']:.1f}% val acc")
            if not ok.all():  # the signal overflowed: stop the line and say so
                j = int(np.argmax(~ok)) - 1
                ax.annotate("overflows to ∞", (xs[j], ys[j]), xytext=(-4, 8), textcoords="offset points",
                            ha="right", fontsize=8.5, color=INK2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel="layer", title=f"{d['depth']}-layer {act} MLP")
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.5)
    axes[0].set_ylabel("activation std at init (log)")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    save(fig, "init_signal")


def batchnorm():
    d = load("ablation_batchnorm")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for (name, color) in [("no BatchNorm", S[1]), ("with BatchNorm", S[0])]:
        rows = d["results"][name]
        xs, ys = [r["lr"] for r in rows], [100 * r["val_acc"]["mean"] for r in rows]
        ax.plot(xs, ys, color=color, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_xscale("log")
    ax.set(xlabel="learning rate (log)", ylabel="validation accuracy, %",
           title=f"{d['depth']}-layer ReLU MLP, SGD + momentum · {d['seeds']} seeds")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5))
    save(fig, "batchnorm_lr")


def scaling():
    d = load("scaling")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    N = np.array([r["non_embedding_params"] for r in d["rows"]], dtype=float)
    L = np.array([r["val_loss"] for r in d["rows"]])
    f = d["fit"]
    grid = np.geomspace(N.min() / 1.3, N.max() * 1.3, 100)
    ax.plot(grid, f["E"] + f["A"] * grid ** (-f["alpha"]), color=MUTED, linewidth=1.5,
            label=(f"fit: {f['E']:.2f} + " if f["E"] >= 0.01 else "fit: ") + f"{f['A']:.2f}·$N^{{-{f['alpha']:.3f}}}$  (R² {f['r2']:.3f})")
    ax.plot(N, L, "o", color=S[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label="trained GPT")
    for r in d["rows"]:
        ax.annotate(f"{r['n_layer']}L×{r['n_embd']}", (r["non_embedding_params"], r["val_loss"]), xytext=(6, 5),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="non-embedding parameters (log)", ylabel="val loss (nats / char)",
           title=f"Scaling at fixed data · {d['iters']:,} steps each")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "scaling")


def tokenizers():
    d = load("tokenizers")
    if not d:
        return
    rows = sorted(d["rows"], key=lambda r: -r["val_bpc"])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    best = min(r["val_bpc"] for r in rows)
    for i, r in enumerate(rows):
        ax.barh(i, r["val_bpc"], height=0.5, color=S[0] if r["val_bpc"] == best else "#c9cbdb")
        ax.annotate(f"{r['val_bpc']:.3f}  ({r['chars_per_token']:.2f} chars/token)", (r["val_bpc"], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [r["tokenizer"] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["val_bpc"] for r in rows) * 1.4)
    ax.set(xlabel="validation bits per character (lower is better)", title="Tokenizers at equal compute")
    save(fig, "tokenizers")


if __name__ == "__main__":
    for fn in (gpt_training, mnist, optimizers, init_signal, batchnorm, scaling, tokenizers):
        fn()
