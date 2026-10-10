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
    # leave room above the panels when the figure has an overall title
    fig.tight_layout(rect=(0, 0, 1, 0.93) if fig._suptitle is not None else (0, 0, 1, 1))
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


BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", [SURFACE] + BLUES)


def heat(ax, m, title, vmin=0, vmax=1, fmt="{:.2f}"):
    m = np.asarray(m)
    im = ax.imshow(m, cmap=blue_cmap(), vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(m.shape[1]), [f"H{i + 1}" for i in range(m.shape[1])])
    ax.set_yticks(range(m.shape[0]), [f"L{i + 1}" for i in range(m.shape[0])])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if (v - vmin) / (vmax - vmin + 1e-12) > 0.55 else INK)
    ax.set_title(title)
    return im


def architecture():
    d = load("architecture")
    if not d:
        return
    rows = d["rows"][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    for i, r in enumerate(rows):
        m, ci = r["val_loss"]["mean"], r["val_loss"]["ci95"]
        base = "vs_baseline" not in r
        ax.errorbar(m, i, xerr=ci, fmt="o", color=MUTED if base else S[0], ecolor=INK2, elinewidth=1.4, capsize=4,
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        note = "baseline" if base else f"Δ {r['vs_baseline']['delta']:+.3f}, p = {r['vs_baseline']['p']:.1g}"
        ax.annotate(f"{m:.3f}  ({note})", (m + ci, i), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r["config"] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    lo = min(r["val_loss"]["mean"] - r["val_loss"]["ci95"] for r in rows)
    hi = max(r["val_loss"]["mean"] + r["val_loss"]["ci95"] for r in rows)
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.75 * (hi - lo))
    ax.set(xlabel="validation loss, nats/char (mean and 95% CI, lower is better)")
    fig.suptitle(f"From GPT-2 to LLaMA, one component at a time · {d['seeds']} seeds × {d['iters']:,} steps",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    save(fig, "architecture")


def interp_flagship():
    d = load("interp_flagship")
    if not d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    heat(axes[0], d["scores"]["previous_token"], "previous-token score")
    heat(axes[1], d["scores"]["induction"], "induction score")
    delta = np.asarray(d["ablation"]["delta"]["text"])
    heat(axes[2], delta, "Δ loss when the head is ablated", vmin=0, vmax=max(delta.max(), 1e-3), fmt="{:.3f}")
    fig.suptitle("Flagship GPT: what each of its 36 attention heads does", x=0.01, ha="left", fontweight="bold")
    save(fig, "interp_heads")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    lens = d["logit_lens"]
    xs = np.arange(len(lens))
    ax.plot(xs[1:], lens[1:], color=S[0], marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, y in zip(xs[1:], lens[1:]):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(xs[1:], [f"after block {i}" for i in xs[1:]])
    ax.set_ylim(min(lens[1:]) - 0.15, max(lens[1:]) + 0.3)
    ax.tick_params(axis="y", length=0)
    ax.set(ylabel="loss (nats/char)", title="Logit lens: decoding the residual stream after every block")
    save(fig, "logit_lens")


def oracle():
    d = load("interp_oracle")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, k in enumerate((4, 8, 16)):
        rows = [r for r in d["rows"] if r["k"] == k]
        xs, ys = [r["window"] for r in rows], [100 * r["coverage"] for r in rows]
        ax.plot(xs, ys, color=S[i], marker="o", markersize=5, markeredgecolor=SURFACE, label=f"repeat of ≥ {k} characters")
        end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.1f}%")
    ax.axvline(128, color=AXIS, linewidth=1)
    ax.annotate("flagship context", (128, ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048], ["32", "64", "128", "256", "512", "1k", "2k"])
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="context window (characters)", ylabel="% of characters",
           title="How often a verbatim repeat is available to copy from (Shakespeare)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "copy_oracle")


def synthetic():
    d = load("interp_synthetic")
    if not d:
        return
    arms = list(d["runs"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for i, arm in enumerate(arms):
        for j, run in enumerate(d["runs"][arm]):
            h = run["history"]
            steps = [e["step"] for e in h]
            axes[0].plot(steps, [e["copy_loss"] for e in h], color=S[i], linewidth=1.6, alpha=0.9,
                         label=arm if j == 0 else None)
            axes[1].plot(steps, [e["induction_max"] for e in h], color=S[i], linewidth=1.6, alpha=0.9,
                         label=arm if j == 0 else None)
    axes[0].axhline(d["chance_loss"], color=AXIS, linewidth=1)
    axes[0].annotate("chance", (0, d["chance_loss"]), xytext=(2, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    axes[0].set(xlabel="training step", ylabel="loss on copyable tokens (nats)", title="In-context copying")
    axes[1].set(xlabel="training step", ylabel="best induction score", title="Induction heads")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle(f"The induction-head phase change · {d['seeds']} seeds per arm", x=0.01, ha="left", fontweight="bold")
    save(fig, "induction_phase_change")


def context():
    d = load("interp_context")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for i, (block, hist) in enumerate(sorted(d["runs"].items(), key=lambda kv: int(kv[0]))):
        steps = [e["step"] for e in hist]
        gap = [e["first_copy"] - e["second_copy"] for e in hist]
        axes[0].plot(steps, gap, color=S[i], label=f"context {block}")
        axes[1].plot(steps, [e["induction_max"] for e in hist], color=S[i], label=f"context {block}")
    axes[0].axhline(0, color=AXIS, linewidth=1)
    axes[0].set(xlabel="training step", ylabel="first-copy − second-copy loss", title="Benefit from seeing the text before")
    axes[1].set(xlabel="training step", ylabel="best induction score", title="Induction heads")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.legend(fontsize=8.5)
    save(fig, "context_induction")


def speculative_fig():
    d = load("speculative")
    if not d:
        return
    drafts = list(dict.fromkeys(r["draft"] for r in d["rows"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for i, dr in enumerate(drafts):
        rows = [r for r in d["rows"] if r["draft"] == dr]
        label = f"{rows[0]['draft_params'] / 1e3:.0f}k draft (α = {np.mean([r['alpha'] for r in rows]):.2f})"
        axes[0].plot([r["k"] for r in rows], [r["speedup"] for r in rows], color=S[i], marker="o", markersize=5,
                     markeredgecolor=SURFACE, label=label)
        axes[1].plot([r["theory_tokens_per_call"] for r in rows], [r["tokens_per_target_call"] for r in rows], "o",
                     color=S[i], markersize=6, markeredgecolor=SURFACE)
    axes[0].axhline(1, color=AXIS, linewidth=1)
    axes[0].set(xlabel="draft length k", ylabel="speedup over plain decoding", title="Wall-clock speedup")
    lim = max(max(r["theory_tokens_per_call"], r["tokens_per_target_call"]) for r in d["rows"]) * 1.08
    axes[1].plot([1, lim], [1, lim], color=MUTED, linewidth=1.2)
    axes[1].set(xlim=(1, lim), ylim=(1, lim), xlabel="predicted (1 − α^(k+1)) / (1 − α)",
                ylabel="measured", title="Tokens per target forward pass")
    axes[0].legend(fontsize=8.5)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Speculative decoding · {d['target_params'] / 1e6:.1f}M target", x=0.01, ha="left", fontweight="bold")
    save(fig, "speculative")


def checkpointing_fig():
    d = load("checkpointing")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for i, flag in enumerate((False, True)):
        rows = [r for r in d["rows"] if r["checkpointing"] == flag]
        name = "with checkpointing" if flag else "standard backprop"
        axes[0].plot([r["n_layer"] for r in rows], [r["peak_mb"] for r in rows], color=S[i], marker="o",
                     markersize=5, markeredgecolor=SURFACE, label=name)
        axes[1].plot([r["n_layer"] for r in rows], [r["step_ms"] for r in rows], color=S[i], marker="o",
                     markersize=5, markeredgecolor=SURFACE, label=name)
    axes[0].set(xlabel="layers", ylabel="peak memory (MB)", title="Memory")
    axes[1].set(xlabel="layers", ylabel="forward + backward (ms)", title="Time")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.legend(fontsize=8.5)
    fig.suptitle(f"Gradient checkpointing · width {d['width']}, batch {d['batch']} × {d['block']}", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "checkpointing")


if __name__ == "__main__":
    for fn in (gpt_training, mnist, optimizers, init_signal, batchnorm, scaling, tokenizers, architecture,
               interp_flagship, oracle, synthetic, context, speculative_fig, checkpointing_fig):
        fn()
