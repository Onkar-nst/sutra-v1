"""
Controlled ablation studies on MNIST, run entirely on Sutra.

  A. Optimizers  — SGD, SGD+momentum, RMSprop, Adam, AdamW. Each gets its learning rate tuned on a
                   5-point grid (tuning seed), then is re-trained on 5 fresh seeds. Welch t-tests vs the best.
  B. Initialisation & signal propagation — a 20-layer MLP (tanh and ReLU) under four inits. We record
                   per-layer activation std and weight-gradient norm at initialisation, then train.
  C. BatchNorm × learning rate — a 10-layer ReLU MLP with and without BatchNorm across a 6-point LR sweep.

All accuracies are on the 5k validation split; the test set is never used here.

    python experiments/ablations.py            # all studies (~10 min on a laptop CPU)
    python experiments/ablations.py --study B
"""
import argparse
import time

import numpy as np
from common import fit, mnist_flat, save, st, summarize, welch

from sutra import init, nn, optim
from sutra.models import MLP

OPTIMIZERS = {
    "SGD": (lambda p, lr: optim.SGD(p, lr=lr), [0.01, 0.03, 0.1, 0.3, 1.0]),
    "SGD+momentum": (lambda p, lr: optim.SGD(p, lr=lr, momentum=0.9), [0.001, 0.003, 0.01, 0.03, 0.1]),
    "RMSprop": (lambda p, lr: optim.RMSprop(p, lr=lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "Adam": (lambda p, lr: optim.Adam(p, lr=lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "AdamW": (lambda p, lr: optim.AdamW(p, lr=lr, weight_decay=0.05), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
}


def study_optimizers(data, seeds=5, epochs=3):
    res = {}
    for name, (make, grid) in OPTIMIZERS.items():
        tuning = {}
        for lr in grid:  # tune on a dedicated seed, fewer epochs
            st.manual_seed(1000)
            m = MLP([784, 256, 256, 10])
            tuning[lr] = fit(m, make(m.parameters(), lr), data, epochs=2)[1]
        best_lr = max(tuning, key=tuning.get)
        accs, curves = [], []
        for s in range(seeds):
            st.manual_seed(s)
            m = MLP([784, 256, 256, 10])
            curve, acc, _ = fit(m, make(m.parameters(), best_lr), data, epochs=epochs)
            accs.append(acc)
            curves.append(curve)
        res[name] = {"lr_grid": {str(k): v for k, v in tuning.items()}, "best_lr": best_lr,
                     "val_acc": summarize(accs), "mean_curve": np.mean(curves, axis=0).tolist()}
        print(f"  {name:13s} best lr {best_lr:<7g} val acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
    best = max(res, key=lambda k: res[k]["val_acc"]["mean"])
    for name in res:
        if name != best:
            t, p = welch(res[best]["val_acc"]["values"], res[name]["val_acc"]["values"])
            res[name]["vs_best"] = {"best": best, "t": t, "p": p}
    return {"best": best, "epochs": epochs, "seeds": seeds, "results": res}


INITS = {
    "N(0, 0.01)": lambda s: init.normal(s, 0.0, 0.01),
    "N(0, 1)": lambda s: init.normal(s, 0.0, 1.0),
    "Xavier": init.xavier_normal,
    "Kaiming": init.kaiming_normal,
}


def deep_mlp(depth, width, act, init_fn):
    m = MLP([784] + [width] * (depth - 1) + [10], activation=act)
    for layer in m.net:
        if isinstance(layer, nn.Linear):
            layer.weight.data = init_fn(layer.weight.shape)
            layer.bias.data = init.zeros(layer.bias.shape)
    return m


def signal_stats(model, x, y):
    """Per-layer activation std (forward) and weight-gradient norm (backward) at initialisation."""
    acts, h = [], st.Tensor(x)
    for layer in model.net:
        h = layer(h)
        if not isinstance(layer, nn.Linear):
            acts.append(float(h.data.std()))
    loss = st.F.cross_entropy(h, y)
    model.zero_grad()
    loss.backward()
    grads = [float(np.linalg.norm(layer.weight.grad)) for layer in model.net if isinstance(layer, nn.Linear)]
    return acts, grads, float(loss.item())


def study_init(data, depth=20, width=256, seeds=3, epochs=2):
    x_probe, y_probe = data[0][:1000], data[1][:1000]
    res = {}
    for act_name, act in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        for init_name, fn in INITS.items():
            st.manual_seed(0)
            acts, grads, loss0 = signal_stats(deep_mlp(depth, width, act, fn), x_probe, y_probe)
            accs, diverged = [], 0
            for s in range(seeds):
                st.manual_seed(s)
                m = deep_mlp(depth, width, act, fn)
                _, acc, div = fit(m, optim.SGD(m.parameters(), lr=0.01, momentum=0.9), data, epochs=epochs)
                accs.append(acc)
                diverged += div
            key = f"{act_name} / {init_name}"
            res[key] = {"activation": act_name, "init": init_name, "act_std": acts, "grad_norm": grads,
                        "initial_loss": loss0, "val_acc": summarize(accs), "diverged": diverged}
            print(f"  {key:22s} last-layer act std {acts[-1]:.2e}  val acc {np.mean(accs):.4f}"
                  f"{'  (diverged x%d)' % diverged if diverged else ''}")
    return {"depth": depth, "width": width, "epochs": epochs, "seeds": seeds, "results": res}


def study_batchnorm(data, depth=10, width=256, seeds=3, epochs=2, lrs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    res = {}
    for bn in (False, True):
        rows = []
        for lr in lrs:
            accs, div = [], 0
            for s in range(seeds):
                st.manual_seed(s)
                m = MLP([784] + [width] * (depth - 1) + [10], batchnorm=bn)
                _, acc, d = fit(m, optim.SGD(m.parameters(), lr=lr, momentum=0.9), data, epochs=epochs)
                accs.append(acc)
                div += d
            rows.append({"lr": lr, "val_acc": summarize(accs), "diverged": div})
            print(f"  BN={bn!s:5s} lr {lr:<6g} val acc {np.mean(accs):.4f}{'  (diverged x%d)' % div if div else ''}")
        res["with BatchNorm" if bn else "no BatchNorm"] = rows
    return {"depth": depth, "width": width, "epochs": epochs, "seeds": seeds, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["A", "B", "C", "all"], default="all")
    args = ap.parse_args()
    data = mnist_flat()
    for key, name, fn in [("A", "ablation_optimizers", study_optimizers), ("B", "ablation_init", study_init),
                          ("C", "ablation_batchnorm", study_batchnorm)]:
        if args.study in (key, "all"):
            print(f"== Study {key}: {name}")
            t0 = time.time()
            out = fn(data)
            out["runtime_s"] = time.time() - t0
            save(name, out)


if __name__ == "__main__":
    main()
