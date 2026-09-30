"""Shared helpers for the experiments: data, a small training loop, and statistics."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sutra as st  # noqa: E402
from sutra.data import DataLoader, mnist  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def mnist_flat():
    """55k train / 5k validation split of MNIST, flattened and standardised."""
    x_tr, y_tr, _, _ = mnist()
    x = ((x_tr.astype(np.float32) / 255.0 - 0.1307) / 0.3081).reshape(len(x_tr), -1)
    return x[:55000], y_tr[:55000], x[55000:], y_tr[55000:]


@st.no_grad()
def accuracy(model, x, y, batch=2500):
    model.eval()
    pred = np.concatenate([model(st.Tensor(x[i:i + batch])).data.argmax(1) for i in range(0, len(x), batch)])
    model.train()
    return float((pred == y).mean())


def fit(model, opt, data, epochs, batch=128, log_every=50):
    """Minimal training loop. Returns (loss curve, final validation accuracy, diverged flag)."""
    x_tr, y_tr, x_val, y_val = data
    curve, window = [], []
    for _ in range(epochs):
        for xb, yb in DataLoader(x_tr, y_tr, batch_size=batch, shuffle=True, drop_last=True):
            loss = st.F.cross_entropy(model(xb), yb)
            if not np.isfinite(loss.item()):
                return curve, 0.1, True
            opt.zero_grad()
            loss.backward()
            opt.step()
            window.append(loss.item())
            if len(window) == log_every:
                curve.append(float(np.mean(window)))
                window = []
    return curve, accuracy(model, x_val, y_val), False


def summarize(xs):
    """Mean, sample std and 95% t-interval half-width."""
    from scipy import stats
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    std = float(xs.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": float(xs.mean()), "std": std, "ci95": half, "values": xs.tolist()}


def welch(a, b):
    """Welch's unequal-variance t-test. Returns (t statistic, two-sided p-value)."""
    from scipy import stats
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def save(name, obj):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1))
    print(f"saved {path}")
