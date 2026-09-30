"""
Train the DigitCNN on MNIST with Sutra.

    python examples/train_mnist.py --epochs 6

Writes runs/mnist/{model.npz, log.json}. Uses a 55k/5k train/validation split for model
selection; the 10k test set is touched once, at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sutra as st  # noqa: E402
from sutra import optim, serialize  # noqa: E402
from sutra.data import DataLoader, mnist  # noqa: E402
from sutra.models import DigitCNN  # noqa: E402

MEAN, STD = 0.1307, 0.3081


def preprocess(x):
    return ((x.astype(np.float32) / 255.0 - MEAN) / STD)[:, None, :, :]


def random_shift(x, y, max_shift=2):
    """Augmentation: translate each image by up to ±max_shift pixels (pad + random crop)."""
    n = len(x)
    p = max_shift
    padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=(-MEAN / STD))
    dx, dy = st.rng().integers(0, 2 * p + 1, n), st.rng().integers(0, 2 * p + 1, n)
    out = np.empty_like(x)
    for i in range(n):
        out[i] = padded[i, :, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]
    return out, y


@st.no_grad()
def predict(model, x, batch=1000):
    model.eval()
    logits = np.concatenate([model(st.Tensor(x[i:i + batch])).data for i in range(0, len(x), batch)])
    model.train()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "mnist"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st.manual_seed(args.seed)

    x_tr, y_tr, x_te, y_te = mnist()
    x_tr, x_te = preprocess(x_tr), preprocess(x_te)
    x_val, y_val = x_tr[55000:], y_tr[55000:]
    x_tr, y_tr = x_tr[:55000], y_tr[:55000]

    model = DigitCNN()
    print(model)
    print(f"parameters: {model.num_parameters():,}")
    loader = DataLoader(x_tr, y_tr, batch_size=args.batch, shuffle=True, drop_last=True, transform=random_shift)
    total = args.epochs * len(loader)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.CosineWarmup(opt, warmup_steps=100, total_steps=total, min_ratio=0.02)

    log = {"config": vars(args), "params": model.num_parameters(), "steps": [], "epochs": []}
    step, best_val, t0 = 0, 0.0, time.time()
    for epoch in range(1, args.epochs + 1):
        run_loss, run_acc = [], []
        for xb, yb in loader:
            logits = model(xb)
            loss = st.F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss.append(loss.item())
            run_acc.append(float((logits.data.argmax(1) == yb.data).mean()))
            if step % 50 == 0:
                log["steps"].append({"step": step, "loss": float(np.mean(run_loss[-50:])),
                                     "acc": float(np.mean(run_acc[-50:])), "lr": opt.lr})
                print(f"epoch {epoch} step {step}/{total} loss {np.mean(run_loss[-50:]):.4f} "
                      f"acc {np.mean(run_acc[-50:]):.3f} lr {opt.lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
        val_acc = float((predict(model, x_val).argmax(1) == y_val).mean())
        log["epochs"].append({"epoch": epoch, "train_loss": float(np.mean(run_loss)), "val_acc": val_acc,
                              "time": time.time() - t0})
        print(f"== epoch {epoch}: val acc {val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            serialize.save(model, out / "model.npz", mean=MEAN, std=STD)

    serialize.load(model, out / "model.npz")
    pred = predict(model, x_te).argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_te, pred), 1)
    log.update(test_acc=float((pred == y_te).mean()), best_val_acc=best_val, confusion=cm.tolist(),
               train_time_s=time.time() - t0, misclassified=np.flatnonzero(pred != y_te)[:64].tolist())
    print(f"TEST ACCURACY: {log['test_acc']:.4f}  ({(pred != y_te).sum()} errors / 10000)")
    (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
