"""
What does int8 weight quantisation cost? Evaluates the trained CNN (MNIST test set) and the flagship
GPT (Shakespeare validation set) before and after a per-row symmetric int8 round-trip.

    python experiments/quantization.py
"""
import math
import sys

import numpy as np
from common import ROOT, save, st

from sutra import quantize, serialize
from sutra.data import get_lm_batch, mnist, tiny_shakespeare
from sutra.models import GPT, DigitCNN, GPTConfig
from sutra.tokenizer import tokenizer_from_dict

sys.path.insert(0, str(ROOT / "examples"))
from train_mnist import predict, preprocess  # noqa: E402


def eval_cnn(model, x, y):
    return float((predict(model, x).argmax(1) == y).mean())


@st.no_grad()
def eval_gpt(model, val_ids, n_batches=100):
    model.eval()
    st.manual_seed(4321)
    losses = [model(*get_lm_batch(val_ids, 32, model.cfg.block_size))[1].item() for _ in range(n_batches)]
    return float(np.mean(losses))


def main():
    out = {}
    _, _, x_te, y_te = mnist()
    x_te = preprocess(x_te)
    cnn = DigitCNN()
    serialize.load(cnn, ROOT / "runs" / "mnist" / "model.npz")
    sd = cnn.state_dict()
    fp32 = eval_cnn(cnn, x_te, y_te)
    errs = quantize.simulate_int8(cnn)
    q8 = eval_cnn(cnn, x_te, y_te)
    out["cnn"] = {"metric": "test accuracy", "fp32": fp32, "int8": q8,
                  "bytes_fp32": quantize.size_bytes(sd, False), "bytes_int8": quantize.size_bytes(sd, True),
                  "max_rel_weight_error": max(errs.values())}
    print(f"CNN  test acc  fp32 {fp32:.4f}  int8 {q8:.4f}   size {out['cnn']['bytes_fp32'] / 1e6:.2f} MB -> "
          f"{out['cnn']['bytes_int8'] / 1e6:.2f} MB")

    sdg, meta = serialize.load_state(ROOT / "runs" / "gpt-char" / "model.npz")
    gpt = GPT(GPTConfig(**meta["config"]))
    gpt.load_state_dict(sdg)
    tok = tokenizer_from_dict(meta["tokenizer"])
    text = tiny_shakespeare()
    val_ids = np.array(tok.encode(text[int(0.9 * len(text)):]), dtype=np.int64)
    fp32 = eval_gpt(gpt, val_ids)
    quantize.simulate_int8(gpt)
    q8 = eval_gpt(gpt, val_ids)
    out["gpt"] = {"metric": "validation loss (nats/char)", "fp32": fp32, "int8": q8,
                  "fp32_bpc": fp32 / math.log(2), "int8_bpc": q8 / math.log(2),
                  "bytes_fp32": quantize.size_bytes(sdg, False), "bytes_int8": quantize.size_bytes(sdg, True)}
    print(f"GPT  val loss  fp32 {fp32:.4f}  int8 {q8:.4f}   size {out['gpt']['bytes_fp32'] / 1e6:.2f} MB -> "
          f"{out['gpt']['bytes_int8'] / 1e6:.2f} MB")
    save("quantization", out)


if __name__ == "__main__":
    main()
