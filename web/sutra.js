/*
 * sutra.js — a zero-dependency inference engine for models trained with Sutra.
 *
 * Loads the int8/float32 bundles written by web/export.py, dequantises them, and runs the
 * DigitCNN and the GPT (with a KV cache) in plain JavaScript on typed arrays. The GPT runs both the
 * GPT-2-style flagship (LayerNorm, learned positions, GELU) and the LLaMA-style one
 * (RMSNorm, rotary embeddings, SwiGLU, no biases).
 */
(function (global) {
  "use strict";

  async function loadBundle(base) {
    const [manifest, buf] = await Promise.all([
      fetch(base + ".json").then((r) => { if (!r.ok) throw new Error(base + ".json " + r.status); return r.json(); }),
      fetch(base + ".bin").then((r) => { if (!r.ok) throw new Error(base + ".bin " + r.status); return r.arrayBuffer(); }),
    ]);
    const tensors = {};
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      let data;
      if (t.dtype === "int8") {
        const q = new Int8Array(buf, t.offset, n);
        const rows = t.shape[0], cols = n / rows;
        const s = new Float32Array(buf, t.scale_offset, rows);
        data = new Float32Array(n);
        for (let r = 0; r < rows; r++) {
          const sc = s[r], o = r * cols;
          for (let c = 0; c < cols; c++) data[o + c] = q[o + c] * sc;
        }
      } else {
        data = new Float32Array(buf, t.offset, n).slice();
      }
      tensors[t.name] = { shape: t.shape, data };
    }
    return { manifest, tensors, bytes: buf.byteLength };
  }

  // ------------------------------------------------------------------ kernels

  /** y[M,N] = x[M,K] · W[N,K]^T + b   (PyTorch weight layout) */
  function linear(x, M, K, W, b, N) {
    const y = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      const xo = m * K;
      for (let n = 0; n < N; n++) {
        const wo = n * K;
        let s = b ? b[n] : 0;
        for (let k = 0; k < K; k++) s += x[xo + k] * W[wo + k];
        y[m * N + n] = s;
      }
    }
    return y;
  }

  function layerNorm(x, D, w, b, eps = 1e-5) {
    const y = new Float32Array(x.length);
    for (let o = 0; o < x.length; o += D) {
      let mu = 0;
      for (let i = 0; i < D; i++) mu += x[o + i];
      mu /= D;
      let v = 0;
      for (let i = 0; i < D; i++) { const d = x[o + i] - mu; v += d * d; }
      const r = 1 / Math.sqrt(v / D + eps);
      for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mu) * r * w[i] + b[i];
    }
    return y;
  }

  function rmsNorm(x, D, w, eps = 1e-5) {
    const y = new Float32Array(x.length);
    for (let o = 0; o < x.length; o += D) {
      let ms = 0;
      for (let i = 0; i < D; i++) ms += x[o + i] * x[o + i];
      const r = 1 / Math.sqrt(ms / D + eps);
      for (let i = 0; i < D; i++) y[o + i] = x[o + i] * r * w[i];
    }
    return y;
  }

  /** Rotary embedding in place on one head vector (rotate-half layout, matching sutra.functional.apply_rope). */
  function ropeInPlace(v, offset, hd, pos, base = 10000) {
    const h = hd >> 1;
    for (let i = 0; i < h; i++) {
      const ang = pos * Math.pow(base, (-2 * i) / hd), c = Math.cos(ang), s = Math.sin(ang);
      const x1 = v[offset + i], x2 = v[offset + i + h];
      v[offset + i] = x1 * c - x2 * s;
      v[offset + i + h] = x2 * c + x1 * s;
    }
  }

  function siluInPlace(x) {
    for (let i = 0; i < x.length; i++) x[i] = x[i] / (1 + Math.exp(-x[i]));
    return x;
  }

  function geluInPlace(x) {
    const c = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      x[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }
    return x;
  }

  function reluInPlace(x) {
    for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
    return x;
  }

  function softmax(x) {
    let m = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    const p = new Float32Array(x.length);
    let s = 0;
    for (let i = 0; i < x.length; i++) { p[i] = Math.exp(x[i] - m); s += p[i]; }
    for (let i = 0; i < x.length; i++) p[i] /= s;
    return p;
  }

  /** stride-1 "same" convolution: x[C,H,W] * w[O,C,k,k] -> [O,H,W] */
  function conv2dSame(x, C, H, W, w, b, O, k) {
    const pad = (k - 1) >> 1, y = new Float32Array(O * H * W);
    for (let o = 0; o < O; o++) {
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          let s = b[o];
          for (let c = 0; c < C; c++) {
            for (let di = 0; di < k; di++) {
              const ii = i + di - pad;
              if (ii < 0 || ii >= H) continue;
              for (let dj = 0; dj < k; dj++) {
                const jj = j + dj - pad;
                if (jj < 0 || jj >= W) continue;
                s += x[(c * H + ii) * W + jj] * w[((o * C + c) * k + di) * k + dj];
              }
            }
          }
          y[(o * H + i) * W + j] = s;
        }
      }
    }
    return y;
  }

  function maxPool2(x, C, H, W) {
    const h = H >> 1, w = W >> 1, y = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++)
      for (let i = 0; i < h; i++)
        for (let j = 0; j < w; j++) {
          const o = (c * H + 2 * i) * W + 2 * j;
          y[(c * h + i) * w + j] = Math.max(x[o], x[o + 1], x[o + W], x[o + W + 1]);
        }
    return y;
  }

  // ------------------------------------------------------------------ models

  class DigitCNN {
    constructor(bundle) {
      this.t = bundle.tensors;
      this.mean = bundle.manifest.mean;
      this.std = bundle.manifest.std;
    }

    /** pixels: 784 values in [0, 255]. Returns logits, probabilities and feature maps. */
    forward(pixels) {
      const t = this.t, x = new Float32Array(784);
      for (let i = 0; i < 784; i++) x[i] = (pixels[i] / 255 - this.mean) / this.std;
      const c1 = reluInPlace(conv2dSame(x, 1, 28, 28, t["conv1.weight"].data, t["conv1.bias"].data, 16, 3));
      const p1 = maxPool2(c1, 16, 28, 28);
      const c2 = reluInPlace(conv2dSame(p1, 16, 14, 14, t["conv2.weight"].data, t["conv2.bias"].data, 32, 3));
      const p2 = maxPool2(c2, 32, 14, 14);
      const h = reluInPlace(linear(p2, 1, 1568, t["fc1.weight"].data, t["fc1.bias"].data, 128));
      const logits = linear(h, 1, 128, t["fc2.weight"].data, t["fc2.bias"].data, 10);
      return { logits, probs: softmax(logits), conv1: c1, conv2: c2 };
    }
  }

  class CharTokenizer {
    constructor(chars) {
      this.chars = [...chars];
      this.stoi = new Map(this.chars.map((c, i) => [c, i]));
    }
    encode(s) { const out = []; for (const ch of s) if (this.stoi.has(ch)) out.push(this.stoi.get(ch)); return out; }
    decode(ids) { return ids.map((i) => this.chars[i]).join(""); }
  }

  class GPT {
    constructor(bundle) {
      this.t = bundle.tensors;
      // older bundles predate these fields: they are GPT-2 style
      this.cfg = Object.assign({ norm: "layer", pos: "learned", mlp: "gelu", bias: true }, bundle.manifest.config);
      this.tok = new CharTokenizer(bundle.manifest.tokenizer.chars);
      const L = this.cfg.n_layer, T = this.cfg.block_size, D = this.cfg.n_embd;
      this.K = Array.from({ length: L }, () => new Float32Array(T * D));
      this.V = Array.from({ length: L }, () => new Float32Array(T * D));
      this.history = [];
      this.pos = 0;
    }

    reset() { this.pos = 0; this.history = []; }

    p(name) { return this.t[name] ? this.t[name].data : null; }

    norm(x, prefix) {
      const D = this.cfg.n_embd;
      return this.cfg.norm === "rms" ? rmsNorm(x, D, this.p(prefix + ".weight"))
        : layerNorm(x, D, this.p(prefix + ".weight"), this.p(prefix + ".bias") || new Float32Array(D));
    }

    mlp(h, pre) {
      const D = this.cfg.n_embd;
      if (this.cfg.mlp === "swiglu") {
        const H = this.t[pre + "mlp.gate.weight"].shape[0];
        const g = siluInPlace(linear(h, 1, D, this.p(pre + "mlp.gate.weight"), this.p(pre + "mlp.gate.bias"), H));
        const u = linear(h, 1, D, this.p(pre + "mlp.up.weight"), this.p(pre + "mlp.up.bias"), H);
        for (let i = 0; i < H; i++) g[i] *= u[i];
        return linear(g, 1, H, this.p(pre + "mlp.proj.weight"), this.p(pre + "mlp.proj.bias"), D);
      }
      const m = geluInPlace(linear(h, 1, D, this.p(pre + "mlp.fc.weight"), this.p(pre + "mlp.fc.bias"), 4 * D));
      return linear(m, 1, 4 * D, this.p(pre + "mlp.proj.weight"), this.p(pre + "mlp.proj.bias"), D);
    }

    /** Feed one token. Returns next-token logits and attention weights [layer][head] over the cached context. */
    step(token) {
      const { n_layer: L, n_head: H, n_embd: D, block_size: T, vocab_size: Vs } = this.cfg;
      this.history.push(token);
      if (this.pos >= T) {  // context full: slide the window by re-encoding the last T-1 tokens
        const keep = this.history.slice(-(T - 1));
        this.history = [];
        this.pos = 0;
        let out;
        for (const tk of keep) out = this.step(tk);
        return out;
      }
      const hd = D / H, scale = 1 / Math.sqrt(hd), pos = this.pos;
      const x = new Float32Array(D), wte = this.p("wte.weight"), wpe = this.p("wpe.weight");
      for (let i = 0; i < D; i++) x[i] = wte[token * D + i] + (wpe ? wpe[pos * D + i] : 0);
      const attn = [];
      for (let l = 0; l < L; l++) {
        const pre = "blocks." + l + ".";
        const h = this.norm(x, pre + "ln1");
        const qkv = linear(h, 1, D, this.p(pre + "attn.qkv.weight"), this.p(pre + "attn.qkv.bias"), 3 * D);
        if (this.cfg.pos === "rope") {
          for (let hh = 0; hh < H; hh++) { ropeInPlace(qkv, hh * hd, hd, pos); ropeInPlace(qkv, D + hh * hd, hd, pos); }
        }
        const Kc = this.K[l], Vc = this.V[l];
        Kc.set(qkv.subarray(D, 2 * D), pos * D);
        Vc.set(qkv.subarray(2 * D, 3 * D), pos * D);
        const y = new Float32Array(D), heads = [];
        for (let hh = 0; hh < H; hh++) {
          const o = hh * hd, sc = new Float32Array(pos + 1);
          for (let t = 0; t <= pos; t++) {
            let s = 0;
            for (let i = 0; i < hd; i++) s += qkv[o + i] * Kc[t * D + o + i];
            sc[t] = s * scale;
          }
          const pr = softmax(sc);
          for (let t = 0; t <= pos; t++) {
            const w = pr[t];
            for (let i = 0; i < hd; i++) y[o + i] += w * Vc[t * D + o + i];
          }
          heads.push(pr);
        }
        attn.push(heads);
        const a = linear(y, 1, D, this.p(pre + "attn.proj.weight"), this.p(pre + "attn.proj.bias"), D);
        for (let i = 0; i < D; i++) x[i] += a[i];
        if (this.t[pre + "ln2.weight"]) {  // attention-only models have no MLP
          const mo = this.mlp(this.norm(x, pre + "ln2"), pre);
          for (let i = 0; i < D; i++) x[i] += mo[i];
        }
      }
      const hf = this.norm(x, "ln_f");
      const logits = linear(hf, 1, D, wte, null, Vs);  // weight-tied head
      this.pos++;
      return { logits, attn, ctxLen: this.pos };  // attention covers the last ctxLen tokens
    }

    /** Sample from logits with temperature and top-k. temperature 0 = greedy. */
    static sample(logits, temperature, topK, rand = Math.random) {
      if (temperature <= 0) {
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
      }
      const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK || logits.length);
      const z = idx.map((i) => logits[i] / temperature), m = Math.max(...z);
      const e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
      let r = rand() * s;
      for (let j = 0; j < idx.length; j++) { r -= e[j]; if (r <= 0) return idx[j]; }
      return idx[idx.length - 1];
    }
  }

  global.Sutra = { loadBundle, DigitCNN, GPT, CharTokenizer,
    kernels: { linear, layerNorm, rmsNorm, softmax, conv2dSame, maxPool2, ropeInPlace } };
})(typeof window !== "undefined" ? window : globalThis);
