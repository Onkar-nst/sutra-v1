/* app.js — wires the Sutra demo page: self-test, digit pad, GPT writer, attention inspector, experiment figures. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const pct = (v, d = 2) => (100 * v).toFixed(d) + "%";
  let demo, cnn, gptBundle, gpt;

  function hexToRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  async function main() {
    try {
      const [d, cb, gb] = await Promise.all([
        fetch("assets/demo.json").then((r) => { if (!r.ok) throw new Error("demo.json " + r.status); return r.json(); }),
        Sutra.loadBundle("assets/cnn"), Sutra.loadBundle("assets/gpt")]);
      demo = d; cnn = new Sutra.DigitCNN(cb); gptBundle = gb; gpt = new Sutra.GPT(gb);
    } catch (e) {
      $("#selftest").classList.add("fail");
      $("#selftest b").textContent = "could not load model files (" + e.message + ")";
      $("#hero-rate").textContent = "unavailable";
      return;
    }
    bindNumbers();
    setupDigits();
    setupGPT();
    renderExperiments();
    setTimeout(selfTest, 50);
    heroWrite();
  }

  // ------------------------------------------------------------------ numbers in the copy
  function bindNumbers() {
    const r = demo.results, cfg = gptBundle.manifest.config;
    const set = (k, v) => document.querySelectorAll(`[data-bind="${k}"]`).forEach((e) => { e.textContent = v; });
    if (r.mnist_log) { set("mnist_acc", pct(r.mnist_log.test_acc)); set("cnn_params", r.mnist_log.params.toLocaleString()); }
    if (r.gpt_log) { set("bpc", r.gpt_log.best_val_bpc.toFixed(2)); set("gpt_params", (r.gpt_log.params / 1e6).toFixed(1) + "M"); }
    if (demo.tests) set("tests", String(demo.tests));
    if (demo.engine_lines) set("engine_lines", String(demo.engine_lines));
    set("gpt_arch", `${cfg.n_layer}-layer, ${cfg.n_head}-head, ${cfg.n_embd}-wide`);
  }

  // ------------------------------------------------------------------ self-test against Python
  function selfTest() {
    let maxd = 0;
    demo.cnn.golden_logits.forEach((g, i) => {
      const out = cnn.forward(demo.cnn.samples[i].pixels).logits;
      g.forEach((v, j) => { maxd = Math.max(maxd, Math.abs(v - out[j])); });
    });
    const m = new Sutra.GPT(gptBundle), ids = m.tok.encode(demo.gpt.prompt);
    let out;
    for (const id of ids) out = m.step(id);
    demo.gpt.last_logits.forEach((v, j) => { maxd = Math.max(maxd, Math.abs(v - out.logits[j])); });
    const greedy = [];
    for (let i = 0; i < demo.gpt.greedy.length; i++) {
      const nx = Sutra.GPT.sample(out.logits, 0);
      greedy.push(nx);
      out = m.step(nx);
    }
    const same = m.tok.decode(greedy) === demo.gpt.greedy;
    const ok = maxd < 1e-3 && same;
    $("#selftest").classList.toggle("fail", !ok);
    $("#selftest b").textContent = (ok ? "✓ matches the Python reference" : "✗ mismatch with the Python reference") +
      ` (max |Δ logit| ${maxd.toExponential(1)}, greedy text ${same ? "identical" : "differs"})`;
  }

  // ------------------------------------------------------------------ hero writer
  let heroToken = 0;
  function heroWrite() {
    const myToken = ++heroToken, m = new Sutra.GPT(gptBundle), pre = $("#hero-text");
    let text = "", out = m.step(m.tok.encode("\n")[0]), n = 0, spent = 0;
    const total = 300;
    const caret = '<span class="caret"></span>';
    const tick = () => {
      if (myToken !== heroToken) return;
      const batch = reduceMotion ? total : 2;
      for (let k = 0; k < batch && n < total; k++, n++) {
        const t0 = performance.now();
        const nx = Sutra.GPT.sample(out.logits, 0.8, 40);
        out = m.step(nx);
        spent += performance.now() - t0;
        text += m.tok.chars[nx];
      }
      pre.textContent = text;
      pre.insertAdjacentHTML("beforeend", caret);
      $("#hero-rate").textContent = `${(spent / n).toFixed(2)} ms per character`;
      if (n < total) setTimeout(tick, 28);
    };
    tick();
  }
  $("#hero-again").addEventListener("click", heroWrite);

  // ------------------------------------------------------------------ digits
  let pixels = null;
  function setupDigits() {
    const pad = $("#pad"), ctx = pad.getContext("2d");
    let drawing = false, last = null, queued = false;
    const pos = (ev) => {
      const r = pad.getBoundingClientRect();
      return [(ev.clientX - r.left) * (pad.width / r.width), (ev.clientY - r.top) * (pad.height / r.height)];
    };
    const schedule = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; pixels = preprocess(pad); classify(); });
    };
    pad.addEventListener("pointerdown", (ev) => {
      drawing = true; last = pos(ev); pad.setPointerCapture(ev.pointerId);
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(last[0], last[1], 10, 0, 2 * Math.PI); ctx.fill();
      schedule();
    });
    pad.addEventListener("pointermove", (ev) => {
      if (!drawing) return;
      const p = pos(ev);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.beginPath(); ctx.moveTo(last[0], last[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
      last = p;
      schedule();
    });
    const stop = () => { drawing = false; };
    pad.addEventListener("pointerup", stop);
    pad.addEventListener("pointercancel", stop);
    $("#clear").addEventListener("click", () => { ctx.clearRect(0, 0, pad.width, pad.height); pixels = null; classify(); });

    const probs = $("#probs");
    for (let d = 0; d < 10; d++) {
      probs.insertAdjacentHTML("beforeend", `<div class="prob" id="p${d}"><span>${d}</span><div class="track"><div class="fill" style="width:0"></div></div><span class="v">–</span></div>`);
    }
    const maps = $("#maps");
    for (let i = 0; i < 16; i++) {
      const c = document.createElement("canvas");
      c.width = 28; c.height = 28; c.setAttribute("aria-label", "feature map " + (i + 1));
      maps.appendChild(c);
    }
    const samples = $("#samples");
    demo.cnn.samples.slice(0, 20).forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.title = "Test image of a " + s.label;
      b.setAttribute("aria-label", "Load test image " + (i + 1) + ", a " + s.label);
      const c = document.createElement("canvas");
      c.width = 28; c.height = 28;
      c.getContext("2d").putImageData(toImage(c.getContext("2d"), s.pixels), 0, 0);
      b.appendChild(c);
      b.addEventListener("click", () => loadSample(s.pixels));
      samples.appendChild(b);
    });
    loadSample(demo.cnn.samples[0].pixels);
  }

  function toImage(ctx, px) {
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) { img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = 255; img.data[4 * i + 3] = px[i]; }
    return img;
  }

  function loadSample(px) {
    const pad = $("#pad"), ctx = pad.getContext("2d"), tmp = document.createElement("canvas");
    tmp.width = 28; tmp.height = 28;
    tmp.getContext("2d").putImageData(toImage(tmp.getContext("2d"), px), 0, 0);
    ctx.clearRect(0, 0, pad.width, pad.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, pad.width, pad.height);
    pixels = Float32Array.from(px);
    classify();
  }

  /** MNIST-style normalisation: crop ink, fit the long side to 20 px, centre by centre of mass in 28x28. */
  function preprocess(pad) {
    const W = pad.width, H = pad.height, a = pad.getContext("2d").getImageData(0, 0, W, H).data;
    let x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (a[4 * (y * W + x) + 3] > 20) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    if (x1 < 0) return null;
    const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
    const tmp = document.createElement("canvas");
    tmp.width = 28; tmp.height = 28;
    const t = tmp.getContext("2d");
    t.imageSmoothingEnabled = true; t.imageSmoothingQuality = "high";
    t.drawImage(pad, x0, y0, w, h, (28 - w * s) / 2, (28 - h * s) / 2, w * s, h * s);
    const d = t.getImageData(0, 0, 28, 28).data, v = new Float32Array(784);
    let m = 0, cx = 0, cy = 0;
    for (let i = 0; i < 784; i++) { v[i] = d[4 * i + 3]; m += v[i]; cx += v[i] * (i % 28); cy += v[i] * Math.floor(i / 28); }
    const dx = Math.round(14 - cx / m), dy = Math.round(14 - cy / m), out = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const sx = x - dx, sy = y - dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = v[sy * 28 + sx];
    }
    return out;
  }

  function classify() {
    const seen = $("#seen").getContext("2d");
    seen.clearRect(0, 0, 28, 28);
    if (!pixels) {
      $("#guess").firstChild.textContent = "–";
      for (let d = 0; d < 10; d++) { const r = $("#p" + d); r.classList.remove("top"); r.querySelector(".fill").style.width = "0"; r.querySelector(".v").textContent = "–"; }
      document.querySelectorAll("#maps canvas").forEach((c) => c.getContext("2d").clearRect(0, 0, 28, 28));
      return;
    }
    seen.putImageData(toImage(seen, pixels), 0, 0);
    const res = cnn.forward(pixels);
    let top = 0;
    for (let d = 1; d < 10; d++) if (res.probs[d] > res.probs[top]) top = d;
    $("#guess").firstChild.textContent = String(top);
    for (let d = 0; d < 10; d++) {
      const r = $("#p" + d);
      r.classList.toggle("top", d === top);
      r.querySelector(".fill").style.width = (100 * res.probs[d]).toFixed(1) + "%";
      r.querySelector(".v").textContent = res.probs[d] >= 0.001 ? pct(res.probs[d], 1) : "<0.1%";
    }
    const lo = hexToRgb(cssVar("--panel") || "#ffffff"), hi = hexToRgb(cssVar("--indigo") || "#3438a8");
    document.querySelectorAll("#maps canvas").forEach((c, k) => {
      const g = c.getContext("2d"), img = g.createImageData(28, 28), off = k * 784;
      let mx = 1e-6;
      for (let i = 0; i < 784; i++) mx = Math.max(mx, res.conv1[off + i]);
      for (let i = 0; i < 784; i++) {
        const t = res.conv1[off + i] / mx;
        for (let ch = 0; ch < 3; ch++) img.data[4 * i + ch] = lo[ch] + (hi[ch] - lo[ch]) * t;
        img.data[4 * i + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    });
  }

  // ------------------------------------------------------------------ GPT writer + attention inspector
  let records = [], running = false;
  function setupGPT() {
    const cfg = gptBundle.manifest.config;
    const layer = $("#attn-layer"), head = $("#attn-head");
    for (let l = 0; l < cfg.n_layer; l++) layer.insertAdjacentHTML("beforeend", `<option value="${l}">${l + 1}</option>`);
    layer.value = String(cfg.n_layer - 1);
    head.insertAdjacentHTML("beforeend", `<option value="-1">mean</option>`);
    for (let h = 0; h < cfg.n_head; h++) head.insertAdjacentHTML("beforeend", `<option value="${h}">${h + 1}</option>`);
    for (const [id, out] of [["temp", "temp-out"], ["topk", "topk-out"], ["len", "len-out"]]) {
      $("#" + id).addEventListener("input", (e) => { $("#" + out).textContent = e.target.value; });
    }
    $("#gen-form").addEventListener("submit", (e) => { e.preventDefault(); running ? (running = false) : generate(); });
    const outEl = $("#gen-out");
    outEl.addEventListener("pointerover", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i !== undefined) inspect(+i); });
    outEl.addEventListener("focusin", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i !== undefined) inspect(+i); });
    layer.addEventListener("change", () => lastInspected >= 0 && inspect(lastInspected));
    head.addEventListener("change", () => lastInspected >= 0 && inspect(lastInspected));
    generate(true);
  }

  function addChar(ch, rec, isPrompt) {
    const s = document.createElement("span");
    s.textContent = ch;
    s.dataset.i = String(records.length);
    if (isPrompt) s.className = "prompt";
    records.push({ ch, attn: rec.attn, ctxLen: rec.ctxLen, el: s });
    $("#gen-out").appendChild(s);
  }

  function generate(initial) {
    const outEl = $("#gen-out"), btn = $("#gen-btn");
    outEl.textContent = "";
    records = [];
    lastInspected = -1;
    gpt.reset();
    const prompt = $("#prompt").value || "\n";
    const ids = gpt.tok.encode(prompt);
    let out = null;
    for (const id of ids) { out = gpt.step(id); addChar(gpt.tok.chars[id], out, true); }
    if (!out) { out = gpt.step(gpt.tok.encode("\n")[0]); addChar("\n", out, true); }
    const n = initial ? 400 : +$("#len").value, T = +$("#temp").value, K = +$("#topk").value;
    let made = 0, spent = 0;
    running = true;
    btn.textContent = "Stop";
    const chunk = () => {
      if (!running) return finish();
      const t0 = performance.now();
      while (made < n && performance.now() - t0 < 14) {
        const nx = Sutra.GPT.sample(out.logits, T, K);
        out = gpt.step(nx);
        addChar(gpt.tok.chars[nx], out, false);
        made++;
      }
      spent += performance.now() - t0;
      $("#gen-rate").textContent = `${Math.round((made / spent) * 1000)} chars/s`;
      if (made < n) requestAnimationFrame(chunk); else finish();
    };
    const finish = () => { running = false; btn.textContent = "Generate"; if (initial) inspect(records.length - 1); };
    requestAnimationFrame(chunk);
  }

  let lastInspected = -1, lit = [];
  const show = (c) => (c === "\n" ? "↵" : c === " " ? "␣" : c);
  function inspect(i) {
    const rec = records[i];
    if (!rec) return;
    lastInspected = i;
    for (const s of lit) { s.style.backgroundColor = ""; s.classList.remove("focus"); }
    lit = [];
    const L = +$("#attn-layer").value, H = +$("#attn-head").value, heads = rec.attn[L];
    const n = rec.ctxLen, w = new Float32Array(n);
    if (H < 0) { for (const h of heads) for (let t = 0; t < n; t++) w[t] += h[t] / heads.length; }
    else w.set(heads[H].subarray(0, n));
    // the attention row for position i covers the ctxLen most recent characters, ending at i
    const start = i - n + 1;
    let mx = 1e-9;
    for (let t = 0; t < n; t++) mx = Math.max(mx, w[t]);
    const top = [];
    for (let t = 0; t < n; t++) {
      const r = records[start + t];
      if (!r) continue;
      const a = Math.round((w[t] / mx) * 60);
      if (a > 2) { r.el.style.backgroundColor = `color-mix(in srgb, var(--indigo) ${a}%, transparent)`; lit.push(r.el); }
      top.push([w[t], r.ch]);
    }
    rec.el.classList.add("focus");
    lit.push(rec.el);
    top.sort((a, b) => b[0] - a[0]);
    const desc = top.slice(0, 3).map(([v, c]) => `“${show(c)}” ${(100 * v).toFixed(0)}%`).join(", ");
    $("#attn-note").textContent = `“${show(rec.ch)}” (layer ${L + 1}, ${H < 0 ? "mean of heads" : "head " + (H + 1)}) attends to ${desc}`;
  }

  // ------------------------------------------------------------------ experiments
  function fig(opts) {
    const f = document.createElement("figure");
    f.className = "fig" + (opts.wide ? " wide" : "");
    f.style.margin = "0";
    f.innerHTML = `<div class="eyebrow">${opts.eyebrow}</div><h3>${opts.title}</h3><p class="finding">${opts.finding}</p>`;
    $("#figs").appendChild(f);
    return f;
  }
  function chartIn(f) { const c = document.createElement("div"); c.className = "chart"; f.appendChild(c); return c; }
  function note(f, html) { const p = document.createElement("p"); p.className = "method"; p.innerHTML = html; f.appendChild(p); }
  function table(f, head, rows) {
    const d = document.createElement("div");
    d.className = "table-scroll";
    d.innerHTML = `<table><thead><tr>${head.map((h) => `<th class="${h.num ? "num" : ""}">${h.t}</th>`).join("")}</tr></thead><tbody>` +
      rows.map((r) => `<tr>${r.map((c, j) => `<td class="${head[j].num ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    f.appendChild(d);
  }

  function renderExperiments() {
    const R = demo.results;
    const g = R.gpt_log;
    if (g) {
      const f = fig({ wide: true, eyebrow: "Flagship model", title: "GPT training on Tiny Shakespeare",
        finding: `Validation loss reached <b>${g.best_val.toFixed(3)}</b> nats per character (<b>${g.best_val_bpc.toFixed(2)}</b> bits per character) after ${g.best_step.toLocaleString()} steps. Training took ${(g.train_time_s / 60).toFixed(0)} minutes on a laptop CPU, about ${Math.round(g.steps[g.steps.length - 1].tok_per_s).toLocaleString()} tokens per second.` });
      Charts.lineChart(chartIn(f), { series: [
        { name: "train", color: "--s1", points: g.evals.map((e) => [e.step, e.train]) },
        { name: "validation", color: "--s2", points: g.evals.map((e) => [e.step, e.val]) }],
        xLabel: "training step", yLabel: "loss (nats/char)", endLabels: true, height: 250 });
      note(f, `${(g.params / 1e6).toFixed(2)}M parameters, context ${g.model.block_size}, batch ${g.config.batch}, AdamW with cosine decay, dropout ${g.config.dropout}, gradient clipping at ${g.config.grad_clip}. Losses are averaged over ${g.config.eval_iters} fixed batches. Sample from the best checkpoint:`);
      const pre = document.createElement("pre"); pre.className = "samples-text"; pre.textContent = g.samples[0].trim(); f.appendChild(pre);
    }

    const m = R.mnist_log;
    if (m) {
      const f = fig({ eyebrow: "Demo 1 model", title: "CNN training on MNIST",
        finding: `<b>${pct(m.test_acc)}</b> test accuracy (${Math.round((1 - m.test_acc) * 10000)} errors in 10,000) after ${m.config.epochs} epochs and ${(m.train_time_s / 60).toFixed(1)} minutes of training.` });
      Charts.lineChart(chartIn(f), { series: [{ name: "training loss", color: "--s1", points: m.steps.map((s) => [s.step, s.loss]) }],
        xLabel: "training step", yLabel: "cross-entropy", yLog: true, height: 230 });
      note(f, "AdamW with warmup and cosine decay, ±2 px random-shift augmentation, a 55k/5k train/validation split for model selection, and the test set used once.");
      const cm = m.confusion, off = [];
      cm.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v) off.push([v, i, j]); }));
      off.sort((a, b) => b[0] - a[0]);
      const f2 = fig({ eyebrow: "Error analysis", title: "Where the errors went",
        finding: `The misclassifications only, by true and predicted digit. The most common confusion is a <b>${off[0][1]}</b> read as a <b>${off[0][2]}</b> (${off[0][0]} images), then ${off[1][1]} as ${off[1][2]} (${off[1][0]}).` });
      Charts.heatmap(chartIn(f2), { matrix: cm, labels: [..."0123456789"], skipDiagonal: true, rowLabel: "true", colLabel: "predicted", aria: "confusion matrix" });
      note(f2, "Rows are the true digit and columns the prediction. Correct predictions on the diagonal are left blank.");
    }

    const A = R.ablation_optimizers;
    if (A) {
      const rows = Object.entries(A.results).map(([k, v]) => ({ k, ...v })).sort((a, b) => b.val_acc.mean - a.val_acc.mean);
      const best = rows[0];
      const lo = Math.min(...rows.map((r) => r.val_acc.mean - r.val_acc.ci95)), hi = Math.max(...rows.map((r) => r.val_acc.mean + r.val_acc.ci95));
      const sig = rows.slice(1).filter((r) => r.vs_best.p < 0.05).map((r) => r.k);
      const spread = best.val_acc.mean - rows[rows.length - 1].val_acc.mean;
      const finding = sig.length
        ? `After tuning each optimizer's learning rate, <b>${best.k}</b> reached the best validation accuracy (${pct(best.val_acc.mean)}) and beat ${sig.join(", ")} significantly (Welch's t-test, p < 0.05).`
        : `Once each learning rate is tuned, all five optimizers finish within <b>${(100 * spread).toFixed(2)} points</b> of each other, and none differs significantly from the best (Welch's t-test, every p > 0.05). On this problem the learning rate matters more than the choice of optimizer.`;
      const f = fig({ eyebrow: "Ablation A", title: "Which optimizer, once each is tuned?", finding });
      Charts.barChart(chartIn(f), { dots: true, labelWidth: 118, valueName: "val accuracy", fmt: (v) => pct(v, 2),
        domain: [lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.1], xLabel: "validation accuracy (mean, 95% CI)",
        bars: rows.map((r) => ({ label: r.k, value: r.val_acc.mean, lo: r.val_acc.mean - r.val_acc.ci95, hi: r.val_acc.mean + r.val_acc.ci95,
          note: `best lr ${r.best_lr} · n=${r.val_acc.n}` + (r.vs_best ? ` · p=${r.vs_best.p.toPrecision(2)} vs best` : "") })) });
      table(f, [{ t: "optimizer" }, { t: "tuned lr", num: true }, { t: "val acc", num: true }, { t: "± 95% CI", num: true }, { t: "p vs best", num: true }],
        rows.map((r) => [r.k, r.best_lr, pct(r.val_acc.mean), pct(r.val_acc.ci95), r.vs_best ? r.vs_best.p.toPrecision(2) : "–"]));
      note(f, `MLP 784-256-256-10. Learning rates tuned on a 5-point grid with a separate seed, then ${A.seeds} fresh seeds × ${A.epochs} epochs.`);
    }

    const B = R.ablation_init;
    if (B) {
      const inits = ["N(0, 0.01)", "N(0, 1)", "Xavier", "Kaiming"], colors = ["--s1", "--s2", "--s3", "--s4"];
      for (const act of ["tanh", "relu"]) {
        const series = inits.map((name, i) => ({ name, color: colors[i],
          points: B.results[`${act} / ${name}`].act_std.map((v, l) => [l + 1, v]).filter((p) => p[1] !== null && p[1] < 1e30)
            .map((p) => [p[0], Math.max(p[1], 1e-30)]) }));
        const acc = inits.map((n) => [n, B.results[`${act} / ${n}`].val_acc.mean]);
        const trained = acc.filter(([, a]) => a > 0.5), failed = acc.filter(([, a]) => a <= 0.5);
        const why = (n) => {
          const s = B.results[`${act} / ${n}`].act_std, last = s[s.length - 1];
          return last < 1e-3 ? `${n} vanishes (std ${last.toExponential(0)} at the last layer)`
            : act === "tanh" ? `${n} saturates every unit at ±1`
            : last >= 1e30 ? `${n} overflows to infinity` : `${n} explodes (std ${last.toExponential(0)})`;
        };
        const f = fig({ eyebrow: "Ablation B · " + act, title: `Signal through ${B.depth} ${act === "relu" ? "ReLU" : "tanh"} layers`,
          finding: `Activation spread at initialisation, layer by layer. ${failed.map(([n]) => why(n)).join("; ")}${failed.length ? ", and neither learns anything." : ""} ` +
            `Only the fan-in-scaled inits train: ${trained.map(([n, a]) => `<b>${n}</b> ${pct(a, 1)}`).join(", ")} after ${B.epochs} epochs.` +
            (act === "relu" ? ` Xavier's signal still shrinks ${Math.round(B.results["relu / Kaiming"].act_std.at(-1) / B.results["relu / Xavier"].act_std.at(-1))}× more than Kaiming's, because it ignores that ReLU zeroes half its inputs. Kaiming's √2 gain corrects for that.` : "") });
        Charts.lineChart(chartIn(f), { series, xLabel: "layer", yLabel: "activation std (log)", yLog: true, height: 230, endLabels: false });
        table(f, [{ t: "init" }, { t: "val acc", num: true }, { t: "layer-1 std", num: true }, { t: `layer-${B.depth - 1} std`, num: true }],
          acc.map(([n, a]) => { const s = B.results[`${act} / ${n}`].act_std; const e = (v) => (v === null || v >= 1e30 ? "∞" : v.toExponential(1));
            return [n, pct(a, 1) + (B.results[`${act} / ${n}`].diverged ? " (diverged)" : ""), e(s[0]), e(s[s.length - 1])]; }));
      }
    }

    const C = R.ablation_batchnorm;
    if (C) {
      const series = [["no BatchNorm", "--s2"], ["with BatchNorm", "--s1"]].map(([k, c]) => ({ name: k, color: c, markers: true,
        points: C.results[k].map((r) => [r.lr, r.val_acc.mean]) }));
      const nb = C.results["no BatchNorm"], wb = C.results["with BatchNorm"];
      const bestNb = nb.reduce((a, b) => (b.val_acc.mean > a.val_acc.mean ? b : a));
      const worstWb = wb.reduce((a, b) => (b.val_acc.mean < a.val_acc.mean ? b : a));
      const lrs = wb.map((r) => r.lr), range = Math.round(Math.max(...lrs) / Math.min(...lrs));
      const f = fig({ eyebrow: "Ablation C", title: "BatchNorm makes the learning rate forgiving",
        finding: `The same ${C.depth}-layer ReLU MLP with PyTorch's default initialisation. Without BatchNorm, the best of ${nb.length} learning rates reaches only <b>${pct(bestNb.val_acc.mean, 1)}</b>: small rates barely move a signal that shrinks at every layer, and large ones diverge. With BatchNorm, every rate from ${Math.min(...lrs)} to ${Math.max(...lrs)}, a ${range}× range, reaches at least <b>${pct(worstWb.val_acc.mean, 1)}</b>.` });
      Charts.lineChart(chartIn(f), { series, xLog: true, xLabel: "learning rate (log)", yLabel: "val accuracy", yFmt: (v) => pct(v, 0), height: 240, endLabels: true, xFmt: (v) => String(v) });
      note(f, `${C.seeds} seeds per point, ${C.epochs} epochs each. A diverged run counts as chance accuracy (10%).`);
    }

    const S = R.scaling;
    if (S) {
      const pts = S.rows.map((r) => [r.non_embedding_params, r.val_loss]);
      const xs = pts.map((p) => p[0]), fitPts = [];
      for (let k = 0; k <= 40; k++) { const n = Math.min(...xs) * Math.pow(Math.max(...xs) / Math.min(...xs), k / 40); fitPts.push([n, S.fit.E + S.fit.A * Math.pow(n, -S.fit.alpha)]); }
      const f = fig({ eyebrow: "Scaling study", title: "A small scaling law",
        finding: `Six GPTs trained on identical data and steps. Validation loss follows L(N) ≈ ${S.fit.E >= 0.01 ? S.fit.E.toFixed(2) + " + " : ""}${S.fit.A.toFixed(2)}·N<sup>−${S.fit.alpha.toFixed(3)}</sup> in non-embedding parameters N (R² = ${S.fit.r2.toFixed(3)}). ` +
          `The curve bends at the largest sizes: with a fixed budget of ${(S.tokens / 1e6).toFixed(1)}M tokens, bigger models are increasingly under-trained.` });
      Charts.lineChart(chartIn(f), { series: [{ name: "trained model", color: "--s1", points: pts, markers: true, noLine: true }],
        fit: { color: "--muted", points: fitPts }, xLog: true, xLabel: "non-embedding parameters (log)", yLabel: "val loss (nats/char)", height: 240 });
      note(f, `${S.iters.toLocaleString()} steps × 4,096 tokens per model, no dropout, the same learning rate for every size. The grey curve is the least-squares fit.`);
    }

    const T = R.tokenizers;
    if (T) {
      const rows = T.rows.slice().sort((a, b) => a.val_bpc - b.val_bpc);
      const f = fig({ eyebrow: "Tokenizer study", title: "Characters or BPE, at equal compute",
        finding: `The same architecture and budget with four tokenizers, compared in bits per character. <b>${rows[0].tokenizer}</b> does best: each token spans ${rows[0].chars_per_token.toFixed(2)} characters, so the model sees more text per step and more context per window.` });
      Charts.barChart(chartIn(f), { labelWidth: 118, valueName: "bits/char", fmt: (v) => v.toFixed(2), xLabel: "validation bits per character (lower is better)",
        bars: rows.map((r, i) => ({ label: r.tokenizer, value: r.val_bpc, emphasis: i === 0 ? undefined : false, note: `vocab ${r.vocab_size} · ${r.chars_per_token.toFixed(2)} chars/token` })) });
      table(f, [{ t: "tokenizer" }, { t: "vocab", num: true }, { t: "chars/token", num: true }, { t: "bits/char", num: true }],
        rows.map((r) => [r.tokenizer, r.vocab_size, r.chars_per_token.toFixed(2), r.val_bpc.toFixed(3)]));
    }

    const Q = R.quantization, Bm = R.benchmark;
    if (Q || Bm) {
      const f = fig({ eyebrow: "Engineering", title: "Speed and size",
        finding: (Q ? `int8 weights make the models ${(Q.gpt.bytes_fp32 / Q.gpt.bytes_int8).toFixed(1)}× smaller. The CNN's test accuracy moves by ${((Q.cnn.int8 - Q.cnn.fp32) * 100).toFixed(2)} points and the GPT's validation loss by ${(Q.gpt.int8 - Q.gpt.fp32 >= 0 ? "+" : "") + (Q.gpt.int8 - Q.gpt.fp32).toExponential(1)} nats. ` : "") +
          (Bm ? `A training step in Sutra takes ${Math.min(...Bm.rows.map((r) => r.slowdown)).toFixed(1)}–${Math.max(...Bm.rows.map((r) => r.slowdown)).toFixed(1)}× as long as in PyTorch on the same CPU.` : "") });
      if (Q) table(f, [{ t: "model" }, { t: "metric" }, { t: "float32", num: true }, { t: "int8", num: true }, { t: "size", num: true }],
        [["CNN", "test accuracy", pct(Q.cnn.fp32), pct(Q.cnn.int8), `${(Q.cnn.bytes_fp32 / 1e6).toFixed(2)} → ${(Q.cnn.bytes_int8 / 1e6).toFixed(2)} MB`],
         ["GPT", "val loss (nats)", Q.gpt.fp32.toFixed(5), Q.gpt.int8.toFixed(5), `${(Q.gpt.bytes_fp32 / 1e6).toFixed(1)} → ${(Q.gpt.bytes_int8 / 1e6).toFixed(1)} MB`]]);
      if (Bm) {
        table(f, [{ t: "training step" }, { t: "Sutra", num: true }, { t: "PyTorch", num: true }, { t: "ratio", num: true }],
          Bm.rows.map((r) => [r.model, r.sutra_ms.toFixed(1) + " ms", r.torch_ms.toFixed(1) + " ms", r.slowdown.toFixed(1) + "×"]));
        note(f, `Median of 20 steps (forward, backward, AdamW) in float32 on ${Bm.machine}, NumPy ${Bm.numpy}, PyTorch ${Bm.torch}.`);
      }
    }
  }

  main();
})();
