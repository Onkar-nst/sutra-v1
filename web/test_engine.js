// Node check that web/sutra.js reproduces the Python reference outputs written by web/export.py.
//   node web/test_engine.js
"use strict";
const fs = require("fs");
const path = require("path");

const ASSETS = path.join(__dirname, "assets");
global.fetch = async (url) => {
  const file = path.join(__dirname, url);
  const buf = fs.readFileSync(file);
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
};
require("./sutra.js");
const { loadBundle, DigitCNN, GPT } = globalThis.Sutra;

(async () => {
  const demo = JSON.parse(fs.readFileSync(path.join(ASSETS, "demo.json"), "utf8"));
  const cnn = new DigitCNN(await loadBundle("assets/cnn"));
  let maxCnn = 0, correct = 0;
  demo.cnn.golden_logits.forEach((g, i) => {
    const out = cnn.forward(demo.cnn.samples[i].pixels).logits;
    g.forEach((v, j) => { maxCnn = Math.max(maxCnn, Math.abs(v - out[j])); });
  });
  for (const s of demo.cnn.samples) {
    const p = cnn.forward(s.pixels).probs;
    if (p.indexOf(Math.max(...p)) === s.label) correct++;
  }

  const gpt = new GPT(await loadBundle("assets/gpt"));
  let out;
  for (const id of gpt.tok.encode(demo.gpt.prompt)) out = gpt.step(id);
  let maxGpt = 0;
  demo.gpt.last_logits.forEach((v, j) => { maxGpt = Math.max(maxGpt, Math.abs(v - out.logits[j])); });
  const ids = [];
  for (let i = 0; i < demo.gpt.greedy.length; i++) { const nx = GPT.sample(out.logits, 0); ids.push(nx); out = gpt.step(nx); }
  const greedy = gpt.tok.decode(ids);

  // sliding window: generating past block_size must keep working
  for (let i = 0; i < gpt.cfg.block_size + 20; i++) out = gpt.step(GPT.sample(out.logits, 0.8, 20));
  const finite = Array.from(out.logits).every(Number.isFinite);

  const result = { maxCnn, sampleAccuracy: correct / demo.cnn.samples.length, maxGpt, greedyMatches: greedy === demo.gpt.greedy, finite };
  console.log(JSON.stringify(result));
  const ok = maxCnn < 1e-3 && maxGpt < 1e-3 && result.greedyMatches && finite && result.sampleAccuracy >= 0.9;
  process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
