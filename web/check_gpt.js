// Checks web/sutra.js's GPT against Python on any exported bundle:
//   node web/check_gpt.js <dir> <name>
// <dir>/<name>.json|.bin is the bundle; <dir>/<name>.golden.json holds {prompt_ids, step_logits, greedy_ids}.
"use strict";
const fs = require("fs");
const path = require("path");
const [dir, name] = process.argv.slice(2);
global.fetch = async (url) => {
  const buf = fs.readFileSync(url);
  return { ok: true, json: async () => JSON.parse(buf.toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
require("./sutra.js");
(async () => {
  const gpt = new globalThis.Sutra.GPT(await globalThis.Sutra.loadBundle(path.join(dir, name)));
  const g = JSON.parse(fs.readFileSync(path.join(dir, name + ".golden.json"), "utf8"));
  let maxd = 0, out;
  g.prompt_ids.forEach((id, t) => {
    out = gpt.step(id);
    g.step_logits[t].forEach((v, j) => { maxd = Math.max(maxd, Math.abs(v - out.logits[j])); });
  });
  const ids = [];
  for (let i = 0; i < g.greedy_ids.length; i++) { const nx = globalThis.Sutra.GPT.sample(out.logits, 0); ids.push(nx); out = gpt.step(nx); }
  const same = ids.every((v, i) => v === g.greedy_ids[i]);
  console.log(JSON.stringify({ maxd, greedyMatches: same }));
  process.exit(maxd < 1e-3 && same ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
