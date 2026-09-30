/*
 * charts.js — small SVG charts for the Sutra report (line, horizontal bar with CI, heatmap).
 * Colors come from CSS custom properties so both themes work; every chart has a hover layer.
 */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function fmt(v, digits) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (digits !== undefined) return v.toFixed(digits);
    if (a >= 1e7 || a < 1e-3) return "1e" + Math.round(Math.log10(a));
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(a >= 10 ? 1 : 2).replace(/\.?0+$/, "");
    if (a >= 0.01) return v.toFixed(3).replace(/0+$/, "");
    return v.toExponential(0);
  }

  function niceTicks(min, max, count) {
    const span = max - min || Math.abs(max) || 1;
    const step0 = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
  }

  function logTicks(min, max) {
    const out = [];
    const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
    const mult = hi - lo <= 2 ? [1, 2, 5] : [1];
    const stride = Math.max(1, Math.ceil((hi - lo) / 6));  // at most ~6 decade ticks
    for (let e = lo; e <= hi; e += stride) for (const m of mult) {
      const v = m * Math.pow(10, e);
      if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
    }
    return out;
  }

  function makeScale(d0, d1, r0, r1, log) {
    if (log) {
      const a = Math.log10(d0), b = Math.log10(d1);
      return (v) => r0 + ((Math.log10(v) - a) / (b - a || 1)) * (r1 - r0);
    }
    return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  }

  function tooltip(container) {
    let tip = container.querySelector(".chart-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      tip.hidden = true;
      container.appendChild(tip);
    }
    return {
      show(html, x, y) {
        tip.innerHTML = html;
        tip.hidden = false;
        const cw = container.clientWidth, tw = tip.offsetWidth;
        tip.style.left = Math.min(Math.max(0, x + 12), cw - tw) + "px";
        tip.style.top = Math.max(0, y - 10) + "px";
      },
      hide() { tip.hidden = true; },
    };
  }

  function legend(container, series) {
    if (series.length < 2) return;
    const lg = document.createElement("div");
    lg.className = "chart-legend";
    for (const s of series) {
      const item = document.createElement("span");
      item.innerHTML = `<i style="background:var(${s.color})"></i>${s.name}`;
      lg.appendChild(item);
    }
    container.appendChild(lg);
  }

  function responsive(container, draw) {
    let lastW = 0;
    const run = () => {
      const w = container.clientWidth;
      if (!w || w === lastW) return;
      lastW = w;
      container.innerHTML = "";
      draw(w);
    };
    run();
    if (global.ResizeObserver) new ResizeObserver(run).observe(container);
  }

  /**
   * opts: { series: [{name, color: "--s1", points: [[x, y], ...], markers?}], xLabel, yLabel, xLog, yLog,
   *         height, yDomain, xFmt, yFmt, endLabels, fit: {color, points} }
   */
  function lineChart(container, opts) {
    responsive(container, (W) => {
      legend(container, opts.series);
      const H = opts.height || 240;
      const m = { l: 50, r: opts.endLabels ? 96 : 14, t: 10, b: 42 };
      const all = opts.series.flatMap((s) => s.points).concat(opts.fit ? opts.fit.points : []);
      let [x0, x1] = [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
      let [y0, y1] = opts.yDomain || [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
      if (!opts.yDomain) {
        if (opts.yLog) { y0 /= 1.4; y1 *= 1.4; } else { const pad = (y1 - y0) * 0.08 || 0.1; y0 -= pad; y1 += pad; }
      }
      if (opts.xLog) { x0 /= 1.15; x1 *= 1.15; }
      const sx = makeScale(x0, x1, m.l, W - m.r, opts.xLog), sy = makeScale(y0, y1, H - m.b, m.t, opts.yLog);
      const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img",
        "aria-label": opts.aria || opts.yLabel || "chart" }, container);
      const yt = opts.yLog ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
      const xt = opts.xLog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 90)));
      const yf = opts.yFmt || fmt, xf = opts.xFmt || fmt;
      for (const v of yt) {
        const y = sy(v);
        svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
        svgEl("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", class: "tick" }, svg).textContent = yf(v);
      }
      svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
      for (const v of xt) {
        svgEl("text", { x: sx(v), y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, svg).textContent = xf(v);
      }
      if (opts.xLabel) svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent = opts.xLabel;
      if (opts.yLabel) svgEl("text", { x: 12, y: m.t + (H - m.b - m.t) / 2, transform: `rotate(-90 12 ${m.t + (H - m.b - m.t) / 2})`, "text-anchor": "middle", class: "axis-label" }, svg).textContent = opts.yLabel;

      const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
      if (opts.fit) svgEl("path", { d: path(opts.fit.points), class: "fit-line", style: `stroke:var(${opts.fit.color || "--muted"})` }, svg);
      for (const s of opts.series) {
        if (s.points.length > 1 && !s.noLine) svgEl("path", { d: path(s.points), class: "series-line", style: `stroke:var(${s.color})` }, svg);
        if (s.markers || s.points.length === 1) {
          for (const p of s.points) svgEl("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4.5, class: "marker", style: `fill:var(${s.color})` }, svg);
        }
      }
      if (opts.endLabels) {  // direct labels at line ends, dropped if they would collide
        const ends = opts.series.map((s) => ({ s, y: sy(s.points[s.points.length - 1][1]), x: sx(s.points[s.points.length - 1][0]) }))
          .sort((a, b) => a.y - b.y);
        const clear = ends.every((e, i) => i === 0 || e.y - ends[i - 1].y >= 13);
        if (clear) for (const e of ends) svgEl("text", { x: e.x + 8, y: e.y + 4, class: "end-label" }, svg).textContent = e.s.name;
      }

      // hover: crosshair + nearest point of every series
      const tip = tooltip(container);
      const cross = svgEl("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
      const dots = opts.series.map((s) => svgEl("circle", { r: 4.5, class: "marker", style: `fill:var(${s.color})`, visibility: "hidden" }, svg));
      const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.b - m.t, fill: "transparent" }, svg);
      const move = (ev) => {
        const r = svg.getBoundingClientRect(), px = ev.clientX - r.left;
        let rows = [], cx = null;
        opts.series.forEach((s, i) => {
          let best = null, bd = Infinity;
          for (const p of s.points) { const d = Math.abs(sx(p[0]) - px); if (d < bd) { bd = d; best = p; } }
          if (!best || bd > 60) { dots[i].setAttribute("visibility", "hidden"); return; }
          dots[i].setAttribute("cx", sx(best[0]));
          dots[i].setAttribute("cy", sy(best[1]));
          dots[i].setAttribute("visibility", "visible");
          cx = cx === null ? sx(best[0]) : cx;
          rows.push(`<div><i style="background:var(${s.color})"></i>${s.name}: <b>${yf(best[1])}</b> <span>@ ${xf(best[0])}</span></div>`);
        });
        if (!rows.length) return leave();
        cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
        tip.show(rows.join(""), px, ev.clientY - r.top);
      };
      const leave = () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); tip.hide(); };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", leave);
    });
  }

  /** Horizontal bars (or dots, for a truncated axis). opts: { bars: [{label, value, lo?, hi?, emphasis?, note?}], dots, xLabel, fmt, domain, color } */
  function barChart(container, opts) {
    responsive(container, (W) => {
      const rowH = 34, m = { l: opts.labelWidth || 120, r: 64, t: 6, b: opts.xLabel ? 40 : 24 };
      const H = m.t + m.b + rowH * opts.bars.length;
      const vals = opts.bars.flatMap((b) => [b.value, b.hi ?? b.value, b.lo ?? b.value]);
      const [d0, d1] = opts.domain || [Math.min(0, ...vals), Math.max(...vals) * 1.05];
      const sx = makeScale(d0, d1, m.l, W - m.r, false), f = opts.fmt || fmt;
      const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": opts.aria || opts.xLabel || "bar chart" }, container);
      for (const v of niceTicks(d0, d1, Math.max(2, Math.floor((W - m.l - m.r) / 90)))) {
        svgEl("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
        svgEl("text", { x: sx(v), y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, svg).textContent = f(v);
      }
      if (opts.xLabel) svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent = opts.xLabel;
      const tip = tooltip(container);
      opts.bars.forEach((b, i) => {
        const y = m.t + i * rowH + rowH / 2, x0 = sx(Math.max(d0, 0)), x1 = sx(b.value);
        const color = b.emphasis === false ? "--bar-muted" : (opts.color || "--s1");
        svgEl("text", { x: m.l - 10, y: y + 4, "text-anchor": "end", class: "bar-label" }, svg).textContent = b.label;
        if (!opts.dots) {
          const w = Math.max(1, x1 - x0), h = 18, r = Math.min(4, w / 2);
          // rounded data-end, square at the baseline
          svgEl("path", { d: `M${x0} ${y - h / 2} H${x0 + w - r} Q${x0 + w} ${y - h / 2} ${x0 + w} ${y - h / 2 + r} V${y + h / 2 - r} Q${x0 + w} ${y + h / 2} ${x0 + w - r} ${y + h / 2} H${x0} Z`,
            style: `fill:var(${color})` }, svg);
        }
        let tipX = x1;
        if (b.lo !== undefined) {
          const a = sx(b.lo), c = sx(b.hi);
          svgEl("line", { x1: a, x2: c, y1: y, y2: y, class: "whisker" }, svg);
          svgEl("line", { x1: a, x2: a, y1: y - 6, y2: y + 6, class: "whisker" }, svg);
          svgEl("line", { x1: c, x2: c, y1: y - 6, y2: y + 6, class: "whisker" }, svg);
          tipX = Math.max(x1, c);
        }
        if (opts.dots) svgEl("circle", { cx: x1, cy: y, r: 5.5, class: "marker", style: `fill:var(${color})` }, svg);
        if (b.showValue !== false) svgEl("text", { x: tipX + 8, y: y + 4, class: "bar-value" }, svg).textContent = f(b.value);
        const hit = svgEl("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, svg);
        hit.addEventListener("pointermove", (ev) => {
          const r = svg.getBoundingClientRect();
          tip.show(`<div><b>${b.label}</b></div><div>${opts.valueName || "value"}: <b>${f(b.value)}</b></div>` +
            (b.lo !== undefined ? `<div>95% CI: ${f(b.lo)} – ${f(b.hi)}</div>` : "") + (b.note ? `<div>${b.note}</div>` : ""),
            ev.clientX - r.left, ev.clientY - r.top);
        });
        hit.addEventListener("pointerleave", () => tip.hide());
      });
    });
  }

  /** Confusion-style heatmap. opts: { matrix, labels, skipDiagonal, rowLabel, colLabel } */
  function heatmap(container, opts) {
    responsive(container, (W) => {
      const n = opts.labels.length, m = { l: 44, t: 26, r: 8, b: 8 };
      const cell = Math.min(34, Math.floor((W - m.l - m.r) / n));
      const H = m.t + m.b + cell * n;
      const svg = svgEl("svg", { width: m.l + m.r + cell * n, height: H, class: "chart-svg", role: "img", "aria-label": opts.aria || "heatmap" }, container);
      let max = 0;
      opts.matrix.forEach((row, i) => row.forEach((v, j) => { if (!(opts.skipDiagonal && i === j)) max = Math.max(max, v); }));
      const tip = tooltip(container);
      const ramp = ["--q1", "--q2", "--q3", "--q4", "--q5", "--q6"];
      for (let j = 0; j < n; j++) svgEl("text", { x: m.l + j * cell + cell / 2, y: m.t - 8, "text-anchor": "middle", class: "tick" }, svg).textContent = opts.labels[j];
      for (let i = 0; i < n; i++) {
        svgEl("text", { x: m.l - 10, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }, svg).textContent = opts.labels[i];
        for (let j = 0; j < n; j++) {
          const v = opts.matrix[i][j], diag = opts.skipDiagonal && i === j;
          const k = v === 0 ? -1 : Math.min(ramp.length - 1, Math.floor((v / (max || 1)) * (ramp.length - 1e-9)));
          const fill = diag ? "var(--diag)" : k < 0 ? "var(--cell-empty)" : `var(${ramp[k]})`;
          const r = svgEl("rect", { x: m.l + j * cell + 1, y: m.t + i * cell + 1, width: cell - 2, height: cell - 2, rx: 3, style: `fill:${fill}` }, svg);
          if (!diag && v > 0) svgEl("text", { x: m.l + j * cell + cell / 2, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "middle", class: k >= 3 ? "cell-text on-dark" : "cell-text" }, svg).textContent = v;
          r.addEventListener("pointermove", (ev) => {
            const b = svg.getBoundingClientRect();
            tip.show(`<div>${opts.rowLabel || "row"} <b>${opts.labels[i]}</b>, ${opts.colLabel || "col"} <b>${opts.labels[j]}</b></div><div><b>${v}</b> images</div>`, ev.clientX - b.left, ev.clientY - b.top);
          });
          r.addEventListener("pointerleave", () => tip.hide());
        }
      }
    });
  }

  global.Charts = { lineChart, barChart, heatmap, fmt };
})(window);
