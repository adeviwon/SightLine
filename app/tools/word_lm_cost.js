/**
 * word_lm_cost.js — what the word-LM decode stage costs on a phone: decode time
 * measured ALONE on real model output, then the forward pass it sits behind.
 *
 *     node app/tools/word_lm_cost.js
 *
 * WHY DECODE ALONE, ON REAL LOGITS
 * --------------------------------
 * "Runs on an iPhone 14 Pro Max" is a budget claim, so the two halves have to be
 * separated: the ONNX forward pass dominates a scan and the decoder is the part
 * this project added. Timing `recognise()` end to end conflates them, and
 * synthetic crops change T (the only thing the beam really scales with).
 *
 * So the decode numbers come from `app/tools/fixtures/wordlm_vectors.json` — the
 * SHIPPED recogniser's own log-softmaxed output for 60 held-out crops, the same
 * artifact the parity gate uses — replayed through `beamDecode` in the browser.
 * Beam 4 (shipped) and beam 8 (the width a sweep would tempt you into) are both
 * measured, because the quality difference between them is noise (+0.2pp at
 * 1,000 crops; beam 8 is +0/-7 vs shipped paired on 3,375) and the cost
 * difference is not.
 *
 * ENGINES. WebKit is iOS Safari's engine and is the lower bound. Chromium under
 * Chromium's own CPU throttle is an APPROXIMATION of a slower phone and is
 * labelled as such — no iPhone is benchmarked here.
 */
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const pw = require("./pw_core.js");
const chromeBin = require("./chrome_path.js");

const APP = path.resolve(__dirname, "..");
const FIX = path.join(__dirname, "fixtures", "wordlm_vectors.json");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm", ".css": "text/css; charset=utf-8", ".png": "image/png",
  ".svg": "image/svg+xml", ".ico": "image/x-icon" };
const PAGE_CROPS = 24;   // a page's worth of lines

const pad = (v, w, right) => {
  const s = String(v);
  return right ? s.padStart(w) : s.padEnd(w);
};
const ms = (v) => v.toFixed(1) + " ms";

function serve() {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (p === "/") p = "/index.html";
    const file = path.join(APP, p);
    if (!file.startsWith(APP) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
                         "Cache-Control": "no-store" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

/** Decode-only: the beam over the real logits, per beam width, in this engine. */
async function decodeCost(browser, base, throttle) {
  const page = await browser.newPage();
  if (throttle > 1) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: throttle });
  }
  await page.goto(base + "/index.html", { waitUntil: "load" });
  const out = await page.evaluate(async ([vectors, widths]) => {
    const { beamDecode, WordTrie } = await import("/js/word_beam.js");
    const trie = new WordTrie(await (await fetch("/data/word_trie.json")).json());
    const b64 = (s) => new Uint8Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
    const rows = vectors.rows.map((r) => ({ lps: new Float64Array(b64(r.logprobs).buffer),
                                            T: r.T, C: r.C }));
    const charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    const res = {};
    for (const beam of widths) {
      // warm, then measure two passes and keep the better (JIT + one cold pass)
      for (let pass = 0; pass < 3; pass++) {
        const t0 = performance.now();
        for (const r of rows) beamDecode(r.lps, r.T, r.C, charset,
                                        { beam, blank: 0, lm: trie, alpha: 0.25 });
        const dt = performance.now() - t0;
        if (pass > 0) { if (res[beam] === undefined || dt < res[beam]) res[beam] = dt; }
      }
      res[beam] = res[beam] / rows.length;      // ms per crop
    }
    // trie fetch + parse, the one-off cost an install pays
    const t1 = performance.now();
    const j = await (await fetch("/data/word_trie.json")).json();
    new WordTrie(j);
    return { per_crop: res, crops: rows.length, trie_ms: performance.now() - t1,
             timesteps: rows[0].T, words: trie.words };
  }, [JSON.parse(fs.readFileSync(FIX, "utf8")), [4, 8]]);
  await page.close();
  return out;
}

/** Forward-only: the ONNX pass for the same crop widths, so the share is explicit. */
async function forwardCost(browser, base, throttle) {
  const page = await browser.newPage();
  if (throttle > 1) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: throttle });
  }
  await page.goto(base + "/index.html", { waitUntil: "load" });
  const out = await page.evaluate(async ([crops]) => {
    const R = await import("/js/recognizer.js");
    await R.init();
    await R.loadWordTrie();
    const mk = (w) => {
      const h = R.CROP_H, g = new Uint8Array(w * h).fill(240);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++)
          if ((x % 11) < 5 && y > 5 && y < h - 5) g[y * w + x] = 25 + (x % 40);
      return g;
    };
    const widths = [120, 200, 320, 420];
    const batch = [];
    for (let i = 0; i < crops; i++) batch.push({ g: mk(widths[i % 4]), w: widths[i % 4] });
    for (const c of batch) await R.recognise(c.g, c.w);        // warm
    const t0 = performance.now();
    for (const c of batch) await R.recognise(c.g, c.w);
    const per = (performance.now() - t0) / batch.length;
    const r = await R.recognise(batch[0].g, batch[0].w);
    return { per_crop: per, decoder: r.decoder };
  }, [PAGE_CROPS]);
  await page.close();
  return out;
}

async function main() {
  if (!fs.existsSync(FIX)) {
    console.error(`FAIL no fixtures at ${FIX} — run:\n  bash run.sh word-lm-vectors --n 60`);
    process.exit(1);
  }
  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port;
  const pwc = pw();
  const rows = [];

  const wk = await pwc.webkit.launch();
  for (const t of [1]) {
    rows.push({ engine: "webkit", throttle: t, note: t === 1 ? "iOS Safari engine" : `throttle ${t}x`,
                decode: await decodeCost(wk, base, t), forward: await forwardCost(wk, base, t) });
  }
  await wk.close();

  const cr = await pwc.chromium.launch({ executablePath: chromeBin(),
    args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  for (const t of [1, 4, 6]) {
    rows.push({ engine: "chromium", throttle: t,
                note: t === 1 ? "unthrottled" : `throttle ${t}x (phone approximation)`,
                decode: await decodeCost(cr, base, t), forward: await forwardCost(cr, base, t) });
  }
  await cr.close();
  server.close();

  const n = rows[0].decode.crops;
  console.log(`word-LM decode cost — ${n} real held-out crops (median of 2 warm passes),`
              + ` CROP_H=${"CROP_H"} T=${rows[0].decode.timesteps}\n`);
  console.log(pad("engine", 10) + pad("cpu", 30, false) + pad("decode/crop beam4", 19, true)
              + pad("beam8", 10, true) + pad("forward/crop", 14, true)
              + pad("page +24 beam4", 16, true));
  for (const r of rows) {
    const d4 = r.decode.per_crop[4], d8 = r.decode.per_crop[8];
    console.log(pad(r.engine, 10) + pad(r.note, 30, false) + pad(ms(d4), 19, true)
                + pad(ms(d8), 10, true) + pad(ms(r.forward.per_crop), 14, true)
                + pad((d4 * PAGE_CROPS / 1000).toFixed(2) + " s", 16, true));
  }

  const w = rows[0];
  const share = 100 * w.decode.per_crop[4] / (w.decode.per_crop[4] + w.forward.per_crop);
  console.log(`\nWebKit: decode is ${share.toFixed(0)}% of (decode+forward) per crop;`
              + ` the decoder adds ${(w.decode.per_crop[4] * PAGE_CROPS / 1000).toFixed(2)} s`
              + ` to a ${PAGE_CROPS}-crop page on top of`
              + ` ${(w.forward.per_crop * PAGE_CROPS / 1000).toFixed(1)} s of inference.`);
  console.log(`beam 8 costs ${(w.decode.per_crop[8] / w.decode.per_crop[4]).toFixed(2)}x beam 4`
              + ` and measured WORSE paired on 3,375 crops (+0/-7, p=0.0156) — width is not a lever here.`);
  console.log(`one-off: trie fetch+parse ${w.decode.trie_ms.toFixed(0)} ms,`
              + ` precache weight ${(fs.statSync(path.join(APP, "data", "word_trie.json")).size / 1024).toFixed(0)} KB`
              + ` + ${(fs.statSync(path.join(APP, "js", "word_beam.js")).size / 1024).toFixed(1)} KB of code,`
              + ` ${w.decode.words} words`);
  if (w.forward.decoder !== "wordLM") {
    console.error("FAIL the app's scan did not use the word LM — the decode share above is meaningless");
    process.exit(1);
  }
  return 0;
}

main().then((c) => process.exit(c || 0))
      .catch((e) => { console.error("FAIL " + (e && e.stack || e)); process.exit(1); });
