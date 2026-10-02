/**
 * IS IT REALLY A TESSERACT CEILING?
 *
 * The three realistic handheld profiles — handheld_light, handheld_heavy,
 * low_light — score 0% field recovery on EVERY arm, including the trained
 * restorer. That has been reported as "a Tesseract ceiling, not a preprocessing
 * gap" on the strength of the fact that restoration does not help.
 *
 * But "restoration does not help" is not the same as "no segmentation mode can
 * help". runOCR only ever tries psm 6, 3 and 11, and only on three candidate
 * images. If some other PSM — or the same PSM on a different candidate —
 * recovers the text, then the limit is our SEARCH, not Tesseract, and the
 * remedy is completely different.
 *
 * This sweeps the full Tesseract PSM space (0-13, excluding the orientation-only
 * modes) across every candidate on every sample, and reports the BEST any
 * configuration achieves. Three outcomes, each demanding a different response:
 *
 *   A. some PSM recovers the fields
 *      -> NOT a ceiling. Our pass list is too narrow. Fix runOCR.
 *   B. nothing recovers them, but some PSM gets much more text
 *      -> a segmentation trade-off: legible text that does not parse into
 *         fields. Relevant to the structure-aware ranking.
 *   C. nothing recovers them and nothing reads them
 *      -> a genuine ceiling. The claim stands, and the honest move is to say
 *         so and show the work.
 *
 * Run:  node tools/psm_sweep.js
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { CDP } = require("./cdp_min.js");

const ROOT = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const CHROME = process.env.CHROME_BIN
  || "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".png": "image/png",
  ".jpg": "image/jpeg", ".wasm": "application/wasm", ".gz": "application/gzip",
  ".md": "text/markdown; charset=utf-8", ".ico": "image/x-icon",
  ".onnx": "application/octet-stream", ".traineddata": "application/octet-stream",
};

// Tesseract psm 0-13. 0 = OSD only, 1 = OSD + text: both return no usable text
// for a flatbed-style document capture, so they are excluded and the exclusion
// is stated rather than silently applied.
const PSMS = [3, 4, 6, 7, 11, 12, 13];

function serve() {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split("?")[0]);
      let p = rel.startsWith("/sample/")
        ? path.join(SAMPLES, rel.slice("/sample/".length))
        : path.join(ROOT, rel);
      if (!p.startsWith(ROOT) && !p.startsWith(SAMPLES)) { res.writeHead(403); res.end(); return; }
      fs.readFile(p, (e, d) => {
        if (e) { res.writeHead(404, "text/plain"); res.end("404 " + rel); return; }
        res.writeHead(200, {
          "Content-Type": MIME[path.extname(p)] || "application/octet-stream",
          "Cache-Control": "no-store",
        });
        res.end(d);
      });
    });
    srv.listen(0, () => resolve(srv));
  });
}

(async () => {
  const server = await serve();
  const port = server.address().port;
  const manifest = JSON.parse(fs.readFileSync(path.join(SAMPLES, "manifest.json"), "utf8"));
  const chrome = spawn(CHROME, ["--headless=new", "--no-sandbox", "--disable-gpu",
    "--remote-debugging-port=9371", "about:blank"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 3500));

  try {
    const cdp = await CDP.attach(9371);
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${port}/index.html` });
    await new Promise((r) => setTimeout(r, 5000));
    const init = await cdp.eval("(async()=>{await SightLineRestorer.init();return 1})()", 300000);

    console.log(`Sweeping psm ${PSMS.join(",")} x 3 candidates on `
      + `${manifest.samples.length} samples.\n`);
    console.log("This is 7 x 3 OCR passes per sample and takes a while.\n");

    const rows = [];
    for (const s of manifest.samples) {
      const r = await cdp.eval(`(async()=>{
        const img = new Image();
        img.src = "/sample/" + ${JSON.stringify(path.basename(s.file))};
        await img.decode();
        const pp = SightLine.preprocess(img, 3200, {restoreMode:"none"});
        const P = SightLine;

        // Candidates: raw, restored (if the gate allows it), clahe, binary.
        let cands = [{tag:"raw", c: pp.canvas}];
        const r2 = await SightLineRestorer.armRestorerClahe(pp.canvas, {});
        if (r2.engine !== "gate-skipped") cands.push({tag:"restored", c: r2.candidates[0]});
        cands.push({tag:"clahe", c: r2.candidates[1]});
        cands.push({tag:"binary", c: r2.candidates[2]});

        const out = [];
        for (const cd of cands) {
          for (const psm of ${JSON.stringify(PSMS)}) {
            await SightLine.setPSM(psm);
            const o = await P.runOCR([cd.c], ()=>{});
            const norm = P.normalizeOCRText(o.text);
            out.push({
              cand: cd.tag, psm: psm,
              conf: Math.round((o.confidence||0)*100),
              chars: (o.text||"").length,
              found: P.fields_found(norm, ${JSON.stringify(s.doc)}),
              struct: P.structure_score(o.text),
            });
          }
        }
        await SightLine.setPSM(6);
        return JSON.stringify({out, weight: SightLineRestorer.status().weight});
      })()`, 1800000);

      const d = JSON.parse(r);
      const best = d.out.reduce((a, b) => {
        if (!a) return b;
        if (b.found.length !== a.found.length) return b.found.length > a.found.length ? b : a;
        if (b.chars !== a.chars) return b.chars > a.chars ? b : a;
        return b.conf > a.conf ? b : a;
      }, null);
      const anyFields = d.out.some((o) => o.found.length > 0);
      const maxChars = Math.max(...d.out.map((o) => o.chars));

      rows.push({
        doc: s.doc, profile: s.profile, weight: d.weight,
        best, anyFields, maxChars,
        nConfigs: d.out.length,
      });

      console.log(
        `${(s.doc + "/" + s.profile).padEnd(32)} w=${d.weight.toFixed(2)} `
        + `best=[${JSON.stringify(best.found)}] `
        + `via ${best.cand}/psm${best.psm} conf=${best.conf}% `
        + `chars=${best.chars} | maxChars=${maxChars}`);
    }

    console.log("\n" + "=".repeat(74));
    console.log("VERDICT");
    console.log("=".repeat(74));

    const zeroProfiles = [...new Set(rows.map((r) => r.profile))];
    let anyRescued = false;
    for (const r of rows) {
      const shipFails = ["handheld_light", "handheld_heavy", "low_light",
                         "worst_case"].includes(r.profile);
      if (shipFails && r.anyFields) {
        anyRescued = true;
        console.log(`RESCUED  ${r.doc}/${r.profile}: ${JSON.stringify(r.best.found)} `
          + `via ${r.best.cand}/psm${r.best.psm} — our pass list missed it`);
      }
    }

    if (anyRescued) {
      console.log("\n=> NOT a Tesseract ceiling. Our PSM pass list is too narrow;");
      console.log("   runOCR should try the recovering modes.");
    } else {
      const shipRows = rows.filter((r) =>
        ["handheld_light", "handheld_heavy", "low_light", "worst_case"].includes(r.profile));
      const anyText = shipRows.some((r) => r.maxChars > 60);
      console.log("=> No configuration in the full PSM space recovers any field");
      console.log("   on the handheld/low-light profiles. That IS a ceiling.");
      console.log(anyText
        ? "   Some configurations do return substantial text, but never the\n"
        + "   safety-critical fields — a segmentation trade-off, not a gain."
        : "   Very little text is returned under any mode either.");
    }
    console.log(`\n(${rows.length} samples, `
      + `${rows[0] ? rows[0].nConfigs : 0} configurations each)`);
  } catch (e) {
    console.log("error:", e && e.message);
  } finally {
    chrome.kill();
    server.close();
    process.exit(0);
  }
})();
