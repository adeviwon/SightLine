/**
 * WHY does restoration lose fields on legal/off_axis?
 *
 * The A/B (tools/restore_ab.js) found:
 *
 *   legal / off_axis   weight 0.43   chars 512 -> 514   fields ["2500","2024-CV-…"] -> []
 *
 * Character count is essentially unchanged, yet BOTH fields vanished.
 * fields_found() is exact substring matching after norm() (strip everything
 * that is not [a-z0-9], lowercase), so the digits cannot have disappeared —
 * they must have been REFORMATTED into something the substring test rejects.
 *
 * The candidate explanations, and they need opposite fixes:
 *
 *   1. Restoration fixed the glyphs and OCR now reads them correctly, but the
 *      correct reading differs from the ground truth (e.g. "2024-CV-O0456"
 *      with a letter O). -> a ground-truth/normaliser problem
 *   2. Restoration made glyphs cleaner but introduced spacing, so the text
 *      reads "2 0 2 4 - C V - 0 0 4 5 6". norm() strips spaces, so this should
 *      NOT matter — which makes it a useful discriminator.
 *   3. runOCR picks the best candidate by confidence. Restoration changed which
 *      of the three candidates (raw / clahe / binary) wins, and the winner is
 *      now a worse one. -> a candidate-selection problem, not a restoration one
 *
 * (3) is the most likely and the most interesting: the restorer does not feed
 * OCR directly, it changes which candidate wins a confidence comparison. This
 * harness dumps the winning candidate's text for both arms so the difference is
 * visible rather than inferred from a count.
 *
 * Run:  node tools/field_regression.js
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
    "--remote-debugging-port=9361", "about:blank"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 3500));

  try {
    const cdp = await CDP.attach(9361);
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${port}/index.html` });
    await new Promise((r) => setTimeout(r, 5000));
    await cdp.eval("(async()=>{await SightLineRestorer.init();return 1})()", 300000);

    for (const s of manifest.samples) {
      const r = await cdp.eval(`(async()=>{
        const img = new Image();
        img.src = "/sample/" + ${JSON.stringify(path.basename(s.file))};
        await img.decode();
        const pp = SightLine.preprocess(img, 3200, {restoreMode:"none"});
        const P = SightLine;

        // Run OCR over each candidate SEPARATELY so we can see which one wins,
        // rather than only seeing runOCR's chosen result.
        const arm = async (useModel) => {
          let cands, tags;
          if (useModel) {
            const r2 = await SightLineRestorer.armRestorerClahe(pp.canvas, {});
            cands = r2.candidates; tags = ["restored","clahe","binary"];
          } else {
            cands = [pp.canvas, P.clahe(pp.canvas,2.0,8),
                     P.binarize(P.unsharp(P.clahe(pp.canvas,2.0,8),1.0,1))];
            tags = ["raw","clahe","binary"];
          }
          const per = [];
          for (let i=0;i<cands.length;i++){
            const o = await SightLine.runOCR([cands[i]], ()=>{});
            const norm = P.normalizeOCRText(o.text);
            per.push({tag:tags[i], conf:o.confidence, psm:o.psm,
                      text:o.text||"", norm,
                      found: P.fields_found(norm, ${JSON.stringify(s.doc)})});
          }
          const best = await P.runOCR(cands, ()=>{});
          const bnorm = P.normalizeOCRText(best.text);
          return {per, bestConf:best.confidence, bestPsm:best.psm,
                  bestFound: P.fields_found(bnorm, ${JSON.stringify(s.doc)}),
                  bestNorm: bnorm, weight: SightLineRestorer.status().weight};
        };
        const off = await arm(false), on = await arm(true);
        return JSON.stringify({off, on, doc: ${JSON.stringify(s.doc)},
                               profile: ${JSON.stringify(s.profile)}});
      })()`, 900000);

      const d = JSON.parse(r);
      const lost = d.off.bestFound.length - d.on.bestFound.length;
      if (lost <= 0) continue;   // only the regressions are interesting

      console.log("=".repeat(78));
      console.log(`${d.doc} / ${d.profile}   gate weight ${d.on.weight.toFixed(2)}   `
        + `fields ${JSON.stringify(d.off.bestFound)} -> ${JSON.stringify(d.on.bestFound)}`);
      console.log("=".repeat(78));

      for (const [label, a] of [["WITHOUT restoration", d.off], ["WITH restoration", d.on]]) {
        console.log(`\n  ${label}  (runOCR chose conf=${(a.bestConf*100).toFixed(0)}% psm=${a.bestPsm})`);
        for (const c of a.per) {
          const win = c.conf.toFixed(3) === a.bestConf.toFixed(3) ? " <- chosen" : "";
          console.log(`    ${c.tag.padEnd(9)} conf=${(c.conf*100).toFixed(0).padStart(3)}% `
            + `psm=${String(c.psm).padStart(2)} found=${JSON.stringify(c.found)}${win}`);
        }
      }

      // Show the neighbourhood of each lost field in the normalised text.
      console.log("\n  where the fields went (normalised text, both arms):");
      for (const f of ["2500", "2024CV00456"]) {
        const a = d.off.bestNorm.includes(f);
        const b = d.on.bestNorm.includes(f);
        if (a === b) continue;
        console.log(`    "${f}": without=${a}  with=${b}`);
        // Print a window of the 'with' text around the closest match of a
        // looser pattern so the reformatting is visible.
        const loose = f.replace(/0/g, "[0O]").replace(/1/g, "[1l]");
        const re = new RegExp(loose);
        const m = re.exec(d.on.bestNorm);
        if (m) {
          const lo = Math.max(0, m.index - 14), hi = Math.min(d.on.bestNorm.length, m.index + 34);
          console.log(`      with-arm reads: ...${d.on.bestNorm.slice(lo, hi)}...`);
        }
      }
      console.log("");
    }
  } catch (e) {
    console.log("error:", e && e.message);
  } finally {
    chrome.kill();
    server.close();
    process.exit(0);
  }
})();
