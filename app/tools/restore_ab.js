/**
 * Does restoration help or hurt OCR, sample by sample?
 *
 * THE QUESTION
 * ------------
 * The E2E harness reports one aggregate number (field recovery 5/12 both ways),
 * which cannot distinguish "the model is neutral" from "the model helps on some
 * profiles and destroys others, netting out to zero".
 *
 * The per-sample data suggests the second. On heavily blurred input the gate
 * saturates to weight 1.0 and the model runs at full strength — but the Python
 * evaluation already showed 0% field recovery on handheld_light / handheld_heavy
 * / low_light for EVERY arm, including restorer. So the model is being applied
 * at full strength exactly where it cannot help, and adding artefacts to an
 * image whose detail was never captured.
 *
 * That is a plausible mechanism, not a measurement. This harness measures it:
 * same image, same session, both arms, comparing what OCR actually returns.
 *
 * The comparison that matters is OCR character yield and field recovery, not
 * PSNR. PSNR asks "is the image closer to the clean original"; the product asks
 * "can the user hear their dosage". A change can improve the first and damage
 * the second.
 *
 * Run:  node tools/restore_ab.js
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { CDP } = require("./cdp_min.js");

const ROOT = path.resolve(__dirname, "..");
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
      // Samples live in /tmp/sightline_samples, outside the app root, and the
      // page fetches them as "/sample/<name>". Map that prefix onto the real
      // directory instead of copying fixtures into the shipped app.
      let p;
      if (rel.startsWith("/sample/")) {
        p = path.join("/tmp/sightline_samples", rel.slice("/sample/".length));
      } else {
        p = path.join(ROOT, rel);
      }
      if (!p.startsWith(ROOT) && !p.startsWith("/tmp/sightline_samples")) {
        res.writeHead(403); res.end(); return;
      }
      fs.readFile(p, (e, d) => {
        if (e) { res.writeHead(404); res.end("not found: " + rel); return; }
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
  const chrome = spawn(CHROME, ["--headless=new", "--no-sandbox", "--disable-gpu",
    "--remote-debugging-port=9351", "about:blank"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 3500));

  // Samples and ground truth come from the renderer's output dir, same as
  // e2e_ocr.js — not from app/, which ships no fixtures.
  const SAMPLES = "/tmp/sightline_samples";
  const manifestPath = path.join(SAMPLES, "manifest.json");
  if (!fs.existsSync(manifestPath)) {
    console.log("SKIP: no sample manifest at " + manifestPath + ". Run:\n"
      + "  bash ../run.sh samples");
    chrome.kill(); server.close(); process.exit(0);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const samples = (manifest.samples || [])
    .filter((s) => fs.existsSync(s.file));

  try {
    const cdp = await CDP.attach(9351);
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${port}/index.html` });
    await new Promise((r) => setTimeout(r, 5000));

    const init = await cdp.eval(
      "(async()=>{const s=await SightLineRestorer.init();return s?'SESSION OK':'NULL';})()", 300000);
    console.log("restorer: " + init + "\n");

    console.log("sample".padEnd(32) + "w".padStart(5)
      + "  |  chars off/on   fields off/on   conf off/on");
    console.log("-".repeat(76));

    let offChars = 0, onChars = 0, offF = 0, onF = 0, n = 0;

    for (const s of samples) {
      const r = await cdp.eval(`(async()=>{
        const img = new Image();
        img.src = "/sample/" + ${JSON.stringify(path.basename(s.file))};
        await img.decode();
        const pp = SightLine.preprocess(img, 3200, {restoreMode:"none"});
        const want = ${JSON.stringify(s.wantFields || [])};

        const runArm = async (useModel) => {
          let cands;
          if (useModel) {
            const r2 = await SightLineRestorer.armRestorerClahe(pp.canvas, {});
            cands = r2.candidates;
          } else {
            const p = SightLine;
            cands = [pp.canvas, p.clahe(pp.canvas,2.0,8), p.binarize(p.unsharp(p.clahe(pp.canvas,2.0,8),1.0,1))];
          }
          const o = await SightLine.runOCR(cands, ()=>{});
          const norm = SightLine.normalizeOCRText(o.text);
          return { chars:(o.text||"").length, conf:o.confidence,
                   found: SightLine.fields_found(norm, ${JSON.stringify(s.doc)}) };
        };

        const off = await runArm(false);
        const on  = await runArm(true);
        return JSON.stringify({off, on, w: SightLineRestorer.status().weight});
      })()`, 600000);

      const d = JSON.parse(r);
      offChars += d.off.chars; onChars += d.on.chars;
      offF += d.off.found.length; onF += d.on.found.length; n++;
      const mark = d.off.chars !== d.on.chars ? (d.on.chars > d.off.chars ? " +" : " -") : "  ";
      console.log(path.basename(s.file).replace(/\.(png|jpg)$/i, "")
        .slice(0, 30).padEnd(32)
        + d.w.toFixed(2).padStart(5)
        + "  | " + String(d.off.chars).padStart(4) + " -> " + String(d.on.chars).padStart(4) + mark
        + "   " + JSON.stringify(d.off.found).slice(0, 18) + " -> "
        + JSON.stringify(d.on.found).slice(0, 18).padEnd(18)
        + "  " + (d.off.conf * 100).toFixed(0) + " -> " + (d.on.conf * 100).toFixed(0));
    }

    console.log("\n=== OFF (classical only)  vs  ON (restorer+clahe) ===");
    console.log("total OCR chars : " + offChars + " -> " + onChars
      + "  (" + (onChars >= offChars ? "+" : "") + (onChars - offChars) + ")");
    console.log("total fields    : " + offF + " -> " + onF
      + "  (" + (onF >= offF ? "+" : "") + (onF - offF) + ")");
    console.log("\n" + (onChars > offChars
      ? "Restoration INCREASES total OCR yield."
      : onChars < offChars
        ? "Restoration DECREASES total OCR yield -- it is destroying legibility."
        : "Restoration is OCR-neutral in aggregate."));
    console.log("Aggregate can hide offsetting wins and losses: read the per-sample table.");
  } catch (e) {
    console.log("error:", e && e.message);
  } finally {
    chrome.kill();
    server.close();
    process.exit(0);
  }
})();
