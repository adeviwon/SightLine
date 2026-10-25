/**
 * Verify the honesty gate: on documents the pipeline reads badly, the app MUST
 * surface a warning. A scan that silently returns a partial result is the
 * dangerous outcome for a medical app — the user has no signal to distrust it.
 *
 * Drives the real app controller (SightLineApp.runScan) in real Chromium over
 * the rendered capture samples, then reads the warning box from the DOM.
 *
 *   node tools/warning_gate.js
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const os = require("os");
const { spawn } = require("child_process");

const APP = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const MANIFEST = path.join(SAMPLES, "manifest.json");
const CHROME = require("./chrome_path.js")();

const MIME = {
  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  // .mjs MUST be a JavaScript MIME type: the app dynamic-imports
  // vendor/ort/ort.min.mjs, and a browser refuses to execute a module served as
  // application/octet-stream. Its absence here made every scan in this harness
  // fail with "Failed to fetch dynamically imported module".
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".png": "image/png",
  ".wasm": "application/wasm", ".gz": "application/gzip", ".md": "text/markdown; charset=utf-8",
  ".ico": "image/x-icon",
};
const { CDP } = require("./cdp_min.js");

let pass = 0, fail = 0;
function check(desc, ok, detail) {
  if (ok) { pass++; console.log("  ok    " + desc); }
  else { fail++; console.log("  FAIL  " + desc + (detail ? "  -> " + detail : "")); }
}

(async () => {
  if (!fs.existsSync(MANIFEST)) {
    console.log("SKIP: no manifest. Run: bash run.sh shell < app/tools/render_samples.py");
    process.exit(0);
  }
  const M = JSON.parse(fs.readFileSync(MANIFEST, "utf8"));
  const samples = M.samples.filter((s) => fs.existsSync(s.file));

  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (p === "/") p = "/index.html";
    const s = samples.find((x) => p === "/sample/" + path.basename(x.file));
    const file = s ? s.file : path.join(APP, p);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end("nf"); return; }
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    fs.createReadStream(file).pipe(res);
  });
  const port = 8000 + Math.floor(Math.random() * 900);
  await new Promise((r) => server.listen(port, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + port;

  const userDir = fs.mkdtempSync(path.join(os.tmpdir(), "sl-warn-"));
  const cport = 9222 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, [
    "--headless=new", "--remote-debugging-port=" + cport,
    "--user-data-dir=" + userDir, "--no-sandbox", "--disable-gpu",
    "--disable-dev-shm-usage", "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });

  let ready = false;
  for (let i = 0; i < 60 && !ready; i++) {
    try {
      await new Promise((res, rej) => {
        http.get({ host: "127.0.0.1", port: cport, path: "/json/version" },
          (r) => { r.resume(); r.on("end", res); }).on("error", rej);
      });
      ready = true;
    } catch (e) { await new Promise((r) => setTimeout(r, 250)); }
  }
  if (!ready) { console.log("SKIP: chromium unavailable"); chrome.kill(); server.close(); process.exit(0); }

  let cdp;
  const rows = [];
  try {
    cdp = await CDP.attach(cport);
    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: base + "/index.html" });
    // Wait for the CONTROLLER, not for a wall-clock guess. 3.5s was enough on a
    // warm machine and too short on a cold one, and the failure mode was a
    // misleading TypeError deep in the eval below.
    let appReady = false;
    for (let i = 0; i < 60 && !appReady; i++) {
      try {
        appReady = await cdp.eval("typeof SightLineApp === 'object'");
      } catch (e) { /* page still navigating */ }
      if (!appReady) await new Promise((r) => setTimeout(r, 500));
    }
    if (!appReady) throw new Error("the app controller never appeared on the page");

    for (const s of samples) {
      const name = path.basename(s.file);
      // Drive the REAL controller so the warning logic under test is the one
      // that ships, not a copy of it.
      const out = await cdp.eval(`(async () => {
        const img = new Image();
        img.src = "/sample/${name}";
        await img.decode();
        let res = null, scanErr = "";
        try {
          res = await SightLineApp.runScan(img, "${name}");
        } catch (e) {
          scanErr = String((e && e.message) || e);
        }
        const box = document.getElementById("warning-box");
        if (!res) {
          // runScan's own catch returns lastResult, which is null on a scan
          // that threw — so a null result means the SCAN failed. Report that,
          // instead of dying on res.ocrConfidence and blaming the harness.
          const banner = document.getElementById("error-banner");
          const shown = banner && !banner.hidden ? (banner.textContent || "").trim() : "";
          return JSON.stringify({ scanError: scanErr || shown || "scan produced no result" });
        }
        return JSON.stringify({
          ocrConf: res.ocrConfidence,
          category: res.category,
          nFields: res.fields.length,
          warnings: res.warnings,
          domVisible: !box.hidden && box.textContent.trim().length > 0,
          domText: box.textContent.trim(),
          // A hard medical invariant: the summary must never assert a figure
          // that is not present in the text the pipeline actually understood.
          //
          // Two oracle corrections, both found by this test going red on
          // correct behaviour:
          //
          // 1. The oracle is NOT fields_found(). That is Python's deliberately
          //    strict all-or-nothing evaluation metric; it rejects plenty of
          //    figures the app correctly shows (1200mg of calcium is not a
          //    prescription dosage and never should have matched).
          // 2. The oracle is the NORMALISED text, not the raw OCR. The summary
          //    is built after normalizeOCRText, which is the whole point of the
          //    normalizer — it recovers "5O0m9" as "500mg". Running the oracle
          //    on raw text called that recovery a fabrication.
          //
          // What must never happen is inventing a figure that is not there.
          assertedDoses: (res.summary.match(/\\d+\\s?mg/gi) || []),
          readDoses: (SightLine.extractEntities(
            SightLine.normalizeOCRText(res.ocrText)) || [])
            .filter((e) => e.label === "DOSAGE")
            .map((e) => e.text.replace(/\\s/g, ""))
        });
      })()`, 300000);
      const r = JSON.parse(out);
      if (r.scanError) {
        fail++;
        console.log("FAIL  " + name + ": " + r.scanError);
        continue;
      }
      const warned = r.warnings.length > 0;
      rows.push({ name, r, warned });

      console.log(s.doc + " / " + s.profile
        + "  OCR " + Math.round(r.ocrConf * 100) + "%  fields " + r.nFields
        + "  warnings " + r.warnings.length);

      if (r.ocrConf < 0.60) {
        check("  low-confidence scan is warned about", warned,
          "no warning at " + Math.round(r.ocrConf * 100) + "%");
        check("  warning reaches the DOM", r.domVisible, r.domText);
      } else if (r.ocrConf >= 0.85 && r.nFields > 0) {
        check("  clean confident scan is not nagged", !warned,
          JSON.stringify(r.warnings));
      }
      if (r.nFields === 0) {
        check("  empty extraction is warned about", warned, "silent empty result");
      }
      // The invariant that matters most: never assert a dose that is not in
      // the recognised text. This is the medical safety property — a summary
      // that invents a figure is far worse than one that admits a gap.
      if (s.doc === "prescription") {
        const bogus = r.assertedDoses.filter((d) =>
          !r.readDoses.includes(d.replace(/\s/g, "")));
        check("  summary asserts no unread dosage", bogus.length === 0,
          "asserted " + JSON.stringify(bogus) + " but text only had "
          + JSON.stringify(r.readDoses));
      }
    }
  } catch (e) {
    fail++;
    console.log("FAIL  harness error: " + e.message);
  } finally {
    if (cdp) cdp.close();
    chrome.kill();
    server.close();
    try { fs.rmSync(userDir, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }

  // ── a pure-JS honesty pin: no document evidence -> no fields ───────────────
  //
  // The speckle page proves the app can hallucinate a PLAUSIBLE field from noise
  // at a confidence INSIDE the range real reads occupy (0.678 against 0.687-0.974),
  // so a confidence threshold cannot catch it; and the ported validators cannot
  // either, because the invented value "110011 000" is a legal 9-digit account
  // number and a legal identifier. Document EVIDENCE catches it: the speckle
  // scores 0 on every document keyword while real reads score 1-17 even when they
  // classify as the wrong type. Pinned here because this is the honesty gate and
  // it is the cheapest place to notice a regression.
  try {
    const P = require(path.resolve(__dirname, "..", "js", "pipeline.js"));
    const SL = P.SightLine || P;
    const gibberish = "EAX CE E 0RO 121G000\n0O0TX 3AXICEC2S0 00 00\nAROE NT 10021";
    const readable = "PRESCRIPTION\nPATIENT JANE DOE DOB 04021979\n"
      + "RX AMOXICILLIN 500MG CAPSULES\nPRESCRIBER DR A AHMED MBBS\nDATE 01102026";
    const g = SL.understand(gibberish);
    const r = SL.understand(readable);
    check("a read with no document evidence asserts no fields",
      g.documentEvidence === 0 && g.fields.length === 0,
      "evidence=" + g.documentEvidence + " fields=" + JSON.stringify(g.fields));
    check("a readable document still surfaces its fields",
      r.fields.length > 0,
      "evidence=" + r.documentEvidence + " fields=" + JSON.stringify(r.fields.map((f) => f.value)));
  } catch (e) {
    check("the no-evidence gate check ran", false, e.message);
  }

  const warned = rows.filter((x) => x.warned).length;
  console.log("\nwarned " + warned + "/" + rows.length + " scans");
  console.log(fail === 0 ? "ALL " + pass + " HONESTY CHECKS PASSED" : pass + " passed, " + fail + " FAILED");
  process.exit(fail === 0 ? 0 : 1);
})();
