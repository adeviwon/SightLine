/**
 * iOS check: run the app in WebKit — the engine iOS Safari actually uses.
 * Chromium smoke (browser_smoke.js) proves the app works in Blink; this proves
 * it parses and runs in WebKit, which is where the iOS-specific failures live
 * (regex lookbehind is a PARSE-TIME syntax error, not a runtime one, and
 * WebAssembly SIMD is the gate on the whole ONNX path).
 *
 * Usage: node tools/ios_webkit_check.js [--keep-open]
 *
 * Requires the Playwright WebKit browser on disk (it is, at
 * ~/.cache/ms-playwright/webkit-*). Set PW_CORE to a playwright-core module
 * directory if the default paths are wrong.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const APP = path.resolve(__dirname, "..");

const PW_CANDIDATES = [
  process.env.PW_CORE,
  "/home/ubuntu/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core",
  "/home/ubuntu/OpenMAIC/node_modules/.pnpm/playwright-core@1.58.2/node_modules/playwright-core",
].filter(Boolean);

const MIME = {
  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
};

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    let p = decodeURIComponent(url.pathname);
    if (p === "/") p = "/index.html";
    const file = path.join(APP, p);
    if (!file.startsWith(APP) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found: " + p);
      return;
    }
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-store",
      "Service-Worker-Allowed": "/",
    });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

// WebAssembly feature probes, run inside the page. The SIMD module bytes are
// the canonical wasm-feature-detect snippet; if the probe returns false on an
// engine that HAS SIMD, the bytes are wrong, so the runner cross-checks it
// against WebKit (which supports SIMD) and says so.
const PROBES = `({
  simd: (() => {
    try {
      return WebAssembly.validate(new Uint8Array([
        0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11
      ]));
    } catch (e) { return "threw: " + e.message; }
  })(),
  sab: typeof SharedArrayBuffer !== "undefined",
  coi: !!globalThis.crossOriginIsolated,
  wasm: typeof WebAssembly === "object",
  ua: navigator.userAgent,
  lookbehind: (() => { try { new RegExp("(?<=a)b"); return true; } catch (e) { return false; } })(),
  tts: typeof window.TTS,
})`;

async function main() {
  const { webkit } = require(PW_CANDIDATES.find((p) => fs.existsSync(p)));

  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port;

  const browser = await webkit.launch();
  // iPhone-ish viewport. UA stays WebKit's own: faking an iOS UA would only
  // change what the app *thinks* it is, and this run is about the engine.
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });

  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => pageErrors.push(String((e && e.message) || e)));

  // ── 1. index.html: does the app shell boot cleanly in WebKit? ──────────
  await page.goto(base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const probe = await page.evaluate(PROBES);

  console.log("── engine ──────────────────────────────────────────────");
  console.log("  UA             :", probe.ua);
  console.log("  WebAssembly    :", probe.wasm);
  console.log("  WASM SIMD      :", probe.simd, probe.simd === true ? "(ok)" : "(ORT cannot run here)");
  console.log("  SharedArrayBuf :", probe.sab, "| crossOriginIsolated:", probe.coi);
  console.log("  lookbehind re  :", probe.lookbehind ? "supported" : "UNSUPPORTED (parse error in shipped code)");
  console.log("  window.TTS     :", probe.tts);
  console.log("  page errors    :", pageErrors.length, pageErrors.slice(0, 5));
  console.log("  console errors :", consoleErrors.length, consoleErrors.slice(0, 5));

  // ── 2. selftest.html: full battery incl. real ONNX inference ───────────
  pageErrors.length = 0;
  await page.goto(base + "/selftest.html", { waitUntil: "load" });
  await page.waitForTimeout(800);
  await page.locator(".technical-checks > summary").click();
  await page.click("#run");
  // The page auto-runs the TEXT battery on load, so a bare wait on the summary
  // returns immediately with the 145-assertion text-only result while the
  // click's `run("all")` is still going. Wait for the ONNX group to exist AND
  // the summary to be final: that is the only combination that proves the
  // image + ONNX batteries ran too.
  await page.waitForFunction(
    () => {
      const s = document.getElementById("summary");
      if (!s || !/ASSERTIONS/.test(s.textContent)) return false;
      const h2 = Array.from(document.querySelectorAll("h2")).map((h) => h.textContent);
      return h2.some((t) => /ONNX runtime/.test(t)) && h2.some((t) => /Image preprocessing/.test(t));
    },
    null,
    { timeout: 300000 }
  );
  const result = await page.evaluate(() => ({
    summary: document.getElementById("summary").textContent,
    groups: Array.from(document.querySelectorAll("h2")).map((h) => h.textContent),
    failures: Array.from(document.querySelectorAll("td.fail"))
      .map((td) => td.parentElement.textContent.replace(/\s+/g, " ").trim())
      .slice(0, 20),
  }));

  console.log("\n── selftest.html in WebKit ─────────────────────────────");
  console.log("  " + result.summary);
  for (const g of result.groups) console.log("   ·", g);
  if (result.failures.length) {
    console.log("\n  FAILURES:");
    for (const f of result.failures) console.log("   ✗", f);
  }
  if (pageErrors.length) {
    console.log("\n  PAGE ERRORS:", pageErrors.slice(0, 5));
  }

  const ok = /ALL \d+ ASSERTIONS PASSED/.test(result.summary) && pageErrors.length === 0;
  console.log("\n" + (ok ? "WEBKIT: PASS" : "WEBKIT: FAIL"));

  if (!process.argv.includes("--keep-open")) {
    await browser.close();
    server.close();
  }
  process.exit(ok ? 0 : 1);
}

main().catch((e) => { console.error("runner failed:", e); process.exit(2); });
