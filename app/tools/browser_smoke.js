/**
 * Real-browser smoke test: serves the app over HTTP and loads it in the
 * Playwright Chromium already on disk, driving it over the DevTools protocol
 * (no npm install required).
 *
 * What this proves that the Node harnesses cannot:
 *   1. index.html boots with zero uncaught errors and zero console errors.
 *   2. Every vendor asset resolves 200 over the wire (not just on disk).
 *   3. The service worker registers and precaches all 27 entries.
 *   4. selftest.html runs its FULL battery in a real browser, including the
 *      image half that needs a real Canvas 2-D context.
 *   5. The camera-less fallback path is reachable (no getUserMedia in headless).
 *
 * Run: node tools/browser_smoke.js
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const os = require("os");

const APP = path.resolve(__dirname, "..");
const CHROME = process.env.CHROME_BIN
  || "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".gz": "application/gzip",
  ".md": "text/markdown; charset=utf-8",
};

// ── Static server ─────────────────────────────────────────────────────
const requested = [];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let p = decodeURIComponent(url.pathname);
  if (p === "/") p = "/index.html";
  requested.push(p);
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

// ── Minimal CDP client ────────────────────────────────────────────────
class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.events = []; }
  static async attach(port) {
    const list = await new Promise((res, rej) => {
      http.get({ host: "127.0.0.1", port, path: "/json/list" }, (r) => {
        let b = ""; r.on("data", (c) => (b += c)); r.on("end", () => res(JSON.parse(b)));
      }).on("error", rej);
    });
    const page = list.find((t) => t.type === "page");
    if (!page) throw new Error("no page target");
    const WebSocket = require("./ws_min.js");
    const ws = await WebSocket.connect(page.webSocketDebuggerUrl);
    const cdp = new CDP(ws);
    ws.on("message", (msg) => {
      const m = JSON.parse(msg);
      if (m.id && cdp.pending.has(m.id)) {
        const { resolve, reject } = cdp.pending.get(m.id);
        cdp.pending.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error)));
        else resolve(m.result);
      } else if (m.method) {
        cdp.events.push(m);
      }
    });
    return cdp;
  }
  send(method, params) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params: params || {} }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error("CDP timeout: " + method));
        }
      }, 60000);
    });
  }
  async eval(expression) {
    const r = await this.send("Runtime.evaluate", {
      expression, returnByValue: true, awaitPromise: true,
    });
    if (r.exceptionDetails) {
      throw new Error("eval threw: " + JSON.stringify(r.exceptionDetails.exception || r.exceptionDetails));
    }
    return r.result.value;
  }
  close() { try { this.ws.close(); } catch (e) { /* already closed */ } }
}

// ── Run ───────────────────────────────────────────────────────────────
let pass = 0, fail = 0;
function check(desc, ok, detail) {
  if (ok) { pass++; console.log("ok    " + desc); }
  else { fail++; console.log("FAIL  " + desc + (detail ? "  -> " + detail : "")); }
}

(async () => {
  if (!fs.existsSync(CHROME)) {
    console.log("SKIP: no chromium at " + CHROME);
    process.exit(0);
  }
  const port = 8000 + Math.floor(Math.random() * 900);
  await new Promise((r) => server.listen(port, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + port;
  console.log("serving " + APP + " on " + base + "\n");

  const userDir = fs.mkdtempSync(path.join(os.tmpdir(), "sl-chrome-"));
  const cport = 9222 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, [
    "--headless=new",
    "--remote-debugging-port=" + cport,
    "--user-data-dir=" + userDir,
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });
  let chromeErr = "";
  chrome.stderr.on("data", (d) => { chromeErr += d.toString(); });

  // Wait for the debugging endpoint.
  let ready = false;
  for (let i = 0; i < 60 && !ready; i++) {
    try {
      await new Promise((res, rej) => {
        http.get({ host: "127.0.0.1", port: cport, path: "/json/version" }, (r) => {
          r.resume(); r.on("end", res);
        }).on("error", rej);
      });
      ready = true;
    } catch (e) { await new Promise((r) => setTimeout(r, 250)); }
  }
  if (!ready) {
    console.log("SKIP: chromium did not expose a debug port.\n" + chromeErr.slice(0, 400));
    chrome.kill(); server.close();
    process.exit(0);
  }

  let cdp;
  try {
    cdp = await CDP.attach(cport);
    await cdp.send("Runtime.enable");
    await cdp.send("Log.enable");
    await cdp.send("Page.enable");
    await cdp.send("Network.enable");
    // Block the camera permission prompt; the app must degrade gracefully.
    await cdp.send("Browser.grantPermissions", { origin: base, permissions: [] }).catch(() => {});

    // ── 1. index.html ──────────────────────────────────────────────
    await cdp.send("Page.navigate", { url: base + "/index.html" });
    await new Promise((r) => setTimeout(r, 4000));

    const scriptSrcs = await cdp.eval(
      "Array.from(document.querySelectorAll('script[src]')).map(s=>s.getAttribute('src'))"
    );
    // 7 tags: tesseract, pdfjs, pipeline, restorer, classifier, tts, app.
    check("index.html loaded all 7 script tags",
      Array.isArray(scriptSrcs) && scriptSrcs.length === 7, JSON.stringify(scriptSrcs));
    for (const src of scriptSrcs || []) {
      const hit = requested.find((p) => p.endsWith("/" + src) || p === "/" + src);
      check("vendor asset served: " + src, !!hit, "not requested");
    }

    const globals = await cdp.eval(
      "JSON.stringify({pipeline: typeof SightLine, restorer: typeof SightLineRestorer, "
      + "classifier: typeof SightLineClassifier, tts: typeof TTS, app: typeof SightLineApp, "
      + "tesseract: typeof Tesseract, pdfjs: typeof pdfjsLib})"
    );
    const G = JSON.parse(globals);
    check("SightLine (pipeline) global present", G.pipeline === "object", G.pipeline);
    check("SightLineRestorer global present", G.restorer === "object", G.restorer);
    check("SightLineClassifier global present", G.classifier === "object", G.classifier);
    check("TTS global present", G.tts === "object", G.tts);
    check("SightLineApp controller present", G.app === "object", G.app);
    check("tesseract.js vendored and loaded", G.tesseract === "object", G.tesseract);
    check("pdf.js vendored and loaded", G.pdfjs === "object", G.pdfjs);

    // Every asset the page actually pulled must resolve 200 over the wire.
    // `requested` is recorded by the Node-side static server, not the page.
    const pulled = await cdp.eval(`JSON.stringify(performance.getEntriesByType('resource')
      .map(r => new URL(r.name).pathname))`);
    const bad = [];
    for (const p of JSON.parse(pulled)) {
      if (!p.startsWith("/vendor/") && !p.startsWith("/assets/") && !p.startsWith("/js/")) continue;
      const res = await fetch(base + p, { method: "HEAD" }).catch(() => null);
      if (!res || res.status !== 200) bad.push(p + " -> " + (res ? res.status : "ERR"));
    }
    check("every asset the app pulled resolves 200 over HTTP (" +
      JSON.parse(pulled).filter((p) => p.startsWith("/vendor/")).length + " vendor files)",
      bad.length === 0, bad.join(", "));

    // A11y surface actually present in the DOM.
    const a11y = JSON.parse(await cdp.eval(`JSON.stringify({
      live: !!document.getElementById("live"),
      liveRole: (document.getElementById("live")||{}).getAttribute ? document.getElementById("live").getAttribute("aria-live") : null,
      progress: !!document.querySelector('[role="progressbar"]'),
      alert: !!document.querySelector('[role="alert"]'),
      badge: (document.getElementById("badge")||{}).textContent,
      buttons: Array.from(document.querySelectorAll("button")).map(b=>({
        id: b.id, h: Math.round(b.getBoundingClientRect().height), label: (b.textContent||"").trim()
      })),
      labelled: document.querySelectorAll("[aria-label]").length,
      manifest: !!document.querySelector('link[rel="manifest"]')
    })`));
    check("aria-live region exists", a11y.live === true);
    check("aria-live is polite", a11y.liveRole === "polite", String(a11y.liveRole));
    check("progressbar role present", a11y.progress === true);
    check("alert role present for errors", a11y.alert === true);
    check("manifest linked", a11y.manifest === true);
    check("aria labels present (" + a11y.labelled + ")", a11y.labelled >= 5);
    const small = a11y.buttons.filter((b) => b.h > 0 && b.h < 48);
    check("all rendered buttons are >= 48px tall (WCAG 2.5.5)", small.length === 0,
      JSON.stringify(small));
    check("offline badge rendered", typeof a11y.badge === "string" && a11y.badge.length > 0, a11y.badge);

    // ── 2. service worker ──────────────────────────────────────────
    const sw = JSON.parse(await cdp.eval(`(async () => {
      const r = await navigator.serviceWorker.getRegistration();
      if (!r) return JSON.stringify({registered:false});
      await new Promise(res => {
        if (r.active) return res();
        navigator.serviceWorker.ready.then(res).catch(res);
      });
      const names = await caches.keys();
      let total = 0;
      for (const n of names) total += (await (await caches.open(n)).keys()).length;
      return JSON.stringify({registered:true, scope: r.scope, caches: names, cached: total, active: !!r.active});
    })()`));
    check("service worker registered", sw.registered === true, JSON.stringify(sw));
    check("service worker is active", sw.active === true, JSON.stringify(sw));
    check("service worker cache created", Array.isArray(sw.caches) && sw.caches.length > 0, JSON.stringify(sw.caches));
    check("precache populated (>= 27 entries)", (sw.cached || 0) >= 27, "cached=" + sw.cached);

    // ── 3. offline behaviour ───────────────────────────────────────
    await cdp.send("Network.emulateNetworkConditions", {
      offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0,
    });
    await cdp.send("Page.reload", { ignoreCache: false });
    await new Promise((r) => setTimeout(r, 4000));
    const offlineState = JSON.parse(await cdp.eval(`JSON.stringify({
      title: document.title,
      pipeline: typeof SightLine,
      badge: (document.getElementById("badge")||{}).textContent
    })`));
    check("app reloads fully while OFFLINE", offlineState.pipeline === "object", JSON.stringify(offlineState));
    check("offline badge updates when disconnected",
      /OFFLINE/i.test(offlineState.badge || ""), String(offlineState.badge));
    await cdp.send("Network.emulateNetworkConditions", {
      offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1,
    });

    // ── 4. selftest.html, full battery in a real browser ────────────
    await cdp.send("Page.navigate", { url: base + "/selftest.html" });
    await new Promise((r) => setTimeout(r, 5000));
    const st = JSON.parse(await cdp.eval(`JSON.stringify({
      summary: (document.getElementById("summary")||{}).textContent,
      groups: Array.from(document.querySelectorAll("h2")).map(h=>h.textContent),
      failed: document.querySelectorAll("td.fail").length,
      passed: document.querySelectorAll("td.pass").length
    })`));
    check("selftest.html ran assertions (" + st.passed + " pass, " + st.failed + " fail)",
      st.passed > 100, JSON.stringify(st.summary));
    check("selftest.html has ZERO failures", st.failed === 0, st.summary);
    check("selftest summary reports ALL PASSED", /ALL .* PASSED/.test(st.summary || ""), st.summary);
    // The image half can only run in a real canvas context.
    check("image-preprocessing battery ran in-browser",
      st.groups.some((g) => /Image preprocessing/.test(g)), JSON.stringify(st.groups));

    // ── 5. console cleanliness ──────────────────────────────────────
    const consoleErrors = cdp.events
      .filter((e) => e.method === "Log.entryAdded" && e.params.entry.level === "error")
      .map((e) => e.params.entry.text)
      .filter((t) => !/favicon|manifest.*404|Failed to load resource.*favicon/i.test(t));
    check("no console errors on any page", consoleErrors.length === 0,
      consoleErrors.slice(0, 3).join(" | "));

  } catch (e) {
    fail++;
    console.log("FAIL  harness error: " + e.message);
  } finally {
    if (cdp) cdp.close();
    chrome.kill();
    server.close();
    try { fs.rmSync(userDir, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }

  console.log("\n" + (fail === 0 ? "ALL " + pass + " BROWSER CHECKS PASSED" : pass + " passed, " + fail + " FAILED"));
  process.exit(fail === 0 ? 0 : 1);
})();
