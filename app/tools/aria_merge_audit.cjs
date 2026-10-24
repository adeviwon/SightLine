// Real public receipt through both shipping controllers; no OCR fixtures/mocks.
const fs = require("fs"),
  path = require("path"),
  http = require("http");
const { chromium } = require("./pw_core.js")();
const root = path.resolve(__dirname, "../.."),
  out = path.join(root, "artifacts/aria-merge");
const mime = {
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".html": "text/html",
  ".css": "text/css",
  ".json": "application/json",
};
const server = http.createServer((req, res) => {
  let url = new URL(req.url, "http://local"),
    aria = url.pathname.startsWith("/aria/"),
    base = aria
      ? process.env.ARIA_ROOT || path.resolve(root, "../aria-reader")
      : path.join(root, "app"),
    p = aria ? url.pathname.slice(5) : url.pathname;
  let file = path.resolve(base, "." + (p.endsWith("/") ? p + "index.html" : p));
  if (
    !file.startsWith(base + "/") ||
    !fs.existsSync(file) ||
    fs.statSync(file).isDirectory()
  ) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {
    "Content-Type": mime[path.extname(file)] || "application/octet-stream",
  });
  fs.createReadStream(file).pipe(res);
});
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + server.address().port;
  const b = await chromium.launch({
    executablePath:
      process.env.CHROME_BIN ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  try {
    for (const app of ["sightline", "aria"]) {
      let c = await b.newContext({ viewport: { width: 390, height: 844 } }),
        p = await c.newPage();
      let errors = [];
      p.on("pageerror", (e) => errors.push(e.message));
      await p.goto(base + (app === "aria" ? "/aria/" : "/"));
      await p.waitForFunction(
        (app) => (app === "aria" ? window.__ariaReady : window.SightLineApp),
        app,
      );
      let t = Date.now();
      await p
        .locator("#file-input")
        .setInputFiles(path.join(out, "sroie-X00016469612.jpg"));
      await p.waitForFunction(
        (app) =>
          app === "aria"
            ? window.__ariaLastScan?.()
            : window.SightLineApp.lastResult,
        app,
        { timeout: 180000 },
      );
      let r = await p.evaluate(
        (app) =>
          app === "aria"
            ? {
                ...window.__ariaLastScan(),
                text: document.querySelector("#transcript").textContent,
                diagnostics: window.__ariaScanJson(),
              }
            : window.SightLineApp.lastResult,
        app,
      );
      r.testElapsed = Date.now() - t;
      r.errors = errors;
      fs.writeFileSync(
        path.join(
          out,
          app + "-" + (process.env.AUDIT_PHASE || "before") + ".json",
        ),
        JSON.stringify(r, null, 2),
      );
      await p.screenshot({
        path: path.join(
          out,
          app + "-" + (process.env.AUDIT_PHASE || "before") + ".png",
        ),
        fullPage: true,
      });
      console.log(
        app,
        JSON.stringify({
          engine: r.engine || r.record?.ocr,
          confidence: r.ocrConfidence || r.record?.ocr?.mean_confidence,
          category: r.category || r.extraction?.docType,
          fields: r.fields || r.extraction?.fields,
          ms: r.ms || r.testElapsed,
          errors,
        }),
      );
      if (app === "sightline" && process.env.AUDIT_PHASE === "final") {
        for (const [width, height] of [
          [320, 568],
          [1440, 1000],
        ]) {
          await p.setViewportSize({ width, height });
          await p.screenshot({
            path: path.join(out, `receipt-final-${width}.png`),
            fullPage: true,
          });
          if (
            await p.evaluate(
              () => document.documentElement.scrollWidth > innerWidth,
            )
          )
            throw Error("receipt layout overflow");
        }
      }
      await c.close();
    }
  } finally {
    await b.close();
    server.close();
  }
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
