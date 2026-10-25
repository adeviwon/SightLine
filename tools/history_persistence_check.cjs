const fs = require("fs"),
  path = require("path"),
  http = require("http"),
  os = require("os");
const { chromium, webkit } = require("./pw_core.js")();
const ROOT = path.resolve(__dirname, "../.."),
  APP = path.join(ROOT, "app"),
  OUT = path.join(ROOT, "artifacts/bilingual-history");
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
};
const server = http.createServer((req, res) => {
  const file = path.resolve(
    APP,
    "." +
      (req.url.split("?")[0] === "/" ? "/index.html" : req.url.split("?")[0]),
  );
  if (
    !file.startsWith(APP + "/") ||
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
  const records = [];
  for (const [engine, type] of [
    ["chromium", chromium],
    ["webkit", webkit],
  ]) {
    if (process.env.HISTORY_ENGINE && process.env.HISTORY_ENGINE !== engine)
      continue;
    const profile = fs.mkdtempSync(
      path.join(os.tmpdir(), "sightline-history-"),
    );
    const opts = {
      headless: true,
      viewport: { width: 390, height: 844 },
      ...(engine === "chromium"
        ? {
            executablePath:
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          }
        : {}),
    };
    let c;
    let checks = [];
    function check(name, ok) {
      checks.push({ name, ok: !!ok });
      console.log((ok ? "PASS " : "FAIL ") + engine + " " + name);
      if (!ok) throw Error(name);
    }
    try {
      c = await type.launchPersistentContext(profile, opts);
      let p = await c.newPage();
      const base = "http://127.0.0.1:" + server.address().port;
      await p.addInitScript(()=>{window.cacheReports=[];navigator.serviceWorker?.addEventListener('message',e=>{if(e.data?.type==='precache-report')cacheReports.push(e.data)})});
      await p.goto(base);
      await p.waitForFunction(() => window.SightLineApp);
      await p.locator("#btn-preferences").click();
      await p.locator("#save-images").check();
      await p.locator("#preferences-dialog .dialog-done").click();
      await p
        .locator("#file-input")
        .setInputFiles(
          path.join(ROOT, "artifacts/aria-merge/sroie-X00016469612.jpg"),
        );
      await p.waitForFunction(() => !!SightLineApp.lastResult, null, {
        timeout: 120000,
      });
      await p.locator("#btn-save-scan").click();
      await p.locator("#consent-save").click();
      await p.waitForFunction(() =>
        /Saved on/.test(document.getElementById("save-status").textContent),
      );
      check(
        "optional image really persists as a Blob",
        await p.evaluate(async () => {
          const r = (await SightLineHistory.list())[0];
          return (
            r.image instanceof Blob &&
            r.image.size > 0 &&
            r.result.fields.every((f) => Number.isFinite(f.confidence))
          );
        }),
      );
      try { await p.waitForFunction(()=>/Ready offline|可離線/.test(document.getElementById('badge').textContent),null,{timeout:30000}); }
      catch(e){const report=await p.evaluate(()=>({badge:document.getElementById('badge').textContent,screen:document.body.dataset.screen,reports:cacheReports}));console.log('CACHE DIAG',JSON.stringify(report));fs.writeFileSync(path.join(OUT,engine+'-cache-failure.json'),JSON.stringify(report,null,2));throw e;}
      await c.close();
      c = await type.launchPersistentContext(profile, opts);
      await c.setOffline(true);
      p = await c.newPage();
      await p.goto(base);
      await p.waitForFunction(() => window.SightLineApp);
      await p.evaluate(() => {
        window.forbiddenOCR = 0;
        SightLine.runOCR = () => {
          forbiddenOCR++;
          throw Error("History must not run OCR");
        };
      });
      await p.locator("#btn-history").click();
      await p.locator(".history-record button").first().click();
      await p.waitForFunction(() => !!SightLineApp.lastResult);
      check(
        "real browser restart and offline reopen retains image and fields",
        await p.evaluate(
          () =>
            SightLineApp.lastResult.category === "receipt" &&
            document.getElementById("source-image").src.startsWith("blob:") &&
            forbiddenOCR === 0,
        ),
      );
      // Quota abort in a real transaction, not a fake successful request.
      await p.evaluate(() => {
        window.realPut = IDBObjectStore.prototype.put;
        IDBObjectStore.prototype.put = function () {
          throw new DOMException("quota", "QuotaExceededError");
        };
      });
      await p.locator("#btn-save-scan").click();
      await p.waitForFunction(() =>
        document
          .getElementById("save-status")
          .textContent.includes("Storage is full"),
      );
      check(
        "quota abort never claims Saved",
        await p.evaluate(
          () =>
            !document
              .getElementById("save-status")
              .textContent.startsWith("Saved"),
        ),
      );
      await p.evaluate(() => (IDBObjectStore.prototype.put = realPut));
      check(
        "failed write did not erase existing record",
        await p.evaluate(
          async () => (await SightLineHistory.list()).length === 1,
        ),
      );
      await p.locator("#btn-save-scan").click();
      await p.waitForFunction(
        () =>
          document.getElementById("save-status").textContent ===
          "Saved on this device",
      );
      check("save retries after a transaction failure", true);
      await p.evaluate(() => {
        window.realIDBOpen = IDBFactory.prototype.open;
        IDBFactory.prototype.open = function () {
          throw new DOMException("blocked", "SecurityError");
        };
      });
      console.log(
        "Permission probe",
        await p.evaluate(async () => {
          try {
            await SightLineHistory.list();
            return "unexpected success";
          } catch (e) {
            return {
              name: e.name,
              message: e.message,
              display: SightLineHistory.message(e),
            };
          }
        }),
      );
      await p.locator("#btn-save-scan").click();
      await p.waitForFunction(
        () => document.getElementById("save-status").textContent !== "Saving…",
      );
      console.log(
        "Permission UI",
        await p.locator("#save-status").textContent(),
      );
      check(
        "permission failure is rendered clearly",
        await p.evaluate(() =>
          document
            .getElementById("save-status")
            .textContent.includes("storage is blocked"),
        ),
      );

      check(
        "denied storage gives a specific recovery reason",
        await p.evaluate(() => !!SightLineApp.lastResult),
      );
      await p.evaluate(() => (IDBFactory.prototype.open = realIDBOpen));
      await p.evaluate(
        () =>
          new Promise((resolve, reject) => {
            const r = indexedDB.deleteDatabase("sightline-scans");
            r.onsuccess = resolve;
            r.onerror = () => reject(r.error);
            r.onblocked = () => reject(Error("blocked"));
          }),
      );
      await p.reload();
      await p.waitForFunction(() => window.SightLineApp);
      await p.locator("#btn-history").click();
      await p.waitForFunction(
        () => !document.getElementById("history-empty").hidden,
      );
      check(
        "erased browser database shows an honest empty history",
        await p.evaluate(
          async () => (await SightLineHistory.list()).length === 0,
        ),
      );
      records.push({ engine, checks });
    } finally {
      await c?.close();
    }
  }
  fs.writeFileSync(
    path.join(OUT, "persistence-checks.json"),
    JSON.stringify(records, null, 2),
  );
  server.close();
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
