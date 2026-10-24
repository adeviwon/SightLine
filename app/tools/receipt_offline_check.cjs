const fs = require("fs"),
  http = require("http"),
  path = require("path");
const { chromium, webkit } = require("./pw_core.js")();
const root = path.resolve(__dirname, "../.."),
  app = path.join(root, "app"),
  out = path.join(root, "artifacts/aria-merge");
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
};
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, "http://local").pathname;
  const file = path.resolve(
    app,
    "." + (pathname === "/" ? "/index.html" : pathname),
  );
  if (
    !file.startsWith(app + "/") ||
    !fs.existsSync(file) ||
    fs.statSync(file).isDirectory()
  ) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {
    "Content-Type": mime[path.extname(file)] || "application/octet-stream",
    "Content-Length": fs.statSync(file).size,
  });
  fs.createReadStream(file).pipe(res);
});
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  let results = [];
  for (const [name, type] of [
    ["chromium", chromium],
    ["webkit", webkit],
  ]) {
    const b = await type.launch(
      name === "chromium"
        ? {
            headless: true,
            executablePath:
              process.env.CHROME_BIN ||
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          }
        : { headless: true },
    );
    try {
      const c = await b.newContext({ viewport: { width: 390, height: 844 } }),
        p = await c.newPage();
      let errors = [];
      p.on("pageerror", (e) => errors.push(e.message));
      let base = "http://127.0.0.1:" + server.address().port;
      await p.goto(base);
      await p.waitForFunction(
        () => document.getElementById("badge").textContent === "Ready offline",
        null,
        { timeout: 60000 },
      );
      await c.setOffline(true);
      await p.reload();
      await p.waitForFunction(() => window.SightLineApp);
      let records = [];
      for (let n = 0; n < 2; n++) {
        const old = await p.evaluate(() => SightLineApp.lastResult?.timestamp);
        await p
          .locator("#file-input")
          .setInputFiles(path.join(out, "sroie-X00016469612.jpg"));
        await p.waitForFunction(
          (old) =>
            !!SightLineApp.lastResult &&
            SightLineApp.lastResult.timestamp !== old,
          old,
          { timeout: 90000 },
        );
        records.push(await p.evaluate(() => SightLineApp.lastResult));
        if (n === 0)
          await p.screenshot({
            path: path.join(out, `${name}-offline-receipt.png`),
            fullPage: true,
          });
        if (n === 0)
          await p
            .locator("#btn-retake")
            .click()
            .catch(async () => {
              await p.evaluate(() => SightLineApp.cancel?.());
            });
      }
      let record = {
        name,
        offline: true,
        badge: await p.locator("#badge").textContent(),
        errors,
        records,
      };
      fs.writeFileSync(
        path.join(out, name + "-offline.json"),
        JSON.stringify(record, null, 2),
      );
      if (
        records.some(
          (r) => r.engine.ocr !== "ppocrv5" || r.category !== "receipt",
        ) ||
        errors.length
      )
        throw Error("offline scan assertions failed");
      console.log(
        "PASS",
        name,
        "offline cold reload and two real PP-OCRv5 receipt scans",
        records.map((r) => r.ms),
      );
      results.push(name);
    } finally {
      await b.close();
    }
  }
  console.log("PASS " + results.length + " browser engines");
  server.close();
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
