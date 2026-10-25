// Browser capability negative controls; no mocked speech playback.
const fs = require("fs"),
  path = require("path"),
  http = require("http");
const { chromium, webkit } = require("./pw_core.js")();
const root = path.resolve(__dirname, "../.."),
  app = path.join(root, "app"),
  out = path.join(root, "artifacts/speech-repair");
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
};
const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://local"),
    file = path.resolve(
      app,
      "." + (u.pathname === "/" ? "/index.html" : u.pathname),
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
  });
  fs.createReadStream(file).pipe(res);
});
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const records = [];
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
      let checks = [];
      function check(name, ok) {
        checks.push({ name, ok: !!ok });
        console.log((ok ? "PASS " : "FAIL ") + name);
        if (!ok) throw Error(name);
      }
      const c = await b.newContext({ viewport: { width: 320, height: 568 } }),
        p = await c.newPage();
      let errors = [];
      p.on("pageerror", (e) => errors.push(e.message));
      await p.goto("http://127.0.0.1:" + server.address().port);
      await p.waitForFunction(
        () =>
          window.SightLineApp &&
          document.getElementById("badge").textContent === "Ready offline",
        null,
        { timeout: 60000 },
      );
      await p.locator("#btn-help").click();
      await p.locator("[data-open-device]").first().click();
      const f = await (
        await p.locator("#device-frame").elementHandle()
      ).contentFrame();
      await f.waitForFunction(() => window.SightLineDeviceReport);
      check(
        name + " verified current cache",
        await f.evaluate(() => SightLineDeviceReport.installed),
      );
      await f.evaluate(() => {
        window.realSIMD = WebAssembly.validate;
        WebAssembly.validate = () => false;
      });
      let old = await f.evaluate(() => SightLineDeviceReport.checkedAt);
      await f.locator("#device-recheck").click();
      await f.waitForFunction(
        (old) => SightLineDeviceReport.checkedAt !== old,
        old,
      );
      check(
        name + " no-SIMD negative control reports incompatibility",
        await f.evaluate(
          () =>
            SightLineDeviceReport.rows.find(
              (r) => r.title === "Reading software",
            ).ok === false,
        ),
      );
      await f.evaluate(() => {
        WebAssembly.validate = realSIMD;
        window.realAvailable = TTS.available;
        TTS.available = () => false;
      });
      old = await f.evaluate(() => SightLineDeviceReport.checkedAt);
      await f.locator("#device-recheck").click();
      await f.waitForFunction(
        (old) => SightLineDeviceReport.checkedAt !== old,
        old,
      );
      check(
        name + " no speech API is reported without pretending audio worked",
        await f.evaluate(
          () =>
            !SightLineDeviceReport.rows.find((r) => r.title === "Read Aloud")
              .ok &&
            !SightLineDeviceReport.audioVerified &&
            document.getElementById("test-voice").disabled,
        ),
      );
      await f.evaluate(() => (TTS.available = realAvailable));
      await f.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      const back = await p.locator("#close-device").boundingBox();
      check(
        name + " top exit stays in viewport",
        back.y >= 0 && back.y + back.height <= 568,
      );
      await p.keyboard.press("Escape");
      await p.waitForFunction(
        () => !document.getElementById("device-dialog").open,
      );
      check(
        name + " Escape restores reader",
        await p.locator("#home-view").isVisible(),
      );
      await p.locator("#btn-preferences").click();
      await p.locator("#appearance").selectOption("dark");
      await p.locator("#preferences-dialog [data-close-dialog]").last().click();
      await p.locator("#btn-help").click();
      await p.locator("[data-open-device]").first().click();
      await f.waitForFunction(
        () => document.documentElement.dataset.theme === "dark",
      );
      check(name + " checks follow reading appearance", true);
      await f.evaluate(
        () => (document.documentElement.style.fontSize = "44px"),
      );
      check(
        name + " narrow checks do not overflow at doubled text",
        await f.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await f.evaluate(() => (document.documentElement.style.fontSize = ""));
      check(name + " no runtime errors", !errors.length);
      records.push({ name, checks, errors });
      await c.close();
    } finally {
      await b.close();
    }
  }
  fs.writeFileSync(
    path.join(out, "device-negative-controls.json"),
    JSON.stringify(records, null, 2),
  );
  server.close();
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
