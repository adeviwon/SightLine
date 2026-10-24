const http = require("http");
const fs = require("fs"),
  path = require("path");
const { chromium } = require("./pw_core.js")();
const AxeBuilder = require(
  process.env.AXE_CORE || "@axe-core/playwright",
).default;
const ROOT = path.resolve(__dirname, "../.."),
  OUT = path.join(ROOT, "artifacts/bilingual-history");
let base;
const server = http.createServer((req, res) => {
  const file = path.resolve(ROOT, "app", "." + (new URL(req.url, "http://local").pathname === "/" ? "/index.html" : new URL(req.url, "http://local").pathname));
  if (!file.startsWith(path.join(ROOT, "app") + "/") || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  const types = {".html":"text/html", ".js":"text/javascript", ".mjs":"text/javascript", ".wasm":"application/wasm", ".css":"text/css", ".json":"application/json"};
  res.writeHead(200, {"Content-Type":types[path.extname(file)] || "application/octet-stream"});
  const stream = fs.createReadStream(file);
  res.on("close", () => stream.destroy());
  stream.pipe(res);
});
(async () => {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  base = "http://127.0.0.1:" + server.address().port;
  const b = await chromium.launch({
    headless: true,
    executablePath:
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    args: ["--disable-gpu"],
  });
  let checks = [];
  function check(name, ok, detail) {
    checks.push({ name, ok: !!ok, detail });
    console.log((ok ? "PASS " : "FAIL ") + name);
    if (!ok) throw Error(name + " " + JSON.stringify(detail));
  }
  try {
    for (const [width, height] of [
      [320, 568],
      [390, 844],
      [768, 1024],
      [1440, 1000],
    ]) {
      const c = await b.newContext({ viewport: { width, height } }),
        p = await c.newPage();
      p.on("pageerror", (e) => console.log("PAGE ERROR", width, e.message));
      p.on("requestfailed", (r) => console.log("REQUEST FAILED", width, r.url(), r.failure()?.errorText));
      await p.goto(base);
      await p.waitForFunction(() => window.SightLineApp);
      await p.locator("#visible-language").selectOption("zh-HK");
      await p.waitForTimeout(150);
      await p.screenshot({
        path: path.join(OUT, "review-home-zh-" + width + ".png"),
        fullPage: true,
      });
      check(
        "Chinese home fits " + width,
        await p.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await p
        .locator("#file-input")
        .setInputFiles(
          path.join(ROOT, "artifacts/aria-merge/sroie-X00016469612.jpg"),
        );
      if (width === 390) {
        await p.locator("#processing-view").waitFor({ state: "visible" });
        await p.screenshot({ path: path.join(OUT, "review-processing-zh-390.png") });
      }
      await p.waitForFunction(() => !!SightLineApp.lastResult, null, {
        timeout: 120000,
      });
      await p.mouse.click(4, 190);
      await p.evaluate(() => document.activeElement?.blur());
      await p.waitForTimeout(100);
      console.log("Receipt focus", width, await p.evaluate(() => ({active:document.activeElement.id || document.activeElement.tagName, skipTop:getComputedStyle(document.querySelector(".skip-link")).top, skipTransform:getComputedStyle(document.querySelector(".skip-link")).transform})));
      await p.screenshot({
        path: path.join(OUT, "review-receipt-zh-" + width + ".png"),
        fullPage: true,
      });
      check(
        "Chinese receipt fits " + width,
        await p.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      if (width === 390) {
        const a = await new AxeBuilder({ page: p })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze();
        fs.writeFileSync(
          path.join(OUT, "axe-receipt-zh.json"),
          JSON.stringify(a.violations, null, 2),
        );
        check(
          "Chinese receipt automated accessibility",
          !a.violations.length,
          a.violations.map((v) => v.id),
        );
      }
      await p.locator("#btn-save-scan").click();
      await p.screenshot({
        path: path.join(OUT, "review-consent-zh-" + width + ".png"),
      });
      await p.locator("#consent-save").click();
      await p.waitForFunction(() =>
        document.getElementById("save-status").textContent.includes("已儲存"),
      );
      await p.locator("#btn-history").click();
      await p.waitForFunction(
        () => document.querySelectorAll(".history-record").length === 1,
      );
      await p.mouse.click(4, 190);
      await p.evaluate(() => document.activeElement?.blur());
      await p.waitForTimeout(100);
      await p.screenshot({
        path: path.join(OUT, "review-history-zh-" + width + ".png"),
        fullPage: true,
      });
      check(
        "Chinese populated history fits " + width,
        await p.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      if (width === 390) {
        const a = await new AxeBuilder({ page: p })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze();
        fs.writeFileSync(
          path.join(OUT, "axe-history-zh.json"),
          JSON.stringify(a.violations, null, 2),
        );
        check(
          "Chinese history automated accessibility",
          !a.violations.length,
          a.violations.map((v) => v.id),
        );
        await p.locator("#btn-preferences").click();
        await p.screenshot({ path: path.join(OUT, "review-preferences-zh-390.png") });
        await p.locator("#preferences-dialog .dialog-done").click();
      }
      await p.locator("#visible-language").selectOption("en-GB");
      await p.mouse.click(4, 190);
      await p.evaluate(() => document.activeElement?.blur());
      await p.waitForTimeout(100);
      check(
        "English history label restores " + width,
        (await p.locator("#history-title").textContent()).trim() ===
          "Your saved scans",
      );
      await p.screenshot({
        path: path.join(OUT, "review-history-en-" + width + ".png"),
        fullPage: true,
      });
      await c.close();
    }
    fs.writeFileSync(
      path.join(OUT, "visual-checks.json"),
      JSON.stringify({ checks }, null, 2),
    );
  } finally {
    await b.close();
    server.close();
  }
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
