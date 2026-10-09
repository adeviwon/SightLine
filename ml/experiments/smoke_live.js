/*
 * Runtime smoke test of the LIVE deployed PWA.
 *
 * Two checks, both against https://adeviwon.github.io/SightLine/:
 *   1. index loads: console errors, failed requests, service-worker registration
 *   2. selftest.html "Run all tests": the in-browser pipeline test — the first
 *      time the rewritten pipeline has ever executed in a real browser.
 *
 * Run: node /tmp/smoke.js   (playwright-core installed in /tmp)
 */
const { chromium } = require("playwright-core");

const BASE = process.env.SMOKE_BASE || "https://adeviwon.github.io/SightLine/";
const EXE = "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";

(async () => {
  const browser = await chromium.launch({
    executablePath: EXE,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });

  const consoleErrs = [];
  const failedReqs = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") {
      consoleErrs.push(`${m.type()}: ${m.text().slice(0, 200)}`);
    }
  });
  page.on("requestfailed", (r) =>
    failedReqs.push(`${r.failure()?.errorText || "?"} ${r.url().slice(0, 120)}`));

  // ---- 1. index ----
  await page.goto(BASE, { waitUntil: "load", timeout: 60000 });
  const title = await page.title();
  await page.waitForTimeout(4000); // let the SW register

  const sw = await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) return "no-sw-support";
    const regs = await navigator.serviceWorker.getRegistrations();
    return regs.length ? `registered(${regs[0].scope})` : "none";
  });

  // ---- 2. selftest ----
  await page.goto(BASE + "selftest.html", { waitUntil: "load", timeout: 60000 });
  const runBtn = page.locator("#run");
  const runText = (await runBtn.textContent() || "").trim();
  await runBtn.click();
  // models load + run; allow up to 4 minutes
  await page.waitForFunction(
    () => {
      const s = document.querySelector("#summary");
      return s && !/Not run yet\./.test(s.textContent);
    },
    { timeout: 240000 },
  );
  await page.waitForTimeout(3000); // let late async rows finish
  const summary = (await page.locator("#summary").textContent()).trim();
  const rows = await page.evaluate(() =>
    Array.from(document.querySelectorAll("#out table tr")).map((tr) => {
      const tds = tr.querySelectorAll("td");
      if (!tds.length) return null;
      const pass = tds[0].textContent.trim() === "PASS";
      return { pass, line: `${tds[1]?.textContent?.trim()} | ${tds[0]?.textContent?.trim()} | ${(tds[2] || {}).textContent?.trim() || ""}` };
    }).filter(Boolean),
  );
  const failing = rows.filter((r) => !r.pass);

  console.log("=== INDEX ===");
  console.log("title      :", title);
  console.log("serviceWorker:", sw);
  console.log("console err/warn:", consoleErrs.length, consoleErrs.slice(0, 5));
  console.log("failed requests:", failedReqs.length, failedReqs.slice(0, 5));
  console.log("\n=== SELFTEST ===");
  console.log("run button :", runText);
  console.log("summary    :", summary);
  console.log("checks:");
  console.log("  passing:", rows.length - failing.length, " failing:", failing.length);
  failing.forEach((r) => console.log("  FAIL:", r.line));
  const onnx = rows.filter((r) => /ONNX|session|inference|timesteps|ctc/.test(r.line));
  onnx.forEach((r) => console.log("  onnx:", r.line));

  await browser.close();
})().catch((e) => { console.error("SMOKE FAILED:", e.message); process.exit(1); });