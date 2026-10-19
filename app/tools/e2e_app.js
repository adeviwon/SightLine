/*
 * End-to-end drive of the REAL app flow (not selftest): choose a photo,
 * wait for the pipeline to produce results, click Read Aloud, capture state.
 *
 * Run: SMOKE_BASE=http://localhost:8000/ node e2e_app.js [imagepath]
 */
const { chromium } = require("./pw_core.js")();   // see pw_core.js: not in node_modules here
const EXE = require("./chrome_path.js")();
const BASE = process.env.SMOKE_BASE || "https://adeviwon.github.io/SightLine/";
const IMG = process.argv[2] || "/tmp/test_document.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: EXE,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const consoleErrs = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrs.push(m.text().slice(0, 220));
  });
  page.on("pageerror", (e) => consoleErrs.push("PAGEERROR: " + String(e).slice(0, 220)));

  await page.goto(BASE, { waitUntil: "load", timeout: 60000 });
  await page.waitForTimeout(2500);

  // 1. upload the document through the real file input
  await page.setInputFiles("#file-input", IMG);

  // 2. wait for the REAL done-signal: #results unhides (or the error banner shows)
  const started = Date.now();
  try {
    await page.waitForFunction(
      () => {
        const r = document.getElementById("results");
        const b = document.getElementById("error-banner");
        const shown = r && !r.hidden;
        const errored = b && !b.hidden && (b.textContent || "").trim();
        return shown || errored;
      },
      { timeout: 240000 },
    );
  } catch (e) { /* fall through to dump whatever is on screen */ }
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  resultText = await page.evaluate(() => document.body.innerText);
  console.log("=== after upload (" + elapsed + "s) ===");
  console.log(resultText.replace(/\n{2,}/g, "\n").slice(0, 1800));
  console.log("=== console errors ===");
  console.log(consoleErrs.length ? consoleErrs.slice(0, 8).join("\n") : "(none)");

  // 3. Read Aloud if the button is visible
  const speakVisible = await page.locator("#btn-speak").isVisible().catch(() => false);
  if (speakVisible) {
    await page.click("#btn-speak");
    await page.waitForTimeout(4000);
    const speaking = await page.evaluate(() => {
      const s = window.speechSynthesis;
      return s ? { speaking: s.speaking || s.pending, voices: s.getVoices().length } : "no speechSynthesis";
    });
    console.log("=== TTS state after Read Aloud ===");
    console.log(JSON.stringify(speaking));
  } else {
    console.log("btn-speak not visible");
  }

  // 4. screenshot for the UX review
  await page.screenshot({ path: "/tmp/app_result.png", fullPage: true });
  console.log("screenshot: /tmp/app_result.png");

  await browser.close();
})().catch((e) => { console.error("E2E FAILED:", e.message); process.exit(1); });