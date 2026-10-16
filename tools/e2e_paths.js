/*
 * Two product-path checks against the REAL app:
 *   A. blank photo -> the honest refusal/low-confidence state (never invented text)
 *   B. the same document as a PDF  -> pdfjs render path feeds the same pipeline
 * Run: SMOKE_BASE=http://localhost:8000/ node e2e_paths.js
 */
const { chromium } = require("playwright-core");
const EXE = "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";
const BASE = process.env.SMOKE_BASE || "https://adeviwon.github.io/SightLine/";

async function drive(page, file) {
  await page.setInputFiles("#file-input", file);
  try {
    await page.waitForFunction(
      () => {
        const r = document.getElementById("results");
        const b = document.getElementById("error-banner");
        return (r && !r.hidden) || (b && !b.hidden && (b.textContent || "").trim());
      },
      { timeout: 180000 },
    );
  } catch (e) { /* dump below */ }
  return page.evaluate(() => {
    const meta = (document.getElementById("result-meta") || {}).textContent || "";
    const banner = (document.getElementById("error-banner") || {}).textContent || "";
    const warn = Array.from(document.querySelectorAll(".warning, [class*=warn]"))
      .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 2);
    const summary = (document.getElementById("summary-text") || {}).textContent || "";
    const fields = Array.from(document.querySelectorAll("#results li, .field"))
      .map((e) => e.textContent.trim()).slice(0, 6);
    return { meta: meta.slice(0, 200), banner: banner.slice(0, 160), warn, summary: summary.slice(0, 120), fields };
  });
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXE, args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (e) => console.log("PAGEERROR:", String(e).slice(0, 160)));
  await page.goto(BASE, { waitUntil: "load", timeout: 60000 });
  await page.waitForTimeout(3000);

  console.log("=== A. blank photo (refusal path) ===");
  console.log(JSON.stringify(await drive(page, "/tmp/blank.png"), null, 1));

  // reset between runs
  const retake = await page.locator("#btn-retake").isVisible().catch(() => false);
  if (retake) await page.click("#btn-retake");

  console.log("=== B. PDF (pdfjs render path) ===");
  console.log(JSON.stringify(await drive(page, "/tmp/test_document.pdf"), null, 1));

  await browser.close();
})().catch((e) => { console.error("E2E FAILED:", e.message); process.exit(1); });