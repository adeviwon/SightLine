/*
 * Two product-path checks against the REAL app:
 *   A. blank photo   -> the honest refusal state (never invented text)
 *   B. a PDF         -> the pdfjs render path feeds the same pipeline
 *
 * Run: SMOKE_BASE=http://127.0.0.1:8000/ node e2e_paths.js
 *      BLANK=/tmp/blank.png PDF=/tmp/letter.pdf  override the fixtures
 *
 * Exits non-zero when a check fails, so it can gate a deploy.
 *
 * Two bugs lived here, both fixed together because they produced the same
 * symptom (case B printing case A's numbers):
 *   1. `drive()` waited for the results panel to be VISIBLE — not for a NEW
 *      result. After case A the panel was already visible, so the wait returned
 *      instantly and case B read A's stale DOM. It now clears the panel first.
 *   2. It drove /tmp/test_document.pdf, a path nothing creates, and a missing
 *      file is indistinguishable from a slow scan: the wait timed out and the
 *      stale state got dumped. Missing fixtures now fail loudly.
 */
const fs = require("fs");
const { chromium } = require("./pw_core.js")();   // see pw_core.js: not in node_modules here
const EXE = require("./chrome_path.js")();
const BASE = process.env.SMOKE_BASE || "https://adeviwon.github.io/SightLine/";
const BLANK = process.env.BLANK || "/tmp/blank.png";
const PDF = process.env.PDF || "/tmp/letter.pdf";

let checks = 0, fails = 0;
function check(name, cond, detail) {
  checks++;
  if (!cond) fails++;
  console.log("  " + (cond ? "ok  " : "FAIL") + "  " + name + (cond || detail === undefined ? "" : "  -> " + detail));
}

async function resetResults(page) {
  await page.evaluate(() => {
    const r = document.getElementById("results");
    const b = document.getElementById("error-banner");
    if (r) r.hidden = true;
    if (b) { b.hidden = true; b.textContent = ""; }
  });
}

async function drive(page, file) {
  await resetResults(page);
  await page.setInputFiles("#file-input", file);
  let timedOut = false;
  try {
    await page.waitForFunction(
      () => {
        const r = document.getElementById("results");
        const b = document.getElementById("error-banner");
        return (r && !r.hidden) || (b && !b.hidden && (b.textContent || "").trim());
      },
      { timeout: 180000 },
    );
  } catch (e) { timedOut = true; }
  const state = await page.evaluate(() => ({
    meta: ((document.getElementById("result-meta") || {}).textContent || "").slice(0, 200),
    banner: ((document.getElementById("error-banner") || {}).textContent || "").slice(0, 160),
    warn: Array.from(document.querySelectorAll(".warning, [class*=warn]"))
      .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 2),
    summary: ((document.getElementById("summary-text") || {}).textContent || "").slice(0, 120),
    fields: Array.from(document.querySelectorAll("#results li, .field"))
      .map((e) => e.textContent.trim()).slice(0, 6),
  }));
  return { timedOut, ...state };
}

(async () => {
  for (const [label, f] of [["blank photo", BLANK], ["PDF", PDF]]) {
    if (!fs.existsSync(f)) {
      console.log("SKIP: no " + label + " fixture at " + f
        + " (create it with: bash run.sh py app/tools/make_fixtures.py)");
      process.exit(2);
    }
  }

  const browser = await chromium.launch({ executablePath: EXE, args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (e) => console.log("PAGEERROR:", String(e).slice(0, 160)));
  await page.goto(BASE, { waitUntil: "load", timeout: 60000 });
  await page.waitForFunction(() => typeof window.SightLineApp === "object", { timeout: 30000 });

  console.log("=== A. blank photo (refusal path) ===");
  const a = await drive(page, BLANK);
  console.log(JSON.stringify(a, null, 1));
  check("blank photo completes without hanging", !a.timedOut, "timed out");
  check("blank photo invents no fields", a.fields.length === 0, JSON.stringify(a.fields));
  check("blank photo is honest about the text (0% or a warning)",
    /Text confidence 0%|hard to read|No key details/i.test(a.meta + " " + a.warn.join(" ")),
    a.meta);

  console.log("\n=== B. PDF (pdfjs render path) ===");
  const b = await drive(page, PDF);
  console.log(JSON.stringify(b, null, 1));
  check("pdf scan completes without hanging", !b.timedOut, "timed out");
  check("pdf result is the PDF, not the previous photo",
    /letter\.pdf|\.pdf/i.test(b.meta), b.meta);
  check("pdf produced recognised text", !/Text confidence 0% \u00b7 .*$/.test(b.meta) || b.fields.length > 0,
    b.meta);

  await browser.close();
  console.log("\n" + (checks - fails) + "/" + checks + " path checks passed");
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error("E2E FAILED:", e.message); process.exit(1); });
