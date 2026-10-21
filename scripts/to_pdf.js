// Render the judge-facing HTML artifacts to PDF.
//
// Why this exists: the Grand Final requires the slide deck SUBMITTED as a file 48 h
// ahead, and a self-contained HTML is not a submission format. It also closes a real
// risk -- both documents had only ever been verified STATICALLY (tag balance, id
// uniqueness, character estimates). Nobody had rendered them, so an overflowing slide
// would have been discovered on the projector.
//
// Usage: node scripts/to_pdf.js
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
// playwright-core is not installed in this repo's node_modules; app/tools/pw_core.js
// resolves it from wherever it lives on this box (and from a vendored copy if present).
// Usage there: `const { chromium } = require("./pw_core.js")()`.
const { chromium } = require(path.join(ROOT, "app", "tools", "pw_core.js"))();
const JOBS = [
  { src: "docs/22_GRAND_FINAL_DECK.html", out: "docs/22_GRAND_FINAL_DECK.pdf", landscape: true,
    note: "10 slides, one per A4 landscape page" },
  { src: "docs/evidence.html", out: "docs/evidence.pdf", landscape: false,
    note: "printable evidence sheet, A4 portrait" },
];

(async () => {
  // prefer the repo's own resolved chromium (app/tools/chrome_path.js) if it exists
  let exe;
  try {
    const cp = require(path.join(ROOT, "app", "tools", "chrome_path.js"));
    exe = typeof cp === "function" ? cp() : cp && (cp.path || cp.CHROME_PATH);
  } catch (e) { /* fall back to playwright's default */ }

  const browser = await chromium.launch(exe ? { executablePath: exe } : {});
  let failed = 0;
  for (const j of JOBS) {
    const src = path.join(ROOT, j.src);
    if (!fs.existsSync(src)) { console.log("MISSING " + j.src); failed++; continue; }
    const page = await browser.newPage();
    const errs = [];
    page.on("pageerror", (e) => errs.push(String(e && e.message || e)));
    // block everything that is not the file itself: these documents must be offline
    let blocked = 0;
    await page.route("**/*", (route) => {
      const u = route.request().url();
      if (u.startsWith("file://")) return route.continue();
      blocked++; return route.abort();
    });
    await page.goto("file://" + src, { waitUntil: "load" });
    await page.emulateMedia({ media: "print" });

    // measure overflow at print width before printing
    const probe = await page.evaluate(() => {
      const out = { slides: 0, overflowing: [], scrollH: [] };
      const slides = document.querySelectorAll(".slide");
      out.slides = slides.length;
      slides.forEach((s, i) => {
        if (s.scrollHeight > s.clientHeight + 2) {
          out.overflowing.push(i + 1);
          out.scrollH.push(s.scrollHeight + ">" + s.clientHeight);
        }
      });
      return out;
    });

    await page.pdf({
      path: path.join(ROOT, j.out),
      format: "A4",
      landscape: !!j.landscape,
      printBackground: true,
      margin: { top: "0", bottom: "0", left: "0", right: "0" },
    });
    await page.close();
    const kb = (fs.statSync(path.join(ROOT, j.out)).size / 1024).toFixed(0);
    console.log("=== " + j.src + " -> " + j.out + "  (" + kb + " KB)");
    console.log("    " + j.note);
    console.log("    slides=" + probe.slides +
      (probe.overflowing.length ? "  OVERFLOW on slides " + probe.overflowing.join(",") +
        "  [" + probe.scrollH.join(" ") + "]" : "  no overflow"));
    console.log("    external requests blocked=" + blocked + "  pageerrors=" + (errs.length ? errs.join(" | ") : 0));
    if (probe.overflowing.length || errs.length) failed++;
  }
  await browser.close();
  console.log(failed ? "RESULT: " + failed + " document(s) need attention" : "RESULT: all rendered clean");
  process.exitCode = failed ? 1 : 0;
})();
