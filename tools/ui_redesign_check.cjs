/** Real UI regression. No OCR mocks: uploads synthetic images through the real
 * file input, including the actual worker and classifier. Separate failure and
 * camera cases inject network failures / a browser test camera explicitly.
 * PW_CORE, CHROME_BIN, AXE_CORE point to development-only test installations.
 */
const fs = require("fs"),
  path = require("path"),
  http = require("http");
const { chromium } = require("./pw_core.js")();
const AxeBuilder = require(
  process.env.AXE_CORE || "@axe-core/playwright",
).default;
const APP = path.resolve(__dirname, ".."),
  ROOT = path.dirname(APP);
const OUT = path.resolve(ROOT, "artifacts/redesign");
fs.mkdirSync(OUT, { recursive: true });
const fixture = path.join(OUT, "fixtures/synthetic-single-label.png");
const types = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};
let failOfflineAsset = false;
const server = http.createServer((req, res) => {
  const pathname = decodeURIComponent(
    new URL(req.url, "http://local").pathname,
  );
  if (failOfflineAsset && pathname.endsWith("/ppocrv5-mobile-rec.onnx")) {
    res.writeHead(503);
    res.end("Test-only unavailable asset");
    return;
  }
  const file = path.resolve(
    APP,
    "." + (pathname === "/" ? "/index.html" : pathname),
  );
  if (
    !file.startsWith(APP + path.sep) ||
    !fs.existsSync(file) ||
    fs.statSync(file).isDirectory()
  ) {
    res.writeHead(404);
    res.end();
    return;
  }
  res.writeHead(200, {
    "Content-Type": types[path.extname(file)] || "application/octet-stream",
    "Cache-Control": "no-store",
  });
  fs.createReadStream(file).pipe(res);
});
const checks = [],
  records = [],
  errors = [];
let browser, base;
function check(name, ok, detail) {
  checks.push({ name, ok: !!ok, detail });
  console.log(
    (ok ? "PASS " : "FAIL ") + name + (ok ? "" : " " + JSON.stringify(detail)),
  );
}
async function shot(page, name, fullPage = false) {
  await page.waitForTimeout(120);
  await page.screenshot({ path: path.join(OUT, name + ".png"), fullPage });
}
async function layout(page, name) {
  const g = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
    targets: [...document.querySelectorAll("button,summary,select")]
      .filter((e) => e.checkVisibility())
      .filter((e) => e.getBoundingClientRect().height < 48)
      .map((e) => ({
        text: e.textContent.trim().slice(0, 45),
        height: e.getBoundingClientRect().height,
      })),
  }));
  check(name + " has no horizontal overflow", g.scroll <= g.width, g);
  check(name + " controls meet 48px minimum", !g.targets.length, g.targets);
}
async function axe(page, name) {
  const r = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  fs.writeFileSync(
    path.join(OUT, "axe-" + name + ".json"),
    JSON.stringify(r.violations, null, 2),
  );
  check(
    name + " WCAG A/AA automated checks",
    !r.violations.length,
    r.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      nodes: v.nodes.map((n) => n.target),
    })),
  );
}
async function scan(page, file = fixture, screenshot) {
  const old = await page.evaluate(() => SightLineApp.lastResult?.timestamp);
  await page.locator("#file-input").setInputFiles(file);
  await page.locator("#processing-view").waitFor({ state: "visible" });
  if (screenshot) await shot(page, screenshot);
  await page.waitForFunction(
    (old) =>
      (SightLineApp.lastResult?.timestamp !== old &&
        !!SightLineApp.lastResult) ||
      document.body.dataset.screen === "error",
    old,
    { timeout: 120000 },
  );
  const r = await page.evaluate(() => SightLineApp.lastResult);
  check(
    "real OCR completes " + path.basename(file),
    !!r,
    await page.locator("#error-banner").textContent(),
  );
  if (!r) throw Error("Actual scan failed");
  records.push({
    fixture: path.basename(file),
    engine: r.engine,
    ms: r.ms,
    characters: r.ocrText.length,
    fields: r.fields.length,
    speechCharacters: r.speechText?.length,
  });
  return r;
}
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  base = "http://127.0.0.1:" + server.address().port;
  browser = await chromium.launch({
    executablePath: require("./chrome_path.js")(),
    headless: true,
    args: [
      "--disable-gpu",
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    colorScheme: "light",
    reducedMotion: "reduce",
    permissions: ["camera"],
  });
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(String(e)));
  const external = [];
  page.on("request", (r) => {
    if (/^https?:/.test(r.url()) && !r.url().startsWith(base))
      external.push(r.url());
  });
  await page.goto(base);
  await page.waitForFunction(
    () => document.getElementById("badge").textContent === "Ready offline",
  );
  check(
    "ready screen has exactly two entry choices",
    (await page.locator("#home-view .entry-actions button:visible").count()) ===
      2,
  );
  check(
    "redundant source tabs removed",
    (await page.locator("[role=tablist]").count()) === 0,
  );
  for (const [w, h] of [
    [320, 568],
    [390, 844],
    [768, 1024],
    [1440, 1000],
  ]) {
    await page.setViewportSize({ width: w, height: h });
    await layout(page, "home " + w);
    await shot(page, "home-" + w);
    await shot(page, "home-full-" + w, true);
  }
  await axe(page, "home");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#btn-preferences").click();
  await axe(page, "preferences");
  await page.selectOption("#appearance", "dark");
  check(
    "dark appearance applies",
    (await page.locator("html").getAttribute("data-theme")) === "dark",
  );
  await shot(page, "preferences-dark-390");
  await axe(page, "preferences-dark");
  await page.selectOption("#appearance", "light");
  await page.selectOption("#speak-lang", "zh-HK");
  await page.locator("#preferences-dialog .dialog-done").click();
  check(
    "dialog restores focus to its opener",
    await page
      .locator("#btn-preferences")
      .evaluate((e) => e === document.activeElement),
  );
  await page.reload();
  await page.locator("#btn-preferences").click();
  check(
    "language preference persists",
    (await page.locator("#speak-lang").inputValue()) === "zh-HK",
  );
  await page.selectOption("#speak-lang", "en-GB");
  await page.locator("#preferences-dialog .dialog-done").click();
  await page
    .getByRole("button", { name: "Scan with Camera", exact: true })
    .click();
  await page.waitForFunction(
    () => document.getElementById("video").videoWidth > 0,
  );
  await shot(page, "camera-test-device-390");
  await layout(page, "camera");
  await page.setViewportSize({ width: 320, height: 568 });
  await layout(page, "camera 320");
  const shutter = await page.locator("#btn-capture").boundingBox();
  check(
    "camera shutter stays within the narrow phone viewport",
    !!shutter && shutter.y + shutter.height <= 568,
    shutter,
  );
  await shot(page, "camera-test-device-320");
  await axe(page, "camera");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Take photo", exact: true }).click();
  check(
    "camera photo review has read and retake choices",
    (await page.locator("#btn-read-photo").isVisible()) &&
      (await page.locator("#btn-camera-retake").isVisible()),
  );
  check(
    "camera stops after taking photo",
    await page.locator("#video").evaluate((e) => e.srcObject === null),
  );
  await shot(page, "camera-review-390");
  await page.getByRole("button", { name: "Retake photo", exact: true }).click();
  await page.waitForFunction(
    () => document.getElementById("video").videoWidth > 0,
  );
  await page.locator("#btn-cancel-camera").click();
  check(
    "cancel camera returns home",
    (await page.locator("body").getAttribute("data-screen")) === "home",
  );
  check(
    "camera stopped on cancel",
    await page.locator("#video").evaluate((e) => e.srcObject === null),
  );
  // Genuine file input and real shipped OCR/classifier, not injected results.
  const r = await scan(page, fixture, "processing-390");
  check(
    "selected PP-OCRv5 genuinely used",
    r.engine.ocr === "ppocrv5",
    r.engine,
  );
  check(
    "optional classifier still used",
    r.engine.classifier === "onnx",
    r.engine,
  );
  check(
    "literal medicine name is prominent",
    await page
      .locator("#fields-list")
      .textContent()
      .then((t) => /amoxicillin/i.test(t)),
  );
  check("medicine results separate strength, amount each time and timing", await page.evaluate(() => {
    const values = new Map(Array.from(document.querySelectorAll('#fields-list .field-row')).map(row => [row.querySelector('th').textContent, row.querySelector('td > span').textContent]));
    return values.get('Medicine type') === 'capsules' && values.get('Strength on the label') === '500 mg' &&
      values.get('Amount each time') === 'one capsule' && values.get('When to take it') === 'three times daily';
  }));
  await page.locator('#spoken-summary').evaluate((details) => { details.open = true; });
  check("read-aloud summary splits medicine information into separate cards", await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll('#spoken-preview .spoken-summary-section'));
    const all = cards.flatMap(card => Array.from(card.querySelectorAll('th'))).map(node => node.textContent);
    return cards.length >= 4 && cards.every(card => !!card.querySelector('table[aria-labelledby]')) && ['Medicine name', 'Medicine type', 'Strength on the label', 'Amount each time', 'When to take it'].every(label => all.includes(label));
  }));
  check("full text stays available without dominating structured medicine results", await page.evaluate(() => !document.getElementById('transcript-details').open));
  await page.evaluate(() => {
    window.__sectionsFields = JSON.stringify(SightLineApp.lastResult.fields);
    window.__sectionsOcr = SightLine.runOCR;
    window.__sectionsOcrCount = 0;
    SightLine.runOCR = (...args) => { __sectionsOcrCount++; return __sectionsOcr(...args); };
  });
  await page.selectOption('#visible-language', 'zh-HK');
  check("medicine section labels and supported values switch to Traditional Chinese", await page.evaluate(() => {
    const row=Array.from(document.querySelectorAll('.field-row')).find(row=>row.querySelector('th').textContent==='每次用量');
    return row?.querySelector('td > span').textContent === '一粒膠囊';
  }));
  check("read-aloud preview keeps distinct Traditional Chinese cards", await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll('#spoken-preview .spoken-summary-section'));
    return cards.length >= 4 && cards.some(card => card.textContent.includes('藥物')) &&
      cards.some(card => card.textContent.includes('每次用量')) &&
      !document.querySelector('#spoken-preview').querySelector(':scope > p:not(.spoken-summary-note)');
  }));
  await page.selectOption('#visible-language', 'en-GB');
  check("medicine language switching preserves extracted fields and never reruns OCR", await page.evaluate(() => __sectionsOcrCount===0 && JSON.stringify(SightLineApp.lastResult.fields)===__sectionsFields));
  await page.evaluate(() => { SightLine.runOCR=__sectionsOcr; });
  check(
    "processing and ready screens hidden on results",
    !(await page.locator("#home-view").isVisible()) &&
      !(await page.locator("#processing-view").isVisible()),
  );
  check(
    "results have no synthetic preview notice",
    !(await page.locator("#warning-box").textContent()).includes(
      "Visual preview",
    ),
  );
  check(
    "results focused for screen-reader navigation",
    await page
      .locator("#results-title")
      .evaluate((e) => e === document.activeElement),
  );
  for (const [w, h] of [
    [320, 568],
    [390, 844],
    [768, 1024],
    [1440, 1000],
  ]) {
    await page.setViewportSize({ width: w, height: h });
    await page.evaluate(() => window.scrollTo(0, 0));
    await layout(page, "results " + w);
    if (w === 320 || w === 390) {
      const first = await page.locator(".field-row .val").first().boundingBox();
      check(
        "first recognised value is visible at " + w,
        !!first && first.y + first.height <= h,
        first,
      );
    }
    await shot(page, "results-" + w);
    await shot(page, "results-full-" + w, true);
  }
  await axe(page, "results");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#source-details summary").click();
  check(
    "original image can be compared",
    (await page.locator("#source-image").isVisible()) &&
      (await page.locator("#source-image").evaluate((e) => e.naturalWidth > 0)),
  );
  await shot(page, "original-comparison-390", true);
  await page.locator("#source-details summary").click();
  // Stub only speech transport: assert the utterance, not physical audio output.
  await page.evaluate(() => {
    window.__spoken = "";
    TTS.speak = async (text) => {
      window.__spoken = text;
    };
  });
  await page.locator("#btn-speak").click();
  check(
    "English reads trusted words without inferred prescription advice",
    await page.evaluate(
      () =>
        __spoken.includes("AMOXICILLIN") &&
        __spoken.includes("500 mg") &&
        !__spoken.includes("This is a medical prescription"),
    ),
  );
  check(
    "speech control preserves its icon",
    (await page.locator("#btn-speak svg").count()) === 1,
  );
  await page.evaluate(() => {
    window.__speechAvailable = TTS.available;
    TTS.available = () => false;
    SightLineApp.render(SightLineApp.lastResult);
  });
  check(
    "speech-device unavailability is distinguished from unclear OCR",
    (await page.locator("#btn-speak").isVisible()) &&
      (await page.locator("#btn-speak").isDisabled()) &&
      (await page
        .locator("#speech-note")
        .textContent()
        .then((t) => t.includes("isn’t available"))),
  );
  await page.evaluate(() => {
    TTS.available = window.__speechAvailable;
    SightLineApp.render(SightLineApp.lastResult);
  });
  await page.evaluate(() => (document.documentElement.style.fontSize = "40px"));
  for (const w of [320, 390, 1440]) {
    await page.setViewportSize({ width: w, height: 844 });
    await layout(page, "results 200% text " + w);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await shot(page, "results-text-200-390", true);
  await page.evaluate(() => (document.documentElement.style.fontSize = ""));
  await page.locator("#btn-retake").click();
  await page.evaluate(() => (document.documentElement.style.fontSize = "40px"));
  await page.setViewportSize({ width: 320, height: 568 });
  await layout(page, "home 200% text");
  await page.evaluate(() => (document.documentElement.style.fontSize = ""));
  await page.locator("#file-input").setInputFiles({
    name: "broken.png",
    mimeType: "image/png",
    buffer: Buffer.from("not an image"),
  });
  await page.locator("#error-view").waitFor({ state: "visible" });
  check(
    "bad file never leaves previous results active",
    await page.evaluate(
      () =>
        !SightLineApp.lastResult && document.getElementById("results").hidden,
    ),
  );
  await shot(page, "error-invalid-file-320");
  await axe(page, "invalid-file");
  const chooserEvent = page.waitForEvent("filechooser");
  await page.locator("#btn-retry").click();
  const chooser = await chooserEvent;
  await chooser.setFiles(fixture);
  await page.waitForFunction(
    () => document.body.dataset.screen === "results",
    {},
    { timeout: 120000 },
  );
  check(
    "retry recovers from invalid image",
    await page.evaluate(
      () => SightLineApp.lastResult?.engine.ocr === "ppocrv5",
    ),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  // Reload offline destroys the warm worker: this is a cold browser-page read.
  await context.setOffline(true);
  await page.reload();
  check(
    "offline reload reaches ready screen",
    await page.locator("#home-view").isVisible(),
  );
  await scan(page);
  check(
    "offline cold-page OCR still uses PP-OCRv5",
    await page.evaluate(
      () => SightLineApp.lastResult?.engine.ocr === "ppocrv5",
    ),
  );
  await context.setOffline(false);
  await page.locator("#btn-retake").click();
  await scan(page, "/tmp/sightline_samples/prescription__studio_clean.png");
  const safety = await page.evaluate(() => ({
    speech: SightLineApp.lastResult.speechText,
    fields: SightLineApp.lastResult.fields.length,
  }));
  check(
    "partial prescription retains conservative refusal",
    safety.speech === "" && safety.fields === 0,
    safety,
  );
  check(
    "unverified medicine offers reading help with its safety explanation",
    (await page.locator("#btn-speak").isVisible()) &&
      (await page.locator("#btn-speak").isEnabled()) &&
      (await page.locator("#btn-speak").textContent()).includes("Hear reading help") &&
      (await page.locator("#speech-note").isVisible()) &&
      (await page.locator("#btn-speak").getAttribute("aria-describedby")) === "speech-note",
  );
  await page.evaluate(() => { window.__spoken = ""; TTS.speak = async text => { window.__spoken = text; }; });
  await page.locator("#btn-speak").click();
  check("unverified medicine speaks only a safe recovery message", await page.evaluate(() =>
    __spoken === "Not clear enough to read aloud. Try another photo in better light, or ask someone to check the label with you."));
  await shot(page, "results-needs-checking-390", true);
  await axe(page, "needs-checking");
  await page.locator("#btn-retake").click();
  await scan(page, "/tmp/sightline_samples/banking__studio_clean.png");
  check(
    "other supported document fields still render",
    (await page.locator(".field-row").count()) > 0,
  );
  await shot(page, "results-banking-390", true);
  await page.locator("#btn-retake").click();
  const general = await scan(
    page,
    path.join(OUT, "fixtures/synthetic-general-document.png"),
  );
  check(
    "unsupported document retains genuinely recognised words",
    !!general.speechText && general.ocrText.includes("GARDEN"),
  );
  check(
    "clear unstructured text is not described as blurry OCR",
    await page
      .locator("#empty-fields h2")
      .textContent()
      .then((t) => t === "Here are the words we found."),
  );
  check(
    "unknown document is not presented as a prescription",
    await page
      .locator("#doc-type")
      .textContent()
      .then((t) => t === "Document"),
  );
  await shot(page, "results-unstructured-390", true);
  await page.locator("#btn-retake").click();
  const multi = await scan(
    page,
    path.join(OUT, "fixtures/synthetic-two-medicines.png"),
  );
  check(
    "multiple-medicine test really has speech-eligible OCR",
    !!multi.speechText,
    { speechCharacters: multi.speechText.length },
  );
  check(
    "unmatched doses are not combined in structured results",
    (await page.locator("#association-notice").isVisible()) &&
      !(await page.locator(".field-row .label").allTextContents()).some((t) =>
        /dose|frequency/i.test(t),
      ),
  );
  await shot(page, "results-multiple-medicines-390", true);
  await page.locator("#btn-language").click();
  await page.selectOption("#speak-lang", "zh-HK");
  await page.locator("#preferences-dialog .dialog-done").click();
  await page.evaluate(() => (window.__spoken = ""));
  await page.locator("#btn-speak").click();
  check(
    "Cantonese refuses unmatched medicine directions",
    (await page.locator("#btn-speak").isEnabled()) && (await page
      .locator("#speech-note")
      .textContent()
      .then(
        (t) => t.includes("can’t safely match") || t.includes("未能安全配對"),
      )) && (await page.evaluate(() => window.__spoken === "未能安全配對藥名同用量。請揀英文聽原文，再同藥劑師確認。")),
  );
  await page.locator("#btn-language").click();
  await page.selectOption("#speak-lang", "en-GB");
  await page.locator("#preferences-dialog .dialog-done").click();
  await page.locator("#btn-retake").click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await scan(page, fixture, "processing-1440");
  await shot(page, "results-1440");
  await shot(page, "results-full-1440", true);
  check("no uncaught application errors", errors.length === 0, errors);
  check("no external runtime requests", external.length === 0, external);
  await context.close();
  // Genuine required-asset HTTP failure. The product must never claim offline
  // readiness for an incomplete cache, and the Help retry must recover.
  failOfflineAsset = true;
  const offlineContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
  });
  const offlinePage = await offlineContext.newPage();
  await offlinePage.goto(base);
  await offlinePage.waitForFunction(
    () =>
      document.getElementById("badge").textContent.includes("did not finish"),
    {},
    { timeout: 60000 },
  );
  check(
    "failed required cache never claims offline readiness",
    await offlinePage
      .locator("#badge")
      .textContent()
      .then((t) => t !== "Ready offline"),
  );
  await offlinePage.locator("#btn-help").click();
  check(
    "offline saving failure offers a useful retry",
    await offlinePage.locator("#btn-save-offline").isVisible(),
  );
  failOfflineAsset = false;
  await offlinePage.locator("#btn-save-offline").click();
  await offlinePage.waitForFunction(
    () => document.getElementById("badge").textContent === "Ready offline",
    {},
    { timeout: 60000 },
  );
  check(
    "saving retry recovers the complete required cache",
    await offlinePage.locator("#btn-save-offline").isHidden(),
  );
  await offlineContext.close();
  // Interrupted model download: verify UI cancellation cannot resurrect a result.
  const cancelContext = await browser.newContext({
    serviceWorkers: "block",
    viewport: { width: 320, height: 568 },
  });
  await cancelContext.route(
    "**/models/ppocrv5-mobile-det.onnx",
    async (route) => {
      await new Promise((r) => setTimeout(r, 500));
      await route.continue().catch(() => {});
    },
  );
  const cancel = await cancelContext.newPage();
  await cancel.goto(base);
  await cancel.locator("#file-input").setInputFiles(fixture);
  await cancel.locator("#processing-view").waitFor({ state: "visible" });
  const cancelButton = await cancel
    .locator("#btn-cancel-reading")
    .boundingBox();
  check(
    "processing cancel stays within the narrow phone viewport",
    !!cancelButton && cancelButton.y + cancelButton.height <= 568,
    cancelButton,
  );
  await shot(cancel, "processing-320");
  await cancel.locator("#btn-cancel-reading").click();
  await cancel.waitForTimeout(900);
  check(
    "cancelled scan cannot restore stale results",
    await cancel.evaluate(
      () => document.body.dataset.screen === "home" && !SightLineApp.lastResult,
    ),
  );
  await cancelContext.unroute("**/models/ppocrv5-mobile-det.onnx");
  await scan(cancel);
  check(
    "new scan works after cancellation",
    await cancel.evaluate(
      () => SightLineApp.lastResult?.engine.ocr === "ppocrv5",
    ),
  );
  await cancelContext.close();
  // A real failed model fetch, followed by real inference on retry.
  const failureContext = await browser.newContext({
    serviceWorkers: "block",
    viewport: { width: 390, height: 844 },
  });
  await failureContext.route("**/models/ppocrv5-mobile-det.onnx", (route) =>
    route.abort(),
  );
  const failure = await failureContext.newPage();
  await failure.goto(base);
  await failure.locator("#file-input").setInputFiles(fixture);
  await failure
    .locator("#error-view")
    .waitFor({ state: "visible", timeout: 90000 });
  check(
    "runtime failure has retry and no false result",
    (await failure.evaluate(() => !SightLineApp.lastResult)) &&
      (await failure.locator("#btn-retry").isVisible()),
  );
  await shot(failure, "error-reader-390");
  await failureContext.unroute("**/models/ppocrv5-mobile-det.onnx");
  await failure.locator("#btn-retry").click();
  await failure.waitForFunction(
    () => document.body.dataset.screen === "results",
    {},
    { timeout: 120000 },
  );
  check(
    "real reader recovers after runtime failure",
    await failure.evaluate(
      () => SightLineApp.lastResult?.engine.ocr === "ppocrv5",
    ),
  );
  await failureContext.close();
})()
  .catch((error) => {
    check("suite completes", false, String(error.stack || error));
  })
  .finally(async () => {
    fs.writeFileSync(
      path.join(OUT, "ui-check.json"),
      JSON.stringify(
        {
          scope:
            "Real Chrome, synthetic documents, test camera; speech transport stub only for utterance assertion",
          checks,
          records,
          errors,
        },
        null,
        2,
      ),
    );
    await browser?.close();
    server.close();
    console.log(
      JSON.stringify({
        pass: checks.filter((c) => c.ok).length,
        fail: checks.filter((c) => !c.ok).length,
      }),
    );
    process.exitCode = checks.some((c) => !c.ok) ? 1 : 0;
  });
