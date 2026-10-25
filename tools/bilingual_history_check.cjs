// Real OCR, real IndexedDB and platform speech. Failure controls are explicitly injected.
const fs = require("fs"),
  path = require("path"),
  http = require("http");
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
  const url = new URL(req.url, "http://local"),
    file = path.resolve(
      APP,
      "." + (url.pathname === "/" ? "/index.html" : url.pathname),
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
let checks = [];
function check(name, ok, detail) {
  checks.push({ name, ok: !!ok, detail });
  console.log((ok ? "PASS " : "FAIL ") + name);
  if (!ok) throw Error(name + " " + JSON.stringify(detail));
}
async function scan(p, file) {
  const old = await p.evaluate(() => SightLineApp.lastResult?.timestamp);
  await p.locator("#file-input").setInputFiles(file);
  await p.waitForFunction(
    (old) =>
      SightLineApp.lastResult && SightLineApp.lastResult.timestamp !== old,
    old,
    { timeout: 120000 },
  );
  return p.evaluate(() => SightLineApp.lastResult);
}
async function language(p, value) {
  await p.locator("#visible-language").selectOption(value);
  await p.waitForFunction(
    (value) => document.documentElement.lang === value,
    value,
  );
  await p.waitForTimeout(80);
}
async function speak(p, locale) {
  await p.evaluate(() => {
    window.speechAudit = [];
    const original =
      window.__nativeSpeak ||
      (window.__nativeSpeak = speechSynthesis.speak.bind(speechSynthesis));
    speechSynthesis.speak = (u) => {
      speechAudit.push({
        text: u.text,
        lang: u.lang,
        voice: u.voice?.name,
        voiceLang: u.voice?.lang,
        local: u.voice?.localService,
      });
      original(u);
    };
  });
  await p.locator("#btn-speak").click();
  await p.waitForFunction(
    () => ["ended", "error"].includes(TTS.getState().status),
    null,
    { timeout: 90000 },
  );
  const result = await p.evaluate(() => ({
    state: TTS.getState(),
    utterances: speechAudit,
  }));
  check(
    locale + " actual local platform speech ends",
    result.state.status === "ended",
    result,
  );
  check(
    locale + " correct voice locale",
    result.utterances.length &&
      result.utterances.every(
        (u) =>
          u.local &&
          (locale === "zh-HK"
            ? /^(zh-HK|yue)/i.test(u.voiceLang)
            : /^en/i.test(u.voiceLang)),
      ),
  );
  return result;
}
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + server.address().port;
  const type = process.env.BROWSER === "webkit" ? webkit : chromium;
  const b = await type.launch(
    type === chromium
      ? {
          headless: true,
          executablePath:
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          args: ["--disable-gpu"],
        }
      : { headless: true },
  );
  try {
    const c = await b.newContext({ viewport: { width: 390, height: 844 } }),
      p = await c.newPage();
    let errors = [];
    p.on("pageerror", (e) => errors.push(e.message));
    await p.goto(base);
    await p.waitForFunction(() => window.SightLineApp);
    check(
      "history starts empty with consent off",
      await p.evaluate(
        async () =>
          !document.querySelector("#save-scans").checked &&
          (await SightLineHistory.list()).length === 0,
      ),
    );
    await p.screenshot({
      path: path.join(OUT, "home-en-390.png"),
      fullPage: true,
    });
    const receipt = await scan(
      p,
      path.join(ROOT, "artifacts/aria-merge/sroie-X00016469612.jpg"),
    );
    check(
      "real receipt uses PP-OCRv5",
      receipt.category === "receipt" && receipt.engine.ocr === "ppocrv5",
    );
    check(
      "scan alone does not save sensitive contents",
      await p.evaluate(
        async () => (await SightLineHistory.list()).length === 0,
      ),
    );
    const eng = await speak(p, "en-GB");
    const stamp = receipt.timestamp,
      text = receipt.ocrText,
      recognizedFields = receipt.fields.map((f) => ({
        label: f.label,
        value: f.value,
        confidence: f.confidence,
      }));
    await p.evaluate(() => {
      window.ocrCount = 0;
      const original = SightLine.runOCR;
      SightLine.runOCR = (...args) => {
        ocrCount++;
        return original(...args);
      };
    });
    await language(p, "zh-HK");
    check(
      "language switches without OCR or source changes",
      await p.evaluate(
        ({ stamp, text }) =>
          SightLineApp.lastResult.timestamp === stamp &&
          SightLineApp.lastResult.ocrText === text,
        { stamp, text },
      ),
    );
    check(
      "receipt values localize while original English source and confidence remain unchanged",
      await p.evaluate((source) => {
        const cash = Array.from(document.querySelectorAll(".field-row")).find((row) =>
          row.querySelector("th")?.textContent.includes("付款方式"));
        return (
          !!cash &&
          cash.querySelector(".val").textContent.includes("現金") &&
          cash.querySelector(".field-original")?.textContent.includes("cash") &&
          JSON.stringify(SightLineApp.lastResult.fields.map((f) => ({label:f.label,value:f.value,confidence:f.confidence}))) === JSON.stringify(source) &&
          ocrCount === 0
        );
      }, recognizedFields),
    );
    check(
      "main labels use Traditional Chinese",
      /朗讀/.test(await p.locator("#btn-speak").textContent()) &&
        /紀錄/.test(await p.locator("#btn-history").textContent()),
    );
    const canto = await speak(p, "zh-HK");
    check(
      "Cantonese receipt is generated Chinese, not English with an accent",
      canto.utterances.some(
        (u) => u.text.includes("收據") && u.text.includes("現金"),
      ),
    );
    check(
      "uncertain receipt total stays out of both languages",
      ![...eng.utterances, ...canto.utterances].some((u) =>
        u.text.includes("9.00"),
      ),
    );
    await language(p, "en-GB");
    check(
      "switching back restores the literal English OCR field without rescanning",
      await p.evaluate(() =>
        Array.from(document.querySelectorAll(".field-row")).some((row) =>
          row.querySelector("th")?.textContent === "Payment method" &&
          row.querySelector(".val")?.textContent.trim() === "cash"
        ) && ocrCount === 0,
      ),
    );
    await language(p, "zh-HK");
    await p.screenshot({
      path: path.join(OUT, "receipt-zh-390.png"),
      fullPage: true,
    });
    await p.locator("#btn-save-scan").click();
    check(
      "explicit consent precedes saving",
      await p.locator("#save-consent").isVisible(),
    );
    await p.locator("#consent-save").click();
    await p.waitForFunction(() =>
      /已儲存|Saved on/.test(
        document.getElementById("save-status").textContent,
      ),
    );
    check(
      "successful transaction stores one record without an image",
      await p.evaluate(async () => {
        const rows = await SightLineHistory.list();
        return (
          rows.length === 1 &&
          rows[0].image === null &&
          rows[0].result.ocrText.length > 0
        );
      }),
    );
    await p.reload();
    await p.waitForFunction(() => window.SightLineApp);
    check(
      "language and consent persist across reload",
      await p.evaluate(
        () =>
          document.documentElement.lang === "zh-HK" &&
          document.getElementById("save-scans").checked,
      ),
    );
    await p.locator("#btn-history").click();
    await p.locator(".history-record button").first().waitFor();
    await p.screenshot({
      path: path.join(OUT, "history-zh-390.png"),
      fullPage: true,
    });
    await p.evaluate(() => {
      window.ocrCount = 0;
      const original = SightLine.runOCR;
      SightLine.runOCR = (...args) => {
        ocrCount++;
        return original(...args);
      };
    });
    await p.locator(".history-record button").first().click();
    await p.waitForFunction(
      () =>
        !!SightLineApp.lastResult && document.body.dataset.screen === "results",
    );
    check(
      "history opens saved source with no OCR",
      await p.evaluate(
        (text) => ocrCount === 0 && SightLineApp.lastResult.ocrText === text,
        text,
      ),
    );
    for (const locale of ["en-GB", "zh-HK", "en-GB", "zh-HK"])
      await language(p, locale);
    check(
      "repeated switching never invokes OCR",
      await p.evaluate(() => ocrCount === 0),
    );
    await p.waitForFunction(
      () =>
        /Ready offline|可離線/.test(
          document.getElementById("badge").textContent,
        ),
      null,
      { timeout: 60000 },
    );
    await c.setOffline(true);
    await p.reload();
    await p.waitForFunction(() => window.SightLineApp);
    await p.locator("#btn-history").click();
    await p.locator(".history-record button").first().click();
    await p.waitForFunction(
      () =>
        !!SightLineApp.lastResult && document.body.dataset.screen === "results",
    );
    check(
      "saved result reopens offline",
      await p.evaluate(
        (text) => SightLineApp.lastResult.ocrText === text,
        text,
      ),
    );
    await language(p, "en-GB");
    await speak(p, "en-GB");
    await language(p, "zh-HK");
    await speak(p, "zh-HK");
    await c.setOffline(false);
    await language(p, "en-GB");
    await p.locator("#btn-retake").click();
    const rx = await scan(
      p,
      path.join(ROOT, "artifacts/redesign/fixtures/synthetic-single-label.png"),
    );
    check(
      "prescription retains full safe source",
      rx.category === "medical" && rx.speechText.includes("500"),
    );
    await p.waitForFunction(() =>
      /Saved on/.test(document.getElementById("save-status").textContent),
    );
    const rxEnglish = await speak(p, "en-GB");
    check("English medicine preserves verified source", rxEnglish.utterances.some((u) => /amoxicillin/i.test(u.text)) && rxEnglish.utterances.some((u) => u.text.includes("500")), rxEnglish);
    await language(p, "zh-HK");
    check(
      "prescription fields localize only supported dosage language",
      await p.evaluate(() => {
        const dosage = Array.from(document.querySelectorAll(".field-row")).find((row) =>
          row.querySelector("th")?.textContent === "強度或用量"
        );
        return !dosage || dosage.querySelector(".val").textContent.includes("毫克");
      }),
    );
    const rxSpeech = await speak(p, "zh-HK");
    check(
      "Cantonese medicine keeps original name and safe strength",
      rxSpeech.utterances.some((u) => u.text.includes("Amoxicillin")) &&
        rxSpeech.utterances.some((u) => u.text.includes("五百毫克")),
    );
    fs.writeFileSync(
      path.join(OUT, "actual-speech.json"),
      JSON.stringify(
        {
          receiptEnglish: eng,
          receiptCantonese: canto,
          prescriptionEnglish: rxEnglish,
          prescriptionCantonese: rxSpeech,
          physicalAudioHeard: false,
        },
        null,
        2,
      ),
    );
    await p.locator("#btn-history").click();
    await p.waitForFunction(
      () => document.querySelectorAll(".history-record").length === 2,
    );
    await p.locator(".history-record button").first().click();
    await p.waitForFunction(
      () =>
        !!SightLineApp.lastResult && document.body.dataset.screen === "results",
    );
    check(
      "saved prescription reopens with medical gates",
      await p.evaluate(
        () =>
          SightLineApp.lastResult.category === "medical" &&
          !!SightLineApp.lastResult.speechText,
      ),
    );
    // Explicit quota/security controls; ordinary scan remains usable and no Saved is claimed.
    await language(p, "en-GB");
    await p.evaluate(() => {
      window.realSave = SightLineHistory.save;
      SightLineHistory.save = async () => {
        throw new DOMException("full", "QuotaExceededError");
      };
    });
    await p.locator("#btn-save-scan").click();
    await p.waitForFunction(() =>
      document
        .getElementById("save-status")
        .textContent.includes("Storage is full"),
    );
    check(
      "quota failure is honest and current result survives",
      await p.evaluate(
        () =>
          !!SightLineApp.lastResult &&
          !document
            .getElementById("save-status")
            .textContent.startsWith("Saved"),
      ),
    );
    await p.evaluate(() => (SightLineHistory.save = realSave));
    await p.locator("#btn-history").click();
    await p.locator(".history-record .text-button").first().click();
    await p.locator("#confirm-delete").click();
    await p.waitForFunction(
      () => document.querySelectorAll(".history-record").length === 1,
    );
    check(
      "single delete commits",
      await p.evaluate(
        async () => (await SightLineHistory.list()).length === 1,
      ),
    );
    await p.locator("#history-clear").click();
    check(
      "bulk deletion requires confirmation",
      await p.locator("#history-delete-dialog").isVisible(),
    );
    await p.locator("#confirm-delete").click();
    await p.waitForFunction(
      () => !document.getElementById("history-empty").hidden,
    );
    check(
      "delete all commits and disables automatic saving",
      await p.evaluate(
        async () =>
          (await SightLineHistory.list()).length === 0 &&
          !document.getElementById("save-scans").checked,
      ),
    );
    for (const [width, height] of [
      [320, 568],
      [390, 844],
      [768, 1024],
      [1440, 1000],
    ]) {
      await p.setViewportSize({ width, height });
      for (const locale of ["en-GB", "zh-HK"]) {
        await language(p, locale);
        check(
          "history no overflow " + width + " " + locale,
          await p.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        );
        await p.screenshot({
          path: path.join(OUT, `history-empty-${locale}-${width}.png`),
          fullPage: true,
        });
      }
    }
    check("no unexplained page errors", !errors.length, errors);
    fs.writeFileSync(
      path.join(OUT, "checks.json"),
      JSON.stringify({ checks, errors, physicalAudioHeard: false }, null, 2),
    );
    console.log(
      JSON.stringify({
        pass: checks.filter((x) => x.ok).length,
        fail: checks.filter((x) => !x.ok).length,
      }),
    );
    await c.close();
  } finally {
    await b.close();
    server.close();
  }
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
