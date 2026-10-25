const fs = require("fs"),
  path = require("path"),
  http = require("http");
const { chromium } = require("./pw_core.js")();
const root = path.resolve(__dirname, "../.."),
  app = path.join(root, "app"),
  out = path.join(root, "artifacts/speech-repair"),
  phase = process.env.SPEECH_PHASE || "before";
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
(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const b = await chromium.launch({
    headless: true,
    args: ["--disable-gpu"],
    executablePath:
      process.env.CHROME_BIN ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  try {
    const c = await b.newContext({ viewport: { width: 390, height: 844 } }),
      p = await c.newPage();
    await p.goto("http://127.0.0.1:" + server.address().port);
    await p.waitForFunction(() => window.SightLineApp);
    const r = await scan(
      p,
      path.join(root, "artifacts/aria-merge/sroie-X00016469612.jpg"),
    );
    const state = await p.evaluate(() => ({
      speechText: SightLineApp.lastResult.speechText,
      visible: !document.querySelector("#btn-speak").hidden,
      note: document.querySelector("#speech-note").textContent,
      voices: speechSynthesis
        .getVoices()
        .map((v) => ({ name: v.name, lang: v.lang, local: v.localService })),
    }));
    fs.writeFileSync(
      path.join(out, phase + ".json"),
      JSON.stringify({ result: r, speech: state }, null, 2),
    );
    await p.screenshot({
      path: path.join(out, phase + "-receipt-390.png"),
      fullPage: true,
    });
    if (phase === "after") {
      await p.evaluate(() => {
        window.__speechAudit = { states: [], utterances: [] };
        TTS.subscribe((s) => __speechAudit.states.push(s));
        const real = speechSynthesis.speak.bind(speechSynthesis);
        speechSynthesis.speak = (u) => {
          __speechAudit.utterances.push({
            text: u.text,
            lang: u.lang,
            voice: u.voice?.name,
            local: u.voice?.localService,
          });
          real(u);
        };
      });
      await p.locator("#btn-speak").click();
      await p.waitForFunction(
        () => ["speaking", "error", "ended"].includes(TTS.getState().status),
        null,
        { timeout: 15000 },
      );
      await p.screenshot({
        path: path.join(out, "after-reading-390.png"),
        fullPage: true,
      });
      await p.waitForFunction(
        () => ["ended", "error"].includes(TTS.getState().status),
        null,
        { timeout: 60000 },
      );
      const first = await p.evaluate(() => ({
        state: TTS.getState(),
        audit: __speechAudit,
      }));
      fs.writeFileSync(
        path.join(out, "actual-speech-events.json"),
        JSON.stringify(first, null, 2),
      );
      if (first.state.status !== "ended")
        throw Error(
          "Actual platform speech did not finish: " + JSON.stringify(first),
        );
      if (first.audit.utterances.some((u) => u.text.includes("9.00")))
        throw Error("Uncertain total reached speech");
      await p.locator("#btn-speak").click();
      await p.locator("#btn-speak").click();
      if ((await p.evaluate(() => TTS.getState().status)) !== "stopped")
        throw Error("Stop did not settle");
      await p.locator("#btn-speak").click();
      await p.waitForFunction(
        () => ["ended", "error"].includes(TTS.getState().status),
        null,
        { timeout: 60000 },
      );
      const replay = await p.evaluate(() => ({state:TTS.getState(), audit:__speechAudit}));
      fs.writeFileSync(path.join(out, "replay-speech-events.json"), JSON.stringify(replay, null, 2));
      if (replay.state.status !== "ended")
        throw Error("Replay failed: " + JSON.stringify(replay.state));
      console.log(
        "PASS real receipt / local platform speech start and end events / Stop / replay; acoustic output unverified",
      );
    }
    if (phase === "after") {
      const checks = [];
      function check(name, ok, detail) {
        checks.push({ name, ok: !!ok, detail });
        console.log((ok ? "PASS " : "FAIL ") + name);
        if (!ok) throw Error(name + " " + JSON.stringify(detail));
      }
      // Real prescription OCR and platform speech; no medicine or voice stub.
      await p.locator("#btn-retake").click();
      const rx = await scan(
        p,
        path.join(
          root,
          "artifacts/redesign/fixtures/synthetic-single-label.png",
        ),
      );
      check(
        "prescription retains complete safe medical text",
        rx.category === "medical" && !!rx.speechText,
      );
      await p.evaluate(() => {
        __speechAudit.utterances = [];
        __speechAudit.states = [];
      });
      await p.locator("#btn-speak").click();
      await p.waitForFunction(
        () => ["speaking", "error", "ended"].includes(TTS.getState().status),
        null,
        { timeout: 15000 },
      );
      check(
        "prescription reaches the local speech engine",
        await p.evaluate(() => TTS.getState().status !== "error"),
      );
      const rxSpeech = await p.evaluate(() => __speechAudit.utterances);
      check(
        "prescription utterance retains medicine and strength",
        rxSpeech.some(
          (u) => /amoxicillin/i.test(u.text) && u.text.includes("500"),
        ),
      );
      await p.locator("#btn-speak").click();
      fs.writeFileSync(
        path.join(out, "prescription-speech.json"),
        JSON.stringify({ result: rx, utterances: rxSpeech }, null, 2),
      );
      // A fresh offline page must have the policy, OCR, checks and local voice.
      await p.waitForFunction(
        () => document.getElementById("badge").textContent === "Ready offline",
      );
      await c.setOffline(true);
      await p.reload();
      await p.waitForFunction(() => window.SightLineApp);
      const offlineReceipt = await scan(
        p,
        path.join(root, "artifacts/aria-merge/sroie-X00016469612.jpg"),
      );
      check(
        "offline reload recognises real receipt",
        offlineReceipt.category === "receipt",
      );
      await p.locator("#btn-speak").click();
      await p.waitForFunction(
        () => ["ended", "error"].includes(TTS.getState().status),
        null,
        { timeout: 60000 },
      );
      check(
        "offline local speech receives real end event",
        await p.evaluate(() => TTS.getState().status === "ended"),
      );
      const stamp = offlineReceipt.timestamp;
      await p.locator("#btn-help").click();
      check(
        "Check Device is directly visible in Help",
        await p.locator("[data-open-device]").first().isVisible(),
      );
      await p.locator("[data-open-device]").first().click();
      const frame = await (
        await p.locator("#device-frame").elementHandle()
      ).contentFrame();
      await frame.waitForFunction(() => window.SightLineDeviceReport, {
        timeout: 20000,
      });
      const previousCheck = await frame.evaluate(
        () => SightLineDeviceReport.checkedAt,
      );
      await frame.locator("#device-recheck").click();
      await frame.waitForFunction(
        (old) => SightLineDeviceReport.checkedAt !== old,
        previousCheck,
      );
      await frame.locator("#test-voice").click();
      await frame.waitForFunction(
        () => ["ended", "error"].includes(TTS.getState().status),
        null,
        { timeout: 20000 },
      );
      check(
        "Device voice test receives a real end event",
        await frame.evaluate(() => TTS.getState().status === "ended"),
      );
      check(
        "voice test asks for human sound confirmation",
        await frame.locator("#voice-confirmation").isVisible(),
      );
      const device = await frame.evaluate(() => SightLineDeviceReport);
      check(
        "device checks recognise complete offline cache and voices",
        device.installed && device.models && device.english,
        device,
      );
      check(
        "checks never claim audio was heard from API presence",
        device.audioVerified === false,
      );
      for (const [width, height] of [
        [320, 568],
        [390, 844],
        [768, 1024],
        [1440, 1000],
      ]) {
        await p.setViewportSize({ width, height });
        check(
          "device layout has no overflow at " + width,
          await frame.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        );
        await frame.evaluate(() =>
          window.scrollTo(0, document.body.scrollHeight),
        );
        const back = await p.locator("#close-device").boundingBox();
        check(
          "Back remains visible after scrolling at " + width,
          back && back.y >= 0 && back.y + back.height <= height,
        );
        await frame.evaluate(() => window.scrollTo(0, 0));
        await p.screenshot({
          path: path.join(out, "after-device-" + width + ".png"),
        });
      }
      await frame.evaluate(
        () => (document.documentElement.style.fontSize = "44px"),
      );
      check(
        "device remains readable at doubled text size",
        await frame.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await frame.evaluate(
        () => (document.documentElement.style.fontSize = ""),
      );
      const AxeBuilder = require(
        process.env.AXE_CORE || "@axe-core/playwright",
      ).default;
      const axe = await new AxeBuilder({ page: p })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      fs.writeFileSync(
        path.join(out, "device-axe.json"),
        JSON.stringify(axe.violations, null, 2),
      );
      check(
        "Device Check passes automated WCAG A/AA",
        !axe.violations.length,
        axe.violations.map((v) => v.id),
      );
      await p.locator("#close-device").click();
      check(
        "return preserves the previous scan",
        await p.evaluate(
          (stamp) =>
            SightLineApp.lastResult.timestamp === stamp &&
            document.body.dataset.screen === "results",
          stamp,
        ),
      );
      await p.setViewportSize({ width: 390, height: 844 });
      await p.locator("#btn-help").click();
      await p.locator("[data-open-device]").first().click();
      await frame.locator("#device-return").click();
      check(
        "Return inside checks also restores reader",
        await p.locator("#results").isVisible(),
      );
      for (const [width, height] of [
        [320, 568],
        [390, 844],
        [768, 1024],
        [1440, 1000],
      ]) {
        await p.setViewportSize({ width, height });
        check(
          "receipt speech control is visible at " + width,
          await p.locator("#btn-speak").isVisible(),
        );
        check(
          "receipt layout has no overflow at " + width,
          await p.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        );
        await p.screenshot({
          path: path.join(out, "after-results-" + width + ".png"),
          fullPage: true,
        });
      }
      for (const [width, height] of [
        [320, 568],
        [390, 844],
        [768, 1024],
        [1440, 1000],
      ]) {
        const fresh = await b.newContext({ viewport: { width, height } }),
          shot = await fresh.newPage();
        await shot.goto("http://127.0.0.1:" + server.address().port);
        await shot.waitForFunction(() => window.SightLineApp);
        await scan(
          shot,
          path.join(root, "artifacts/aria-merge/sroie-X00016469612.jpg"),
        );
        await shot.waitForTimeout(200);
        await shot.screenshot({
          path: path.join(out, "after-results-" + width + ".png"),
          fullPage: true,
        });
        await shot.goto(
          "http://127.0.0.1:" + server.address().port + "/selftest.html",
        );
        await shot.waitForFunction(() => window.SightLineDeviceReport);
        await shot.screenshot({
          path: path.join(out, "after-device-full-" + width + ".png"),
          fullPage: true,
        });
        await fresh.close();
      }
      fs.writeFileSync(
        path.join(out, "journey-checks.json"),
        JSON.stringify(
          { checks, device, physicalAudioVerified: false },
          null,
          2,
        ),
      );
    } else {
      await p.goto(
        "http://127.0.0.1:" + server.address().port + "/selftest.html",
      );
      await p.screenshot({
        path: path.join(out, phase + "-device-390.png"),
        fullPage: true,
      });
    }
    console.log(
      JSON.stringify({
        category: r.category,
        fields: r.fields.length,
        visible: state.visible,
        voices: state.voices.length,
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
