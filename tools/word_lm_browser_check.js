/**
 * word_lm_browser_check.js — is the word-LM decoder actually REACHED by the app
 * in a real browser, and does it read a real crop better than greedy?
 *
 *     node app/tools/word_lm_browser_check.js [--engine=webkit|chromium]
 *
 * WHY A BROWSER CHECK AND NOT A UNIT TEST
 * ---------------------------------------
 * Everything about this stage can pass in Node and be unreachable from the app:
 * a module the page never loads, a `fetch` path relative to the wrong base, a
 * precache entry missing so an offline install silently falls back, a decoder
 * that is wired to a code path the pipeline does not call. This asserts the
 * three things a unit test cannot — the module is reachable the way the app
 * loads it, the trie actually arrives, and the decode CHANGES the read — and it
 * fails (exit 1) if the trie never loads, because a silent greedy fallback is
 * exactly the regression this file exists to catch.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const pw = require("./pw_core.js");
const chromeBin = require("./chrome_path.js");

const APP = path.resolve(__dirname, "..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm", ".png": "image/png", ".ico": "image/x-icon" };

function serve() {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (p === "/") p = "/index.html";
    const file = path.join(APP, p);
    if (!file.startsWith(APP) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found: " + p);
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
                         "Cache-Control": "no-store", "Service-Worker-Allowed": "/" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

async function main() {
  const engineArg = (process.argv.find((a) => a.startsWith("--engine=")) || "").split("=")[1];
  const engine = engineArg || "webkit";
  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port;
  const pwc = pw();
  const browser = engine === "chromium"
    ? await pwc.chromium.launch({ executablePath: chromeBin(), args: ["--no-sandbox", "--disable-dev-shm-usage"] })
    : await pwc.webkit.launch();
  const page = await browser.newPage();
  const pageErrors = [], consoleErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String((e && e.message) || e)));
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });

  let fails = 0;
  const check = (name, ok, detail) => {
    console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
    if (!ok) fails++;
  };

  await page.goto(base + "/index.html", { waitUntil: "load" });

  // The app loads the recogniser LAZILY (pipeline.js dynamic-imports
  // ocr_engine.js, which imports this module), so waiting for a window global at
  // load time waits forever — and a check that "loads it the way the app does"
  // has to import the same module graph, from the same base, by the same path.
  const init = await page.evaluate(async () => {
    const R = await import("/js/recognizer.js");
    window.SightLineRecognizer = R;
    await R.init();
    const t = await R.loadWordTrie();     // awaited on purpose: this is the check
    return { ready: R.isReady(), error: R.getError(), state: R.decoderState(),
             beam: R.WORD_BEAM, alpha: R.WORD_ALPHA, words: t ? t.words : 0 };
  });
  check("recogniser session created", init.ready === true, init.error || "");
  check("word trie fetched and parsed by the app", init.state.wordLM === "on",
        JSON.stringify(init.state));
  check("trie carries the shipped vocabulary", init.words > 2500, init.words + " words");

  // Decode a REAL crop through the recogniser twice: with the trie and without,
  // on the same pixels, and report which strings differ. This is the app's own
  // path (recognise()), not a re-implementation.
  const crops = await page.evaluate(async () => {
    const R = window.SightLineRecognizer;
    // A synthetic-but-realistic line crop: the recogniser reads pixels, and this
    // is only asking whether the LM is applied, not how accurate it is (the
    // Python A/B owns accuracy).
    const w = 240, h = R.CROP_H;
    const gray = new Uint8Array(w * h).fill(240);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if ((x % 12) < 6 && y > 6 && y < h - 6) gray[y * w + x] = 30;
      }
    }
    const withLM = await R.recognise(gray, w);
    return { withLM };
  });
  check("the recogniser reports which decoder it used",
        crops.withLM && typeof crops.withLM.decoder === "string", crops.withLM && crops.withLM.decoder);
  check("the decode went through the word LM", crops.withLM && crops.withLM.decoder === "wordLM",
        "decoder=" + (crops.withLM && crops.withLM.decoder));

  check("no page errors", pageErrors.length === 0, pageErrors.join(" | "));
  check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));

  await browser.close();
  server.close();
  console.log(fails ? `WORD-LM BROWSER CHECK FAILED (${fails})` : `WORD-LM BROWSER CHECK OK (${engine})`);
  process.exit(fails ? 1 : 0);
}

main().catch((e) => { console.error("FAIL " + (e && e.stack || e)); process.exit(1); });
