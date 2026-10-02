/**
 * Why does the neural restorer fall back to "classical" in the browser?
 *
 * `tools/e2e_ocr.js` reported `engine={"restorer":"classical"}` on every one of
 * 12 samples even with USE_RESTORER=1, and field recovery stayed at 5/12. Two
 * explanations are possible and they demand opposite fixes:
 *
 *   1. the ONNX model is absent or unloadable  -> bundle problem
 *   2. the model loads and inference runs, but the GATE skips it -> gate problem
 *
 * The E2E's try/catch swallowed the distinction, so this probe reports each
 * stage separately: manifest, module presence, init() outcome, status(), and
 * the raw HTTP status of the model file itself.
 *
 * Run:  node tools/restorer_probe.js
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const CDP = require("./cdp_min.js");

const ROOT = path.resolve(__dirname, "..");
const CHROME = process.env.CHROME_BIN
  || "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm",
  ".traineddata": "application/octet-stream", ".gz": "application/octet-stream",
  ".png": "image/png", ".jpg": "image/jpeg",
  ".webmanifest": "application/manifest+json", ".onnx": "application/octet-stream",
};

function serve() {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split("?")[0]);
      const p = path.join(ROOT, rel);
      if (!p.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
      fs.readFile(p, (e, d) => {
        if (e) { res.writeHead(404); res.end("not found: " + rel); return; }
        res.writeHead(200, {
          "Content-Type": MIME[path.extname(p)] || "application/octet-stream",
        });
        res.end(d);
      });
    });
    srv.listen(0, () => resolve(srv));
  });
}

(async () => {
  const server = await serve();
  const port = server.address().port;
  const chrome = spawn(CHROME, ["--headless=new", "--no-sandbox", "--disable-gpu",
    "--remote-debugging-port=9345", "about:blank"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 3500));

  try {
    const cdp = await CDP.attach(9345);
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${port}/index.html` });
    await new Promise((r) => setTimeout(r, 5000));

    const ev = async (expr) => {
      const o = await cdp.eval(expr, 300000);
      return o;
    };

    const manifest = await ev("fetch('models/ort.json').then(r=>r.text()).catch(e=>'FETCH FAIL: '+e)");
    console.log("1. manifest      :", manifest);

    console.log("2. modules       : restorer=" + (await ev("typeof SightLineRestorer"))
      + " gating=" + (await ev("typeof SightLineGating"))
      + " pipeline=" + (await ev("typeof SightLine")));

    const modelHttp = await ev("fetch('models/restorer.onnx').then(r=>'HTTP '+r.status+' len='+r.headers.get('content-length')).catch(e=>'FAIL: '+e)");
    console.log("3. model over HTTP:", modelHttp);

    const dataHttp = await ev("fetch('models/restorer.onnx.data').then(r=>'HTTP '+r.status+' len='+r.headers.get('content-length')).catch(e=>'FAIL: '+e)");
    console.log("4. weight blob HTTP:", dataHttp);

    const ortHttp = await ev("fetch('vendor/ort/ort.min.mjs').then(r=>'HTTP '+r.status+' len='+r.headers.get('content-length')).catch(e=>'FAIL: '+e)");
    console.log("5. ort module HTTP:", ortHttp);

    const init = await ev(`(async()=>{
      try {
        const s = await SightLineRestorer.init();
        return s ? 'SESSION CREATED' : 'NULL (init returned null)';
      } catch(e) { return 'THREW: ' + ((e && e.message) || e); }
    })()`);
    console.log("6. init()        :", init);

    console.log("7. status()      :", await ev("JSON.stringify(SightLineRestorer.status())"));

    // The decisive test: run restore() on a synthetic tile and see which branch
    // it takes. engine "gate-skipped" means the model loaded and the gate said
    // "clean"; engine "onnx" means inference ran; "classical" means the model
    // was never available.
    const restore = await ev(`(async()=>{
      const c = document.createElement('canvas');
      c.width = 900; c.height = 580;
      const x = c.getContext('2d');
      x.fillStyle = '#fff'; x.fillRect(0,0,900,580);
      x.fillStyle = '#000';
      x.font = '28px sans-serif';
      for (let i=0;i<12;i++) x.fillText('Account Number: 40218877 Sort Code: 40-11-04', 60, 70+i*38);
      try {
        const r = await SightLineRestorer.restore(c, {});
        return JSON.stringify({engine: r.engine, weight: r.weight, error: r.error, tiles: r.tiles});
      } catch(e) { return 'THREW: ' + ((e && e.message) || e); }
    })()`);
    console.log("8. restore()     :", restore);
    console.log("   (engine=onnx -> inference ran | gate-skipped -> model loaded, "
      + "gate said clean | classical -> model unavailable)");

  } catch (e) {
    console.log("probe error:", e && e.message);
  } finally {
    chrome.kill();
    server.close();
    process.exit(0);
  }
})();
