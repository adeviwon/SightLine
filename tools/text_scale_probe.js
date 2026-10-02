/**
 * IS THE TESSERACT "CEILING" ACTUALLY A TEXT-SIZE PROBLEM?
 *
 * We concluded from tools/psm_sweep.js (336 configurations) that Tesseract
 * cannot read the handheld profiles. That sweep varied SEGMENTATION. It never
 * varied SCALE — and Tesseract's LSTM is notoriously scale-sensitive:
 *
 *   The engine is trained on ~30px x-height. Below ~20px cap height accuracy
 *   falls off a cliff, and no amount of layout analysis recovers glyphs that
 *   were never resolved at the pixel level.
 *
 * Measured on the clean render (blur-independent, so this is the renderer's
 * own choice, not a capture artefact):
 *
 *   text band height   18 px
 *   glyph x-height   ~9 px
 *   Tesseract wants  ~30 px
 *
 * That is a 3.3x deficit, present even on studio_clean -- which OCRs at 100%.
 * The captures are all rendered at the same size, so every degraded profile
 * inherits the same undersized text and then adds blur on top. We may have been
 * measuring a renderer bug and calling it an engine limit.
 *
 * This measures x-height per profile, then tests upscaling factors 1x/2x/3x/4x
 * with Lanczos, reporting field recovery for each. The question is narrow and
 * falsifiable: if 3-4x upscaling recovers handheld_light fields, the ceiling was
 * ours, not Tesseract's.
 *
 * Run:  node tools/text_scale_probe.js
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { CDP } = require("./cdp_min.js");

const ROOT = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const CHROME = process.env.CHROME_BIN
  || "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";
const SCALES = [1, 2, 3, 4];

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".png": "image/png",
  ".jpg": "image/jpeg", ".wasm": "application/wasm", ".gz": "application/gzip",
  ".md": "text/markdown; charset=utf-8", ".ico": "image/x-icon",
  ".onnx": "application/octet-stream", ".traineddata": "application/octet-stream",
};

function serve() {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split("?")[0]);
      let p = rel.startsWith("/sample/")
        ? path.join(SAMPLES, rel.slice("/sample/".length))
        : path.join(ROOT, rel);
      if (!p.startsWith(ROOT) && !p.startsWith(SAMPLES)) { res.writeHead(403); res.end(); return; }
      fs.readFile(p, (e, d) => {
        if (e) { res.writeHead(404, "text/plain"); res.end("404 " + rel); return; }
        res.writeHead(200, {
          "Content-Type": MIME[path.extname(p)] || "application/octet-stream",
          "Cache-Control": "no-store",
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
  const manifest = JSON.parse(fs.readFileSync(path.join(SAMPLES, "manifest.json"), "utf8"));
  const chrome = spawn(CHROME, ["--headless=new", "--no-sandbox", "--disable-gpu",
    "--remote-debugging-port=9381", "about:blank"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 3500));

  try {
    const cdp = await CDP.attach(9381);
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${port}/index.html` });
    await new Promise((r) => setTimeout(r, 5000));
    await cdp.eval("(async()=>{await SightLineRestorer.init();return 1})()", 300000);

    console.log(`Testing upscale factors ${SCALES.join("x, ") + "x"} `
      + `(Lanczos) on ${manifest.samples.length} samples.\n`);
    console.log("Field recovery is all-or-nothing, same rule as evaluate.py.\n");

    // tally[profile][scale] = number of samples with ALL fields recovered
    const tally = {};
    for (const s of manifest.samples) {
      const r = await cdp.eval(`(async()=>{
        const img = new Image();
        img.src = "/sample/" + ${JSON.stringify(path.basename(s.file))};
        await img.decode();
        const P = SightLine;
        const pp = P.preprocess(img, 3200, {restoreMode:"none"});
        const base = (await SightLineRestorer.armRestorerClahe(pp.canvas, {})).candidates[0];

        const upscale = (src, k) => {
          if (k === 1) return src;
          const c = document.createElement("canvas");
          c.width = src.width * k; c.height = src.height * k;
          const x = c.getContext("2d");
          x.imageSmoothingEnabled = true;
          x.imageSmoothingQuality = "high";
          x.drawImage(src, 0, 0, c.width, c.height);
          return c;
        };

        // Measure ink-row band height = text size, blur-independent.
        const measure = (src) => {
          const g = src.getContext("2d", {willReadFrequently:true})
                        .getImageData(0,0,src.width,src.height).data;
          const H = src.height, W = src.width;
          const rowInk = new Uint32Array(H);
          for (let y=0;y<H;y++){
            let n=0;
            for (let x=0;x<W;x++){
              const i=(y*W+x)*4;
              if (g[i]<128) n++;
            }
            rowInk[y]=n;
          }
          const runs=[]; let cur=0;
          for (let y=0;y<H;y++){
            if (rowInk[y]>3) cur++;
            else if (cur){ if(cur>3) runs.push(cur); cur=0; }
          }
          if (cur>3) runs.push(cur);
          runs.sort((a,b)=>a-b);
          return runs.length ? runs[runs.length>>1] : 0;
        };

        const out = [];
        for (const k of ${JSON.stringify(SCALES)}) {
          const big = upscale(base, k);
          const band = measure(base);
          const o = await P.runOCR([big], ()=>{});
          const norm = P.normalizeOCRText(o.text);
          out.push({ k, band: band * k, chars:(o.text||"").length,
                     conf: Math.round((o.confidence||0)*100),
                     found: P.fields_found(norm, ${JSON.stringify(s.doc)}) });
        }
        return JSON.stringify({out, rawW: base.width, rawH: base.height});
      })()`, 1800000);

      const d = JSON.parse(r);
      tally[s.profile] = tally[s.profile] || {};
      for (const o of d.out) {
        tally[s.profile][o.k] = tally[s.profile][o.k] || { fields: 0, n: 0, chars: 0 };
        tally[s.profile][o.k].n++;
        tally[s.profile][o.k].fields += o.found.length;
        tally[s.profile][o.k].chars += o.chars;
      }
      console.log(
        `${(s.doc + "/" + s.profile).padEnd(32)} ${d.rawW}x${d.rawH} `
        + d.out.map((o) => `${o.k}x:${JSON.stringify(o.found)}(${o.band}px,${o.conf}%)`).join(" "));
    }

    console.log("\n" + "=".repeat(74));
    console.log("FIELDS RECOVERED, summed across doc types (higher is better)");
    console.log("=".repeat(74));
    const profiles = Object.keys(tally);
    console.log("profile".padEnd(20) + SCALES.map((k) => `${k}x`.padStart(8)).join(""));
    for (const p of profiles) {
      const row = SCALES.map((k) => String(tally[p][k].fields).padStart(8)).join("");
      console.log(p.padEnd(20) + row);
    }
    console.log("\n(mean chars returned)");
    for (const p of profiles) {
      const row = SCALES.map((k) => String(Math.round(tally[p][k].chars / tally[p][k].n)).padStart(8)).join("");
      console.log(p.padEnd(20) + row);
    }
  } catch (e) {
    console.log("error:", e && e.message);
  } finally {
    chrome.kill();
    server.close();
    process.exit(0);
  }
})();
