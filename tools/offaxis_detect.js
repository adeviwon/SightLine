/**
 * offaxis_detect.js — DOES THE APP'S OWN DETECTOR LOSE THE off_axis LINES, AND
 * WHICH FIX GETS THEM BACK WITHOUT BREAKING THE PROFILES THAT WORK?
 *
 * THE GAP
 * -------
 * End-to-end word accuracy is 35.0%, but the `off_axis` profile sits at 1.1%
 * word accuracy / CER 0.882. The cause is the DETECTOR, not the model:
 * legal/off_axis yields 2 detected bands where the same document yields 15 in
 * every other profile. ~85% of those lines never reach the recogniser, so no
 * model, policy or validation change can recover the text.
 *
 * `off_axis` is dict(perspective=0.045, rot=4.0) — a perspective warp plus a 4
 * degree rotation. A tilt smears a row-projection: a line occupies extra rows
 * and touches its neighbour, so the valleys close up and the bander sees fewer,
 * taller bands. ml/src/line_detect.py has a `slope_aware` bander (shear the ink
 * projection flat before banding) that was measured REGRESSED and not adopted —
 * but that measurement ran on PROFILES[:4] = studio_clean, handheld_light,
 * handheld_heavy, low_light: a corpus with NO rotated pages. The shear was only
 * ever tested where there is no rotation to correct.
 *
 * WHAT THIS MEASURES, ON THE APP'S OWN CODE
 * -----------------------------------------
 * For every bench page (3 off_axis + the un-rotated bench profiles as a
 * control) it runs app/js/layout.js detectLines() — the SHIPPED detector — over
 * four candidate fixes and compares bands against the DRAWN line count from
 * ml/src/capture.py DOCS (cached at /tmp/sightline_samples/gt_lines.json, the
 * arm_ab.js convention):
 *
 *   unfixed      detectLines(raw source)                    (no fix)
 *   deskew_app   detectLines(preprocess-deskewed source)    (a) what the app
 *                already does — pipeline.js estimateSkew+rotate
 *   deskew_oracle detectLines(rotate(source, +/-4 deg))      (a) the generator's
 *                own rot, i.e. a PERFECT de-rotation
 *   shear_shipped detectLines(source, ink sheared by +slope) (b) line_detect.py's
 *                slope_aware AS IT SHIPS
 *   shear_fixed  detectLines(source, ink sheared by -slope)  (b) the same shear
 *                with the SIGN corrected (see the finding below)
 *   both         detectLines(deskew_oracle, ink sheared)     (c)
 *
 * THE PYTHON HALF (ml/tests/test_offaxis_detect.py) runs the REAL
 * line_detect.variant_bands, calls not re-implements, and writes
 * /tmp/offaxis_gt_boxes.json — exact ground-truth line boxes on the warped
 * off_axis page. This half then recognises GT-box crops against detected-box
 * crops with the app's own recognizer.js (models/crnn.onnx): the PRIZE, which
 * separates "the detector is the only problem" from "detector and model both".
 *
 * Run:
 *   ./.venv/bin/python ml/tests/test_offaxis_detect.py     # first: writes GT boxes
 *   node app/tools/offaxis_detect.js --engine=chromium [--limit=off]
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const pw = require("./pw_core.js")();
const chromeBin = require("./chrome_path.js")();

const APP = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const GT_CACHE = path.join(SAMPLES, "gt_lines.json");
const GT_BOXES = "/tmp/offaxis_gt_boxes.json";
const MAX_DIM = 3200;
const TRUE_ROT = 4.0;          // capture.CAPTURE_PROFILES["off_axis"]["rot"]

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".png": "image/png", ".css": "text/css",
};

/** ground truth: the lines the renderer DREW (arm_ab.js convention). */
function groundTruthLines() {
  const c = JSON.parse(fs.readFileSync(GT_CACHE, "utf8"));
  return { lines: c, source: "ml/src/capture.py DOCS (cached)" };
}

/** serve APP at /, the bench samples at /sample/, the off-manifest control
 *  renders (capture.py __main__'s own dumps) at /extra/. */
function serve() {
  const srv = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split("?")[0]);
    let p;
    if (rel.startsWith("/sample/")) p = path.join(SAMPLES, rel.slice(8));
    else if (rel.startsWith("/extra/")) p = path.join("/tmp", rel.slice(7));
    else p = path.join(APP, rel);
    if (!p.startsWith(APP) && !p.startsWith(SAMPLES) && !p.startsWith("/tmp/")) {
      res.writeHead(403); return res.end("no");
    }
    fs.readFile(p, (e, d) => {
      if (e) { res.writeHead(404); return res.end("not found: " + rel); }
      res.writeHead(200, { "Content-Type": MIME[path.extname(p)] || "application/octet-stream",
                           "Cache-Control": "no-store" });
      res.end(d);
    });
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r(srv)));
}

// ── everything that runs INSIDE the page ─────────────────────────────────────
async function inPage(cfg) {
  try {
    const L = await import(cfg.base + "js/layout.js");
    const R = await import(cfg.base + "js/recognizer.js");
    if (typeof SightLine === "undefined") return { ok: false, error: "SightLine not on the page" };
    await R.init();
    let return_dataUrls = null;

    const toGray = (c) => {
      const w = c.width, h = c.height;
      const d = c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
      const g = new Uint8Array(w * h);
      for (let i = 0, p = 0; i < g.length; i++, p += 4) g[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
      return { gray: g, w, h };
    };
    const mask = (g, w, h) => {
      const S = L.scaleFor(h);
      return L.inkMask(g, w, h, S.median, S.delta);
    };

    // ── the JS mirror of ml/src/line_detect.py _slope_of (least squares dy/dx
    //    over per-column ink centroids, one centroid per occupied column) ──
    function slopeOf(ink, w, h) {
      let nInk = 0, cols = [], cents = [];
      for (let x = 0; x < w; x++) {
        let s = 0, n = 0;
        for (let y = 0; y < h; y++) if (ink[y * w + x]) { s += y; n++; }
        if (n) { cols.push(x); cents.push(s / n); nInk += n; }
      }
      if (nInk < 200 || cols.length < 20) return 0;
      const m = cols.length;
      let mx = 0, my = 0;
      for (let i = 0; i < m; i++) { mx += cols[i]; my += cents[i]; }
      mx /= m; my /= m;
      let cov = 0, var_ = 0;
      for (let i = 0; i < m; i++) { cov += (cols[i] - mx) * (cents[i] - my); var_ += (cols[i] - mx) ** 2; }
      return var_ ? cov / var_ : 0;
    }

    // ── the JS mirror of _shear_proj, returning a 0/1 MASK (detectLines wants
    //    one). dy = round(slope*(x-xc)); column content is moved DOWN by dy. ──
    function shearInk(ink, w, h, slope) {
      const xc = w / 2, out = new Uint8Array(ink.length);
      for (let x = 0; x < w; x++) {
        const d = Math.round(slope * (x - xc));
        if (d === 0) { for (let y = 0; y < h; y++) out[y * w + x] = ink[y * w + x]; }
        else if (d > 0) { for (let y = d; y < h; y++) out[y * w + x] = ink[(y - d) * w + x]; }
        else { for (let y = 0; y < h + d; y++) out[y * w + x] = ink[(y - d) * w + x]; }
      }
      return out;
    }

    // ── rotate a gray page, matching pipeline.js rotate(): canvas
    //    ctx.rotate(-deg*PI/180) about the centre, white outside. Nearest
    //    neighbour — this is a candidate detector fix, not the shipped arm. ──
    function rotateGray(g, w, h, deg) {
      const a = (-deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
      const cx = w / 2, cy = h / 2, out = new Uint8Array(w * h).fill(255);
      for (let Y = 0; Y < h; Y++) {
        const v = Y - cy;
        for (let X = 0; X < w; X++) {
          const u = X - cx;
          const sx = Math.round(cx + u * ca - v * sa);   // inverse map
          const sy = Math.round(cy + u * sa + v * ca);
          if (sx >= 0 && sx < w && sy >= 0 && sy < h) out[Y * w + X] = g[sy * w + sx];
        }
      }
      return out;
    }

    const heights = (bs) => bs.map((b) => b.y1 - b.y0);
    const medOf = (a) => {
      if (!a.length) return 0;
      const s = a.slice().sort((x, y) => x - y);
      return s[Math.floor(s.length / 2)];
    };

    // HOW MANY DRAWN LINES FALL IN EACH DETECTED BAND -> the mechanism:
    // 1 = clean, 2 = a merged pair, >2 = a block, and any drawn line whose
    // centre is in NO band is a fragment/collapse loss.
    function linesPerBand(bands, gtBoxes) {
      if (!gtBoxes) return null;
      const inBand = new Array(gtBoxes.length).fill(-1);
      for (let i = 0; i < gtBoxes.length; i++) {
        const cy = (gtBoxes[i].y0 + gtBoxes[i].y1) / 2;
        for (let j = 0; j < bands.length; j++) {
          if (cy >= bands[j].y0 && cy < bands[j].y1) { inBand[i] = j; break; }
        }
      }
      const per = new Array(bands.length).fill(0);
      for (const b of inBand) if (b >= 0) per[b]++;
      return { per, missed: inBand.filter((b) => b < 0).length };
    }

    const docs = [];
    for (const s of cfg.samples) {
      const img = new Image();
      img.src = s.src;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0);

      const src = toGray(c);
      const inkSrc = mask(src.gray, src.w, src.h);

      // what the APP actually does: preprocess() deskews by estimateSkew.
      const pp = SightLine.preprocess(img, cfg.maxDim, { restoreMode: "none" });
      const dsk = toGray(pp.candidates[0]);
      const inkDsk = mask(dsk.gray, dsk.w, dsk.h);

      const slopeSrc = slopeOf(inkSrc, src.w, src.h);
      const slopeDsk = slopeOf(inkDsk, dsk.w, dsk.h);

      const run = (g, ink) => L.detectLines(g, src.w, src.h, { ink })
        .slice().sort((a, b) => a.y0 - b.y0);

      const variants = {};
      let variants__oracleRot = null;

      // (0) no fix
      variants.unfixed = run(src.gray, inkSrc);

      // (a) deskew: what the app ships, and the generator's own rot as oracle
      variants.deskew_app = (dsk.w === src.w && dsk.h === src.h)
        ? run(dsk.gray, inkDsk) : [];
      const orc = [];
      let gOraGray = null;
      if (s.offAxis) {
        for (const r of [cfg.trueRot, -cfg.trueRot]) {
          const g = rotateGray(src.gray, src.w, src.h, r);
          orc.push({ rot: r, bands: run(g, mask(g, src.w, src.h)) });
        }
        orc.sort((a, b) => b.bands.length - a.bands.length);
        variants.deskew_oracle = orc[0].bands;
        variants__oracleRot = orc.map((o) => o.rot + "deg:" + o.bands.length).join(" ");
        gOraGray = rotateGray(src.gray, src.w, src.h, orc[0].rot);
      }

      // (b) the slope_aware shear, both signs
      variants.shear_shipped = run(src.gray, shearInk(inkSrc, src.w, src.h, +slopeSrc));
      variants.shear_fixed = run(src.gray, shearInk(inkSrc, src.w, src.h, -slopeSrc));

      // (c) both: the best de-rotation, corrected-sign shear on top
      if (variants.deskew_oracle) {
        const g = rotateGray(src.gray, src.w, src.h, orc[0].rot);
        const ink = mask(g, src.w, src.h);
        variants.both = run(g, shearInk(ink, src.w, src.h, -slopeOf(ink, src.w, src.h)));
      }

      // ── the row projection, for the mechanism figure ──────────────────
      const projOf = (ink) => {
        const p = new Float64Array(src.h);
        for (let y = 0; y < src.h; y++) { let n = 0; for (let x = 0; x < src.w; x++) n += ink[y * src.w + x]; p[y] = n / src.w; }
        return p;
      };
      const spanOf = (p) => {
        let lo = -1, hi = -1;
        for (let y = 0; y < p.length; y++) if (p[y] >= 0.04) { if (lo < 0) lo = y; hi = y; }
        return { lo, hi, peak: Math.max.apply(null, Array.from(p)) };
      };
      const pPlain = projOf(inkSrc);
      const pShear = projOf(shearInk(inkSrc, src.w, src.h, -slopeSrc));
      const projInfo = {
        plain: spanOf(pPlain), sheared_fixed: spanOf(pShear),
        slope_src: +slopeSrc.toFixed(4), slope_dsk: +slopeDsk.toFixed(4),
      };

      // WHICH WAY DOES THE APP'S OWN DESKEW ROTATE? Compare preprocess()'s
      // deskewed canvas against this harness rotating the source by +skew and
      // by -skew. The smaller mean abs difference is the app's convention.
      if (s.offAxis && (pp.skew || 0) !== 0) {
        const d = (a, b) => { let s2 = 0; for (let i = 0; i < a.length; i++) s2 += Math.abs(a[i] - b[i]); return s2 / a.length; };
        const rp = rotateGray(src.gray, src.w, src.h, +pp.skew);
        const rm = rotateGray(src.gray, src.w, src.h, -pp.skew);
        projInfo.rot_match = {
          plus: +d(dsk.gray, rp).toFixed(2), minus: +d(dsk.gray, rm).toFixed(2),
          plain: +d(dsk.gray, src.gray).toFixed(2),
          app_dir: d(dsk.gray, rp) < d(dsk.gray, rm) ? "+skew" : "-skew",
        };
      }

      // the figure: plain projection (left half) vs sheared (right half),
      // the DRAWN lines green, the DETECTED bands red (left) / blue (right).
      let figPath = null;
      if (s.offAxis && cfg.figureDir) {
        const cw = 900, ch = src.h;
        const cv = document.createElement("canvas");
        cv.width = cw; cv.height = ch;
        const g2 = cv.getContext("2d");
        g2.fillStyle = "#fff"; g2.fillRect(0, 0, cw, ch);
        const half = cw / 2, mp = Math.max(0.001, projInfo.plain.peak), ms = Math.max(0.001, projInfo.sheared_fixed.peak);
        g2.fillStyle = "#bbb";
        for (let y = 0; y < ch; y++) {
          g2.fillRect(0, y, pPlain[y] / mp * (half - 6), 1);
          g2.fillRect(half + 3, y, pShear[y] / ms * (half - 6), 1);
        }
        const gtB = (cfg.gtBoxes && cfg.gtBoxes[s.doc]) || [];
        g2.fillStyle = "#0a0";
        for (const b of gtB) g2.fillRect(0, b.y0, cw, 1);
        g2.fillStyle = "#d00";
        for (const b of (variants.unfixed || [])) g2.fillRect(0, b.y0, half, 1);
        g2.fillStyle = "#00d";
        for (const b of (variants.shear_fixed || [])) g2.fillRect(half, b.y0, half, 1);
        g2.fillStyle = "#000"; g2.font = "13px monospace";
        g2.fillText("row projection, off_axis:  LEFT = plain (red = detected bands)", 6, 14);
        g2.fillText("RIGHT = slope_aware shear, corrected sign (blue = detected)", half + 6, 14);
        g2.fillText("every green line = one DRAWN line (ground truth)", 6, 30);
        figPath = cfg.figureDir + "/offaxis_projection_" + s.doc + ".png";
        return_dataUrls = return_dataUrls || {};
        return_dataUrls[figPath] = cv.toDataURL("image/png");
      }

      // ── band geometry + the mechanism histogram ───────────────────────
      const gtBoxes = (cfg.gtBoxes && cfg.gtBoxes[s.doc]) || null;
      const rec = { name: s.name, doc: s.doc, profile: s.profile,
                    w: src.w, h: src.h, offAxis: !!s.offAxis,
                    appSkew: +((pp.skew || 0)).toFixed(3),
                    bands: {}, heights: {}, linesPerBand: {}, missed: {},
                    projInfo };
      for (const k of Object.keys(variants)) {
        const bs = variants[k] || [];
        rec.bands[k] = bs.length;
        rec.heights[k] = { med: medOf(heights(bs)), max: Math.max(0, ...heights(bs)) };
        const lpb = linesPerBand(bs, gtBoxes);
        if (lpb) { rec.linesPerBand[k] = lpb.per; rec.missed[k] = lpb.missed; }
      }
      if (variants__oracleRot) rec.deskew_oracle_rot = variants__oracleRot;
      if (figPath) rec.fig = figPath;

      // ── the PRIZE: the recogniser on GT boxes vs detected boxes ────────
      if (s.recognise) {
        const crop = (g, w, box) => {
          const cr = L.cropForRecognizer(g, w, box, R.CROP_H, R.EXPORT_W);
          return { gray: cr.data, width: cr.width };
        };
        const texts = async (boxes, g) => {
          const out = await R.recogniseAll(boxes.map((b) => crop(g, src.w, b)));
          return out.map((r) => r.text);
        };
        const pack = async (boxes, key, g) => {
          if (!boxes.length) return null;
          const t = await texts(boxes, g || src.gray);
          return { key, n: boxes.length, texts: t,
                   bh: boxes.map((b) => b.y1 - b.y0) };
        };
        rec.prize = {
          gt: cfg.gtBoxes ? await pack(cfg.gtBoxes[s.doc] || [], "gt") : null,
          detected: await pack(variants.unfixed, "detected"),
          derot_detected: gOraGray ? await pack(variants.deskew_oracle, "derot_detected", gOraGray) : null,
        };
      }
      docs.push(rec);
    }
    return { ok: true, docs, figures: return_dataUrls };
  } catch (e) {
    const msg = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : String(e));
    return { ok: false, error: msg,
             stack: e && e.stack ? String(e.stack).split("\n").slice(0, 4).join(" | ") : null };
  }
}

// ── metrics (node side, on the returned crop texts) ──────────────────────────
const norm = (s) => String(s == null ? "" : s).toLowerCase().replace(/[^a-z0-9]/g, "");
function lev(a, b) {
  const n = a.length, m = b.length;
  if (!n) return m; if (!m) return n;
  let prev = new Array(m + 1), cur = new Array(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;
  for (let i = 1; i <= n; i++) {
    cur[0] = i; const ai = a.charCodeAt(i - 1);
    for (let j = 1; j <= m; j++) {
      const cost = ai === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev; prev = cur; cur = t;
  }
  return prev[m];
}
const words = (s) => String(s || "").split(/\s+/).filter(Boolean).map(norm).filter(Boolean);
function wordAcc(gt, hyp) {
  const g = words(gt);
  if (!g.length) return 0;
  const h = new Set(words(hyp));
  return g.filter((w) => h.has(w)).length / g.length;
}

const VKEYS = ["unfixed", "deskew_app", "deskew_oracle", "shear_shipped", "shear_fixed", "both"];

function main(docs, gt, outPath) {
  const truth = (d) => (gt.lines[d.doc] || []).length;
  const tag = (d) => d.doc + "/" + d.profile;

  console.log("\n=== 1. DETECTED BANDS vs DRAWN LINES  (app/js/layout.js detectLines) ===");
  console.log("   truth = lines capture.py DREW (gt_lines.json); every band count below is bands, not lines.");
  console.log("\n   OFF_AXIS (the profile that fails):");
  const hdr = "     " + "page".padEnd(30) + "truth" + VKEYS.map((k) => k.slice(0, 7).padStart(9)).join("");
  console.log(hdr);
  for (const d of docs.filter((x) => x.offAxis)) {
    console.log("     " + tag(d).padEnd(30) + String(truth(d)).padStart(5)
      + VKEYS.map((k) => String(d.bands[k] == null ? "-" : d.bands[k]).padStart(9)).join("")
      + "      appSkew " + d.appSkew + "deg  oracle " + (d.deskew_oracle_rot || "-"));
  }
  console.log("\n   CONTROL profiles (un-rotated — the shear's original corpus was only these):");
  console.log(hdr);
  for (const d of docs.filter((x) => !x.offAxis)) {
    console.log("     " + tag(d).padEnd(30) + String(truth(d)).padStart(5)
      + VKEYS.map((k) => String(d.bands[k] == null ? "-" : d.bands[k]).padStart(9)).join(""));
  }

  console.log("\n=== 2. BAND HEIGHTS (px) — a merged multi-line block shows up as a TALL band ===");
  console.log("     " + "page".padEnd(30) + VKEYS.map((k) => k.slice(0, 7).padStart(12)).join(""));
  for (const d of docs) {
    console.log("     " + tag(d).padEnd(30) + VKEYS.map((k) => {
      const h = d.heights[k]; return h ? (h.med + "/" + h.max).padStart(12) : "-".padStart(12);
    }).join(""));
  }

  console.log("\n=== 3. MECHANISM — drawn lines per detected band, off_axis ===");
  for (const d of docs.filter((x) => x.offAxis)) {
    console.log("   " + tag(d) + "  (" + truth(d) + " drawn lines)");
    for (const k of VKEYS) {
      const per = d.linesPerBand[k];
      if (!per || d.bands[k] == null) continue;
      console.log("     " + k.padEnd(14) + "bands " + String(d.bands[k]).padStart(2)
        + "  lines/band [" + per.join(",") + "]  lines-missed " + d.missed[k]);
    }
  }

  console.log("\n=== 4. THE ROW PROJECTION (the in-image evidence) ===");
  for (const d of docs.filter((x) => x.offAxis)) {
    const p = d.projInfo;
    console.log("   " + tag(d) + "  slope_src " + p.slope_src + "  slope_after_app_deskew " + p.slope_dsk);
    console.log("     plain      rows>=4% ink: " + p.plain.lo + ".." + p.plain.hi
      + "  span " + Math.max(0, p.plain.hi - p.plain.lo) + "  peak " + p.plain.peak.toFixed(3));
    console.log("     sheared(-) rows>=4% ink: " + p.sheared_fixed.lo + ".." + p.sheared_fixed.hi
      + "  span " + Math.max(0, p.sheared_fixed.hi - p.sheared_fixed.lo) + "  peak " + p.sheared_fixed.peak.toFixed(3));
  }

  // ── DO NOT TRUST SECTION 5's NUMBERS YET ────────────────────────────────────
// Every arm below reports exact 0.0% / lineCER 1.000 / wordAcc 0.0%, INCLUDING the
// ground-truth-box arm -- and CER 1.000 uniformly means the recogniser returned an
// EMPTY string for every crop in every arm. That is not a result about off_axis; it
// is the recogniser pass not receiving the crops, almost certainly the `pack(boxes,
// key, g)` plumbing this file was mid-way through editing when its author stopped.
// Sections 1-4 (detection counts, band heights, the projection and the slope
// measurements) are coherent and are the useful part; section 5 needs re-running
// before anyone quotes it. Separately, the ground-truth arm is degenerate BY
// CONSTRUCTION here: its boxes are axis-aligned over a ~4 degree tilt, so they have
// a 163px median height and contain neighbours -- bh med/max 163/291.
console.log("\n=== 5. THE PRIZE — recogniser on GROUND-TRUTH boxes vs DETECTED boxes, off_axis ===");
  console.log("   (separates 'the detector is the only problem' from 'detector AND model')");
  console.log("   gt            = GT line boxes on the degraded page (perfect identity,");
  console.log("                   but axis-aligned over a 4deg tilt: TALL, neighbours inside)");
  console.log("   detected      = the app's boxes today (unfixed)");
  console.log("   derot_detected = de-rotate the page by the generator's own rot, then");
  console.log("                   the app's detector: 'the page-level fix, then detect'");
  const PKEYS = ["gt", "detected", "derot_detected"];
  const agg = {};
  for (const k of PKEYS) agg[k] = { w: 0, gw: 0, ed: 0, ref: 0, exact: 0, n: 0 };
  for (const d of docs.filter((x) => x.prize)) {
    const gl = gt.lines[d.doc] || [];
    for (const key of PKEYS) {
      const pack = d.prize[key];
      if (!pack) { console.log("   " + tag(d) + "  " + key.padEnd(15) + " NO BOXES"); continue; }
      let exact = 0, ed = 0, ref = 0;
      const n = Math.min(pack.n, gl.length);
      for (let i = 0; i < n; i++) {
        const hyp = norm(pack.texts[i]), r = norm(gl[i]);
        ref += r.length; ed += lev(hyp, r);
        if (hyp === r && r.length) exact++;
      }
      const refDoc = gl.join(" "), hypDoc = pack.texts.join(" ");
      const wa = wordAcc(refDoc, hypDoc), gtw = words(refDoc).length;
      console.log("   " + tag(d).padEnd(30) + key.padEnd(16) + "crops " + String(pack.n).padStart(2)
        + "  paired " + String(n).padStart(2)
        + "  exact " + (100 * exact / Math.max(1, n)).toFixed(0).padStart(3) + "%"
        + "  lineCER " + (ed / Math.max(1, ref)).toFixed(3)
        + "  wordAcc " + (100 * wa).toFixed(1).padStart(5) + "%"
        + "  bh med/max " + med(pack.bh) + "/" + Math.max(0, ...pack.bh));
      const A = agg[key];
      A.exact += exact; A.n += n; A.ed += ed; A.ref += ref; A.gw += gtw; A.w += wa * gtw;
    }
  }
  console.log("\n   OFF_AXIS TOTAL:");
  for (const k of PKEYS) {
    const A = agg[k];
    console.log("     " + k.padEnd(16) + "crops " + String(A.n).padStart(2)
      + "  exact " + (100 * A.exact / Math.max(1, A.n)).toFixed(1) + "%"
      + "  lineCER " + (A.ed / Math.max(1, A.ref)).toFixed(3)
      + "  wordAcc " + (100 * A.w / Math.max(1, A.gw)).toFixed(1) + "%");
  }
  console.log("\n   controls: the SAME document under studio_clean reads " +
    "(see arm_ab.js: app documents score ~17-22% word acc overall).");

  const summary = {
    offaxis: docs.filter((x) => x.offAxis).map((d) => ({
      page: tag(d), truth: truth(d), bands: d.bands, heights: d.heights,
      linesPerBand: d.linesPerBand, missed: d.missed, appSkew: d.appSkew,
      deskew_oracle_rot: d.deskew_oracle_rot || null, projInfo: d.projInfo,
      prize_texts: d.prize ? Object.fromEntries(Object.entries(d.prize)
        .filter(([, p]) => p).map(([k, p]) => [k, { n: p.n, texts: p.texts, bh: p.bh }])) : null })),
    control: docs.filter((x) => !x.offAxis).map((d) => ({
      page: tag(d), truth: truth(d), bands: d.bands, heights: d.heights })),
    prize: Object.fromEntries(PKEYS.map((k) => [k, {
      crops: agg[k].n, exact: agg[k].exact, ed: agg[k].ed, ref: agg[k].ref,
      wordsHit: agg[k].w, gtWords: agg[k].gw }])),
  };
  if (outPath) { fs.writeFileSync(outPath, JSON.stringify(summary, null, 1) + "\n"); console.log("\n   wrote " + outPath); }
  return summary;
}

const med = (xs) => { if (!xs.length) return 0; const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

(async () => {
  const engArg = (process.argv.find((a) => a.startsWith("--engine=")) || "").split("=")[1];
  const engine = engArg === "webkit" ? "webkit" : "chromium";
  const outPath = (process.argv.find((a) => a.startsWith("--json=")) || "").split("=")[1];
  const limit = (process.argv.find((a) => a.startsWith("--limit=")) || "").split("=")[1] === "off";

  const manifest = JSON.parse(fs.readFileSync(path.join(SAMPLES, "manifest.json"), "utf8"));
  const gt = groundTruthLines();
  let gtBoxes = null;
  try { gtBoxes = JSON.parse(fs.readFileSync(GT_BOXES, "utf8")); }
  catch (e) { console.log("NOTE: no " + GT_BOXES + " — run ml/tests/test_offaxis_detect.py for the prize row."); }

  let samples = manifest.samples.filter((s) => fs.existsSync(s.file)).map((s) => ({
    name: path.basename(s.file), doc: s.doc, profile: s.profile,
    src: "/sample/" + path.basename(s.file), offAxis: s.profile === "off_axis",
    recognise: s.profile === "off_axis",
  }));
  if (limit) samples = samples.filter((s) => s.offAxis);
  // The shear's original corpus membership: handheld_heavy + low_light, rendered
  // by capture.py __main__ (document index 0 = prescription).
  for (const p of ["handheld_heavy", "low_light"]) {
    const f = "/tmp/sample_" + p + ".png";
    if (fs.existsSync(f) && !limit) samples.push({ name: path.basename(f), doc: "prescription", profile: p,
      src: "/extra/sample_" + p + ".png", offAxis: false, recognise: false });
  }

  console.log("engine=" + engine + "  fixtures=" + samples.length
    + "  (off_axis " + samples.filter((s) => s.offAxis).length + ", control "
    + samples.filter((s) => !s.offAxis).length + ")"
    + "  model=" + path.join(APP, "models", "crnn.onnx")
    + (gtBoxes ? "  gt_boxes=present" : "  gt_boxes=ABSENT"));

  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port + "/";
  let browser = null;
  try {
    browser = engine === "chromium" ? await pw.chromium.launch({ executablePath: chromeBin })
                                    : await pw.webkit.launch();
    const page = await browser.newPage();
    page.setDefaultTimeout(600000);
    await page.goto(base + "index.html", { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(4000);
    const res = await page.evaluate(inPage, { base, maxDim: MAX_DIM, samples, gtBoxes,
                                              figureDir: "/tmp", trueRot: TRUE_ROT });
    if (!res || !res.ok) {
      console.log("PAGE ERROR: " + ((res && (res.stack || res.error)) || "evaluate returned undefined"));
      process.exitCode = 1; return;
    }
    if (res.figures) {
      for (const [p, url] of Object.entries(res.figures)) {
        fs.writeFileSync(p, Buffer.from(url.split(",")[1], "base64"));
        console.log("  wrote figure " + p);
      }
    }
    const summary = main(res.docs, gt, outPath);

    console.log("\n=== 6. PROSE CHANGE (no source file is modified by this tool) ===");
    const off = summary.offaxis, ctl = summary.control;
    const unfixed = off.map((d) => d.bands.unfixed), fixed = off.map((d) => d.bands.both || d.bands.shear_fixed);
    const brokenControls = ctl.filter((d) => (d.bands.shear_shipped || 0) < d.truth).length;
    console.log("  off_axis bands: unfixed [" + unfixed.join(",") + "] -> best-fix ["
      + fixed.join(",") + "]  (truth " + off.map((d) => d.truth).join(",") + ")");
    console.log("  control pages where the SHIPPED-sign shear loses lines: "
      + brokenControls + "/" + ctl.length);
    console.log("  -> the fix belongs in app/js/layout.js detectLines(), on the ink mask,");
    console.log("     BEFORE the row projection; see the report prose.");

    console.log("\n=== PROJECTION FIGURE ===");
    console.log("  (per-page projInfo in section 4; the PNG dump is written by the report step)");
  } catch (e) {
    console.log("error: " + ((e && e.message) || e));
    process.exitCode = 1;
  } finally {
    if (browser) { try { await browser.close(); } catch (e) { /* gone */ } }
    server.close();
  }
})();
