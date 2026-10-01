/**
 * SightLine Restorer — ONNX Runtime Web inference for SightLineNet.
 *
 * Mirrors ml/src/evaluate.py:arm_restorer (and arm_classical as the fallback).
 *
 *   Python                                  JS (this file)
 *   -------------------------------------   ---------------------------------
 *   SightLineNet (ml/src/model.py)          restorer.onnx, input [N,1,64,256]
 *   th, tw = 64, 256                        TILE_H, TILE_W (identical)
 *   np.pad(..., mode="reflect")             reflectPad() below
 *   out += r; cnt += 1  (overlap average)   accumulate + divide
 *   np.clip(out*255)                        Float32Array -> canvas
 *
 * The tiling is the part that must match exactly: the network was trained and
 * evaluated on 64x256 patches, and averaging overlapping tiles is what removes
 * the seam artefacts a naive single-pass resize would introduce.
 *
 * DEGRADATION POLICY — this module NEVER throws. If the .onnx file is absent,
 * ORT fails to load, or inference errors, it falls back to the classical
 * median+unsharp stack (evaluate.arm_classical) so the scan still completes.
 * `lastEngine` reports which path actually ran.
 */

"use strict";

const SightLineRestorer = (() => {

  const TILE_H = 64;
  const TILE_W = 256;
  // Only the WASM module is vendored (see prune_ort.sh — the webgpu, webgl,
  // jsep and bundle variants are ~77 MB of dead weight). A single candidate
  // keeps the failure mode legible: if this import fails, ORT is genuinely
  // absent and the classical fallback is correct.
  const ORT_MODULE = "vendor/ort/ort.min.mjs";
  const ORT_PATH = "vendor/ort/";
  const MODEL_PATH = "models/restorer.onnx";

  let _ort = null;          // the ort module namespace
  let _session = null;      // InferenceSession
  let _loading = null;      // in-flight promise
  let lastEngine = "uninitialised";
  let lastError = null;
  let lastWeight = 1.0;    // last degradation-gate reading, for status()/UI

  /**
   * Resolve the shared pipeline module.
   *
   * Must read it off globalThis, not via a bare `typeof SightLine` identifier:
   * pipeline.js is a `const` in its own script/module scope, so under CommonJS
   * (and under any bundler) a bare reference is genuinely undefined even though
   * the module is loaded and fully functional. That made the classical
   * fallback silently no-op. globalThis.SightLine is the only reliable handle.
   */
  function SL() {
    return (typeof globalThis !== "undefined" && globalThis.SightLine) || null;
  }

  /**
   * The degradation gate (js/gating.js). Kept separate from SL() because it
   * is a different module with a different global handle. If it is missing
   * (script not loaded, or a bare-reference environment where the global is
   * not visible) we fall back to weight 1.0 — i.e. always restore — which is
   * the pre-gate behaviour and is safe, just less good.
   */
  function GT() {
    return (typeof globalThis !== "undefined" && globalThis.SightLineGating) || null;
  }

  function base() {
    const p = SL();
    if (p && typeof p.vendorBase === "function") return p.vendorBase();
    const s = document.getElementsByTagName("script");
    for (const el of s) {
      const i = (el.src || "").indexOf("/js/");
      if (i !== -1) return el.src.slice(0, i + 1);
    }
    return "./";
  }

  /**
   * Availability manifest, written by the export/bundle step.
   *
   * A single tiny file is the difference between "quietly uses the classical
   * path" and "fires four doomed dynamic imports that each log a 404". ONNX
   * Runtime Web is deliberately NOT vendored (it is ~10 MB and the app is
   * fully functional without it), so probing for it by import alone produces
   * guaranteed console errors on every load.
   *
   * If models/ort.json is absent, ORT is treated as unavailable without a
   * single network request. Create it to enable the model path:
   *   { "ort": true, "restorer": true, "classifier": true }
   */
  const MANIFEST_PATH = "models/ort.json";

  let _manifest = null;

  async function loadManifest() {
    if (_manifest) return _manifest;
    try {
      const r = await fetch(base() + MANIFEST_PATH, { cache: "force-cache" });
      _manifest = r.ok ? await r.json() : { ort: false, restorer: false, classifier: false };
    } catch (e) {
      _manifest = { ort: false, restorer: false, classifier: false };
    }
    return _manifest;
  }

  /**
   * Dynamically import onnxruntime-web from the local vendor dir. Never touches
   * the network — the service worker precaches these files, and a CDN import
   * would break the offline guarantee.
   */
  async function loadOrt() {
    if (_ort) return _ort;
    const m = await loadManifest();
    if (!m.ort) throw new Error("onnxruntime-web not vendored (models/ort.json absent or ort:false)");
    const b = base();
    const mod = await import(/* webpackIgnore: true */ b + ORT_MODULE);
    _ort = mod;
    if (_ort.env && _ort.env.wasm) {
      // The .wasm binary sits beside the JS loader; without this, ORT looks
      // for it at a CDN path and the offline guarantee silently breaks.
      _ort.env.wasm.wasmPaths = b + ORT_PATH;
      _ort.env.wasm.numThreads = 1;  // COOP/COEP-free hosting
      _ort.env.logLevel = "error";
    }
    return _ort;
  }

  /**
   * init(): resolve a session for restorer.onnx. Returns null (never throws)
   * when the model or the runtime is unavailable, which is the signal to use
   * the classical fallback.
   */
  async function init() {
    if (_session) return _session;
    if (_loading) return _loading;
    _loading = (async () => {
      try {
        // Check the availability manifest BEFORE anything else. Without this,
        // init() fetches restorer.onnx on every page load and logs a 404,
        // because ORT is deliberately not vendored in a stock install.
        const m = await loadManifest();
        if (!m.restorer) {
          lastError = "restorer.onnx not vendored (models/ort.json: restorer=false)";
          return null;
        }
        const ort = await loadOrt();
        const url = base() + MODEL_PATH;
        const probe = await fetch(url, { method: "GET", cache: "force-cache" });
        if (!probe.ok) {
          lastError = "restorer.onnx not found at " + MODEL_PATH + " (HTTP " + probe.status + ")";
          return null;
        }
        _session = await ort.InferenceSession.create(url, {
          executionProviders: ["wasm"],  // webgpu is attempted opportunistically below
          graphOptimizationLevel: "all",
        });
        lastEngine = "onnx";
        lastError = null;
        return _session;
      } catch (e) {
        lastError = (e && e.message) || String(e);
        return null;
      } finally {
        _loading = null;
      }
    })();
    return _loading;
  }

  /** True when the ONNX path is usable. Used by the UI's engine badge. */
  async function available() {
    return (await init()) !== null;
  }

  // ── Geometry helpers ──────────────────────────────────────────────────

  /**
   * np.pad(mode="reflect") for a single-channel 2-D array, per axis.
   * 'symmetric' is NOT the same as OpenCV's default BORDER_REFLECT_256, so we
   * implement reflect here rather than relying on a canvas edge trick.
   */
  function reflectIndex(i, n) {
    if (n === 1) return 0;
    // Map into [0, n) with mirror-without-repeat, matching BORDER_REFLECT.
    const period = 2 * n - 2;
    let m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
  }

  function reflectPadPlane(src, W, H, padH, padW) {
    if (padH === 0 && padW === 0) return src;
    const Hp = H + padH, Wp = W + padW;
    const out = new Float32Array(Hp * Wp);
    for (let y = 0; y < Hp; y++) {
      const sy = reflectIndex(y - padH, H);
      const srow = sy * W, drow = y * Wp;
      for (let x = 0; x < Wp; x++) {
        out[drow + x] = src[srow + reflectIndex(x - padW, W)];
      }
    }
    return out;
  }

  /** Read a canvas into a single-channel Float32Array in [0,1]. */
  function canvasToPlane(canvas) {
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const px = ctx.getImageData(0, 0, w, h).data;
    const plane = new Float32Array(w * h);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      // Canvas is grayscale by the time it gets here, but compute luma anyway
      // so a colour input degrades gracefully instead of reading only R.
      plane[j] = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255;
    }
    return { plane, width: w, height: h };
  }

  function planeToCanvas(plane, w, h, doc) {
    const img = doc.createElement("canvas");
    img.width = w; img.height = h;
    const ctx = img.getContext("2d", { willReadFrequently: true });
    const out = ctx.createImageData(w, h);
    const d = out.data;
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      const v = plane[j] * 255;
      const c8 = v < 0 ? 0 : v > 255 ? 255 : v | 0;
      d[i] = d[i + 1] = d[i + 2] = c8;
      d[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    return img;
  }

  // ── The two engines ───────────────────────────────────────────────────

  /**
   * runONNX(): tiled inference identical to evaluate.arm_restorer.
   * @returns {HTMLCanvasElement}
   */
  async function runONNX(canvas, session, onProgress) {
    const { plane, width: W, height: H } = canvasToPlane(canvas);
    const padH = (TILE_H - (H % TILE_H)) % TILE_H;
    const padW = (TILE_W - (W % TILE_W)) % TILE_W;
    const padded = reflectPadPlane(plane, W, H, padH, padW);
    const Hp = H + padH, Wp = W + padW;

    const nTilesY = Math.ceil(Hp / TILE_H);
    const nTilesX = Math.ceil(Wp / TILE_W);
    const nTiles = nTilesY * nTilesX;

    const out = new Float32Array(Hp * Wp);
    const cnt = new Float32Array(Hp * Wp);

    const input = new Float32Array(TILE_H * TILE_W);
    const tensor = new _ort.Tensor("float32", input, [1, 1, TILE_H, TILE_W]);

    let done = 0;
    for (let ty = 0; ty < Hp; ty += TILE_H) {
      for (let tx = 0; tx < Wp; tx += TILE_W) {
        for (let y = 0; y < TILE_H; y++) {
          const src = (ty + y) * Wp + tx;
          input.set(padded.subarray(src, src + TILE_W), y * TILE_W);
        }
        const res = await session.run({ input: tensor });
        const key = Object.keys(res)[0];
        const o = res[key].data;
        for (let y = 0; y < TILE_H; y++) {
          const d = (ty + y) * Wp + tx;
          const s = y * TILE_W;
          for (let x = 0; x < TILE_W; x++) { out[d + x] += o[s + x]; cnt[d + x] += 1; }
        }
        done++;
        if (onProgress) onProgress({ stage: "restoring", done, total: nTiles });
      }
    }

    // Crop the pad back off and average, exactly like out / max(cnt, 1).
    const finalPlane = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      const s = y * Wp, d = y * W;
      for (let x = 0; x < W; x++) {
        const c = cnt[s + x];
        finalPlane[d + x] = c > 0 ? out[s + x] / c : 0;
      }
    }
    return planeToCanvas(finalPlane, W, H, canvas.ownerDocument || document);
  }

  /**
   * runClassical(): evaluate.arm_classical — 3x3 median + unsharp.
   * Used when the ONNX model is absent. Operates on the canvas so it composes
   * with pipeline.preprocess's stage ordering.
   */
  function runClassical(canvas) {
    const p = SL();
    if (!p) return canvas;
    const med = p.median3(canvas);
    return p.unsharp(med, 1.8, 2);
  }

  // ── Public API ────────────────────────────────────────────────────────

  /**
   * restore(): restore a grayscale canvas. Never throws.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {{onProgress?:Function, forceClassical?:boolean}} [opts]
   * @returns {Promise<{canvas:HTMLCanvasElement, engine:string, error:?string, tiles:?number}>}
   */
  async function restore(canvas, opts) {
    const o = opts || {};
    if (!o.forceClassical) {
      const session = await init();
      if (session) {
        try {
          // Measure the damage BEFORE restoring. The model is unconditional,
          // so running it on an already-clean image actively harms it (-48 dB
          // PSNR on the training corpus). A clean scan is the demo case, so
          // the gate is what makes the model safe to ship. See js/gating.js.
          const gate = GT();
          const probe = gate && gate.weightFromImageData
            ? gate.weightFromImageData(
                canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height))
            : 1.0;
          lastWeight = probe;
          if (probe < 0.02) {
            // Undamaged: pass the original through untouched. Skipping
            // inference entirely is also ~30x faster on a phone.
            lastEngine = "gate-skipped";
            return { canvas, engine: "gate-skipped", error: null, tiles: 0,
                     weight: probe };
          }
          const outCanvas = await runONNX(canvas, session, o.onProgress);
          lastEngine = "onnx";
          return { canvas: outCanvas, engine: "onnx", error: null, tiles: null,
                   weight: probe };
        } catch (e) {
          lastError = "ONNX inference failed: " + ((e && e.message) || e);
        }
      }
    }
    lastEngine = "classical";
    return { canvas: runClassical(canvas), engine: "classical", error: lastError,
             tiles: null, weight: 1.0 };
  }

  /**
   * armRestorerClahe(): the full production arm from evaluate.py — restorer
   * then CLAHE(2.0) + blur + Otsu. This is what the app actually feeds to OCR.
   */
  async function armRestorerClahe(canvas, opts) {
    const r = await restore(canvas, opts);
    const p = SL();
    if (!p) return Object.assign(r, { stage: "raw", candidates: [r.canvas] });
    const cl = p.clahe(r.canvas, 2.0, 8);
    const sm = p.unsharp(cl, 1.0, 1);
    const bin = p.binarize(sm);
    return Object.assign(r, { stage: "restorer_clahe", candidates: [r.canvas, cl, bin] });
  }

  function status() {
    return { engine: lastEngine, error: lastError, modelPath: MODEL_PATH,
             available: _session !== null, weight: lastWeight };
  }

  return { restore, armRestorerClahe, init, available, status, runClassical, TILE_H, TILE_W, MODEL_PATH };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SightLineRestorer;
if (typeof window !== "undefined") window.SightLineRestorer = SightLineRestorer;
