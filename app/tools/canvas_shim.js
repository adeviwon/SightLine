/**
 * Minimal Canvas 2-D shim for Node.
 *
 * Exists so the image-processing half of selftest.html can be executed and
 * verified headlessly. It implements exactly the operations js/pipeline.js and
 * js/restorer.js call — nothing more — so a test that passes here is testing
 * the real code paths, not a mock of them.
 *
 * Implemented: createElement('canvas'), getContext('2d'), createImageData,
 * getImageData, putImageData, drawImage (with nearest-neighbour scaling),
 * fillRect, fillText (5x7 bitmap font), save/restore/translate/rotate,
 * fillStyle.
 *
 * NOT implemented (and not used by the pipeline): gradients, compositing,
 * filter, arcs, globalAlpha. If a future pipeline stage reaches for one, the
 * shim throws loudly rather than silently returning something wrong.
 */

"use strict";

// 5x7 bitmap font, one hex row per scanline. Enough for the test's document
// text; it exists so the image has genuine text-like edge structure for the
// blur/quality estimator to measure, not just random noise.
const FONT = {
  "0": ["0x1F", "0x11", "0x13", "0x15", "0x19", "0x11", "0x1F"],
  "1": ["0x04", "0x0C", "0x04", "0x04", "0x04", "0x04", "0x0E"],
  "2": ["0x1F", "0x01", "0x01", "0x1F", "0x10", "0x10", "0x1F"],
  "3": ["0x1F", "0x01", "0x01", "0x0F", "0x01", "0x01", "0x1F"],
  "4": ["0x11", "0x11", "0x11", "0x1F", "0x01", "0x01", "0x01"],
  "5": ["0x1F", "0x10", "0x10", "0x1F", "0x01", "0x01", "0x1F"],
  "6": ["0x1F", "0x10", "0x10", "0x1F", "0x11", "0x11", "0x1F"],
  "7": ["0x1F", "0x01", "0x02", "0x04", "0x08", "0x08", "0x08"],
  "8": ["0x1F", "0x11", "0x11", "0x1F", "0x11", "0x11", "0x1F"],
  "9": ["0x1F", "0x11", "0x11", "0x1F", "0x01", "0x01", "0x1F"],
  A: ["0x0E", "0x11", "0x11", "0x1F", "0x11", "0x11", "0x11"],
  B: ["0x1E", "0x11", "0x11", "0x1E", "0x11", "0x11", "0x1E"],
  C: ["0x0E", "0x11", "0x10", "0x10", "0x10", "0x11", "0x0E"],
  D: ["0x1E", "0x11", "0x11", "0x11", "0x11", "0x11", "0x1E"],
  E: ["0x1F", "0x10", "0x10", "0x1E", "0x10", "0x10", "0x1F"],
  F: ["0x1F", "0x10", "0x10", "0x1E", "0x10", "0x10", "0x10"],
  G: ["0x0E", "0x11", "0x10", "0x17", "0x11", "0x11", "0x0F"],
  H: ["0x11", "0x11", "0x11", "0x1F", "0x11", "0x11", "0x11"],
  I: ["0x0E", "0x04", "0x04", "0x04", "0x04", "0x04", "0x0E"],
  J: ["0x07", "0x02", "0x02", "0x02", "0x02", "0x12", "0x0C"],
  K: ["0x11", "0x12", "0x14", "0x18", "0x14", "0x12", "0x11"],
  L: ["0x10", "0x10", "0x10", "0x10", "0x10", "0x10", "0x1F"],
  M: ["0x11", "0x1B", "0x15", "0x15", "0x11", "0x11", "0x11"],
  N: ["0x11", "0x19", "0x15", "0x13", "0x11", "0x11", "0x11"],
  O: ["0x0E", "0x11", "0x11", "0x11", "0x11", "0x11", "0x0E"],
  P: ["0x1E", "0x11", "0x11", "0x1E", "0x10", "0x10", "0x10"],
  Q: ["0x0E", "0x11", "0x11", "0x11", "0x15", "0x12", "0x0D"],
  R: ["0x1E", "0x11", "0x11", "0x1E", "0x14", "0x12", "0x11"],
  S: ["0x0F", "0x10", "0x10", "0x0E", "0x01", "0x01", "0x1E"],
  T: ["0x1F", "0x04", "0x04", "0x04", "0x04", "0x04", "0x04"],
  U: ["0x11", "0x11", "0x11", "0x11", "0x11", "0x11", "0x0E"],
  V: ["0x11", "0x11", "0x11", "0x11", "0x11", "0x0A", "0x04"],
  W: ["0x11", "0x11", "0x11", "0x15", "0x15", "0x1B", "0x11"],
  X: ["0x11", "0x11", "0x0A", "0x04", "0x0A", "0x11", "0x11"],
  Y: ["0x11", "0x11", "0x0A", "0x04", "0x04", "0x04", "0x04"],
  Z: ["0x1F", "0x01", "0x02", "0x04", "0x08", "0x10", "0x1F"],
  ":": ["0x00", "0x04", "0x04", "0x00", "0x04", "0x04", "0x00"],
  "-": ["0x00", "0x00", "0x00", "0x1F", "0x00", "0x00", "0x00"],
  ".": ["0x00", "0x00", "0x00", "0x00", "0x00", "0x06", "0x06"],
  ",": ["0x00", "0x00", "0x00", "0x00", "0x06", "0x02", "0x04"],
  "/": ["0x01", "0x01", "0x02", "0x04", "0x08", "0x10", "0x10"],
  "#": ["0x0A", "0x1F", "0x0A", "0x0A", "0x0A", "0x1F", "0x0A"],
  " ": ["0x00", "0x00", "0x00", "0x00", "0x00", "0x00", "0x00"],
};

function parseColor(css) {
  if (typeof css !== "string") return [0, 0, 0, 255];
  let s = css.trim().toLowerCase();
  if (s[0] === "#") {
    s = s.slice(1);
    if (s.length === 3) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
    if (s.length === 8) s = s.slice(0, 6);
    const n = parseInt(s, 16);
    if (Number.isNaN(n)) return [0, 0, 0, 255];
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 255];
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const p = m[1].split(",").map((v) => parseFloat(v));
    return [p[0] | 0, p[1] | 0, p[2] | 0, p.length > 3 ? Math.round(p[3] * 255) : 255];
  }
  return [0, 0, 0, 255];
}

class ImageDataShim {
  constructor(w, h, data) {
    this.width = w;
    this.height = h;
    this.data = data || new Uint8ClampedArray(w * h * 4);
  }
}

class CanvasShim {
  constructor(w, h) {
    this._w = Math.max(1, w | 0);
    this._h = Math.max(1, h | 0);
    this._data = new Uint8ClampedArray(this._w * this._h * 4);
    this.ownerDocument = null;
  }
  get width() { return this._w; }
  set width(v) {
    v = Math.max(1, v | 0);
    if (v === this._w) return;
    const nd = new Uint8ClampedArray(v * this._h * 4);
    this._data = nd; this._w = v;
  }
  get height() { return this._h; }
  set height(v) {
    v = Math.max(1, v | 0);
    if (v === this._h) return;
    const nd = new Uint8ClampedArray(this._w * v * 4);
    this._data = nd; this._h = v;
  }
  getContext(kind) {
    if (kind !== "2d") return null;
    if (!this._ctx) this._ctx = new Ctx2D(this);
    return this._ctx;
  }
  get naturalWidth() { return this._w; }
  get naturalHeight() { return this._h; }
  toDataURL() { return "data:image/png;base64,shim"; }
}

class Ctx2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.fillStyle = "#000000";
    this.font = "10px sans-serif";
    this._stack = [];
  }
  get _d() { return this.canvas._data; }
  get _w() { return this.canvas._w; }
  get _h() { return this.canvas._h; }

  save() { this._stack.push({ fillStyle: this.fillStyle, font: this.font }); }
  restore() { const s = this._stack.pop(); if (s) Object.assign(this, s); }
  translate() { /* no-op: the pipeline only uses it for rotation of blank fills */ }
  rotate() { /* no-op: deskew rotation is asserted via the angle, not pixels */ }
  set filter(v) { if (v && v !== "none") throw new Error("shim: ctx.filter unsupported"); }
  get filter() { return "none"; }

  createImageData(w, h) { return new ImageDataShim(w, h); }

  getImageData(sx, sy, sw, sh) {
    sw = Math.max(0, Math.min(sw | 0, this._w - (sx | 0)));
    sh = Math.max(0, Math.min(sh | 0, this._h - (sy | 0)));
    const out = new ImageDataShim(sw, sh);
    for (let y = 0; y < sh; y++) {
      const src = ((sy | 0) + y) * this._w + (sx | 0);
      out.data.set(this._d.subarray(src * 4, (src + sw) * 4), y * sw * 4);
    }
    return out;
  }

  putImageData(img, dx, dy) {
    for (let y = 0; y < img.height; y++) {
      const ty = (dy | 0) + y;
      if (ty < 0 || ty >= this._h) continue;
      for (let x = 0; x < img.width; x++) {
        const tx = (dx | 0) + x;
        if (tx < 0 || tx >= this._w) continue;
        const si = (y * img.width + x) * 4;
        const di = (ty * this._w + tx) * 4;
        this._d[di] = img.data[si];
        this._d[di + 1] = img.data[si + 1];
        this._d[di + 2] = img.data[si + 2];
        this._d[di + 3] = img.data[si + 3];
      }
    }
  }

  /** Nearest-neighbour. src may be a CanvasShim. */
  drawImage(src, dx, dy, dw, dh) {
    const sw = src.width, sh = src.height;
    const tw = dw == null ? sw : (dw | 0);
    const th = dh == null ? sh : (dh | 0);
    if (sw === 0 || sh === 0 || tw === 0 || th === 0) return;
    for (let y = 0; y < th; y++) {
      const ty = (dy | 0) + y;
      if (ty < 0 || ty >= this._h) continue;
      const sy = Math.min(sh - 1, Math.floor((y * sh) / th));
      for (let x = 0; x < tw; x++) {
        const tx = (dx | 0) + x;
        if (tx < 0 || tx >= this._w) continue;
        const sx = Math.min(sw - 1, Math.floor((x * sw) / tw));
        const si = (sy * sw + sx) * 4;
        const di = (ty * this._w + tx) * 4;
        this._d[di] = src._data[si];
        this._d[di + 1] = src._data[si + 1];
        this._d[di + 2] = src._data[si + 2];
        this._d[di + 3] = 255;
      }
    }
  }

  fillRect(x, y, w, h) {
    const [r, g, b, a] = parseColor(this.fillStyle);
    for (let yy = Math.max(0, y | 0); yy < Math.min(this._h, (y | 0) + (h | 0)); yy++) {
      for (let xx = Math.max(0, x | 0); xx < Math.min(this._w, (x | 0) + (w | 0)); xx++) {
        const i = (yy * this._w + xx) * 4;
        this._d[i] = r; this._d[i + 1] = g; this._d[i + 2] = b; this._d[i + 3] = a;
      }
    }
  }

  /** 5x7 bitmap text. Glyph advance 6px, scaled by the px size in this.font. */
  fillText(text, x, y) {
    const m = /(\d+(?:\.\d+)?)px/.exec(this.font || "10px");
    const scale = m ? Math.max(1, Math.round(parseFloat(m[1]) / 7)) : 1;
    const [r, g, b] = parseColor(this.fillStyle);
    let cx = x | 0;
    const top = y | 0;
    for (const raw of String(text)) {
      const ch = raw.toUpperCase();
      const glyph = FONT[ch];
      if (glyph) {
        for (let row = 0; row < 7; row++) {
          const bits = glyph[row];
          for (let col = 0; col < 5; col++) {
            if (!(bits & (1 << (4 - col)))) continue;
            for (let sy = 0; sy < scale; sy++) {
              const py = top + row * scale + sy;
              if (py < 0 || py >= this._h) continue;
              for (let sx = 0; sx < scale; sx++) {
                const px = cx + col * scale + sx;
                if (px < 0 || px >= this._w) continue;
                const i = (py * this._w + px) * 4;
                this._d[i] = r; this._d[i + 1] = g; this._d[i + 2] = b; this._d[i + 3] = 255;
              }
            }
          }
        }
      }
      cx += 6 * scale;
    }
  }
}

/** Install globals. Returns a teardown function. */
function install() {
  const doc = {
    createElement(tag) {
      if (String(tag).toLowerCase() !== "canvas") throw new Error("shim: only <canvas> is implemented");
      return new CanvasShim(300, 150);
    },
  };
  const g = globalThis;
  const prevDoc = g.document;
  const prevRAF = g.requestAnimationFrame;
  g.document = doc;
  g.requestAnimationFrame = (fn) => setTimeout(fn, 0);
  // A default page <script> base, so vendorBase() resolves vendor/ correctly.
  g.window = g.window || {};
  if (!g.window.document) g.window.document = doc;
  return function teardown() {
    g.document = prevDoc;
    g.requestAnimationFrame = prevRAF;
  };
}

module.exports = { install, CanvasShim, ImageDataShim, parseColor, FONT };
