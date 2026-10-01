/**
 * SightLine Classifier — ONNX Runtime Web inference for the MiniLM classifier.
 *
 * Mirrors ml/src/train_classifier.py + ml/src/export_onnx.py.
 *
 *   Python                                        JS (this file)
 *   -------------------------------------------    -----------------------------
 *   all-MiniLM-L6-v2 encoder (frozen)              minilm_encoder.onnx
 *   mean-pool over attention_mask, then L2-norm    meanPoolL2() below
 *   ClassifierHead Linear(384→128)+ReLU+            minilm_head.onnx
 *     Dropout+Linear(128→4)
 *   CATEGORIES = [banking, medical, legal,          CATEGORIES below
 *     general]                                     (order is the index map)
 *   F.softmax(logits, 1)                           softmax()
 *
 * Two models, two sessions. The encoder dominates cost (22.7M params, 384-dim
 * mean-pooled output); the head is 49.8K params and effectively free.
 *
 * DEGRADATION POLICY — never throws. If the encoder, the head, the tokenizer or
 * the runtime is missing, `classify()` returns null and the caller uses
 * pipeline.js's keyword classifier. A null result is NOT an error — the app is
 * fully functional without any of these files present.
 */

"use strict";

const SightLineClassifier = (() => {

  /** MUST match corpus.CATEGORIES and the head's output index order. */
  const CATEGORIES = ["banking", "medical", "legal", "general"];

  const ENCODER_PATH = "models/minilm_encoder.onnx";
  const HEAD_PATH = "models/minilm_head.onnx";
  const TOKENIZER_PATH = "models/tokenizer.json";
  const MAX_TOKENS = 256;   // matches embed_corpus(max_length=256)

  // Only the WASM module is vendored (see prune_ort.sh — the webgpu, webgl,
  // jsep and bundle variants are ~77 MB of dead weight for a CPU-only PWA).
  const ORT_MODULE = "vendor/ort/ort.min.mjs";
  const ORT_PATH = "vendor/ort/";

  let _ort = null;
  let _enc = null;
  let _head = null;
  let _tokenizer = null;
  let _booting = null;
  let lastEngine = "uninitialised";
  let lastError = null;

  /**
   * Resolve the shared pipeline module.
   *
   * Read off globalThis, not via a bare `typeof SightLine` identifier: pipeline.js
   * is a `const` in its own script/module scope, so under CommonJS (or any
   * bundler) a bare reference is genuinely undefined even though the module is
   * loaded. That would have made the keyword fallback silently unreachable.
   */
  function SL() {
    return (typeof globalThis !== "undefined" && globalThis.SightLine) || null;
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
   * Availability manifest, written by the export/bundle step. See the identical
   * helper in restorer.js for the rationale: probing for onnxruntime-web by
   * dynamic import alone logs guaranteed 404s on every load, because ORT is
   * deliberately not vendored.
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
      _ort.env.wasm.numThreads = 1;
      _ort.env.logLevel = "error";
    }
    return _ort;
  }

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: "force-cache" });
    if (!r.ok) throw new Error("HTTP " + r.status + " for " + url);
    return r.json();
  }

  /**
   * loadTokenizer(): wordpiece tokenizer built from tokenizer.json.
   *
   * We implement the BERT wordpiece algorithm directly (vocab lookup ->
   * greedy longest-match-first subword -> [CLS]/[SEP]) rather than pulling in
   * @xenova/transformers, which would add ~1MB to the precache for a job that
   * is about 60 lines. Only what the encoder needs: input_ids + attention_mask.
   */
  async function loadTokenizer() {
    if (_tokenizer) return _tokenizer;
    const spec = await fetchJSON(base() + TOKENIZER_PATH);
    const model = spec.model || spec;
    const vocab = model.vocab;
    if (!vocab) throw new Error("tokenizer.json has no model.vocab");
    // tokenizer.json stores vocab as {token: id}; normalize to id->token.
    const idToToken = new Array(Object.keys(vocab).length);
    for (const t of Object.keys(vocab)) idToToken[vocab[t]] = t;
    _tokenizer = {
      idToToken,
      unk: vocab["[UNK]"] != null ? vocab["[UNK]"] : 100,
      cls: vocab["[CLS]"] != null ? vocab["[CLS]"] : 101,
      sep: vocab["[SEP]"] != null ? vocab["[SEP]"] : 102,
      pad: vocab["[PAD]"] != null ? vocab["[PAD]"] : 0,
      lowercase: !!(model.uncased || model.do_lower_case !== false),
    };
    return _tokenizer;
  }

  /** BasicTokenizer: lowercase, strip accents, split on whitespace+punct. */
  function basicTokenize(text) {
    let s = String(text || "");
    if (_tokenizer.lowercase) s = s.toLowerCase();
    s = s.normalize("NFD").replace(/[̀-ͯ]/g, "");
    const out = [];
    for (const piece of s.split(/\s+/)) {
      if (!piece) continue;
      // split punctuation into standalone tokens
      const parts = piece.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu);
      if (parts) for (const p of parts) out.push(p);
    }
    return out;
  }

  /** Wordpiece: greedy longest-match-first, ## continuation. */
  function wordpiece(token) {
    const v = _tokenizer.idToToken;
    const pieces = [];
    let start = 0;
    while (start < token.length) {
      let end = token.length, cur = null;
      while (start < end) {
        const sub = start === 0 ? token.slice(start, end) : "##" + token.slice(start, end);
        const id = lookup(v, sub);
        if (id != null) { cur = id; break; }
        end--;
      }
      if (cur == null) return null;  // cannot segment -> whole token is [UNK]
      pieces.push(cur);
      start = end;
    }
    return pieces;
  }

  function lookup(idToToken, piece) {
    // Linear scan is fine: the model is only used on 1-3 documents per session
    // and a Map gives O(1). Built lazily on first use.
    if (!lookup._map || lookup._len !== idToToken.length) {
      lookup._map = new Map();
      for (let i = 0; i < idToToken.length; i++) {
        if (idToToken[i] != null) lookup._map.set(idToToken[i], i);
      }
      lookup._len = idToToken.length;
    }
    const v = lookup._map.get(piece);
    return v == null ? null : v;
  }

  /**
   * encode(): text -> {inputIds: Int32Array, attentionMask: Int32Array, seq}.
   * Single sequence, batch dim 1 — the app classifies one document at a time.
   */
  function encode(text) {
    if (!_tokenizer) throw new Error("tokenizer not loaded");
    const ids = [_tokenizer.cls];
    for (const tok of basicTokenize(text)) {
      const pieces = wordpiece(tok);
      if (pieces) {
        for (const p of pieces) {
          if (ids.length >= MAX_TOKENS - 1) break;
          ids.push(p);
        }
      } else {
        if (ids.length < MAX_TOKENS - 1) ids.push(_tokenizer.unk);
      }
      if (ids.length >= MAX_TOKENS - 1) break;
    }
    ids.push(_tokenizer.sep);
    const n = ids.length;
    const inputIds = new Int32Array(n);
    const attentionMask = new Int32Array(n);
    for (let i = 0; i < n; i++) { inputIds[i] = ids[i]; attentionMask[i] = 1; }
    return { inputIds, attentionMask, seq: n };
  }

  /**
   * meanPoolL2(): (h * mask).sum(1) / mask.sum(1), then L2-normalise.
   * Identical to embed_corpus() in train_classifier.py. h is [seq, 384].
   */
  function meanPoolL2(h, mask, seq, dim) {
    const emb = new Float32Array(dim);
    for (let t = 0; t < seq; t++) {
      const m = mask[t];
      if (!m) continue;
      for (let d = 0; d < dim; d++) emb[d] += h[t * dim + d] * m;
    }
    let count = 0;
    for (let t = 0; t < seq; t++) count += mask[t];
    if (count < 1e-9) count = 1e-9;
    let norm = 0;
    for (let d = 0; d < dim; d++) { emb[d] /= count; norm += emb[d] * emb[d]; }
    norm = Math.sqrt(norm);
    if (norm > 1e-12) for (let d = 0; d < dim; d++) emb[d] /= norm;
    return emb;
  }

  /** Numerically-stable softmax over 4 logits. */
  function softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    const probs = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { probs[i] = Math.exp(logits[i] - max); sum += probs[i]; }
    if (sum > 0) for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    return probs;
  }

  /**
   * init(): load tokenizer + both sessions. Returns false (never throws) when
   * any piece is missing, which puts the app on the keyword fallback.
   */
  async function init() {
    if (_enc && _head && _tokenizer) return true;
    if (_booting) return _booting;
    _booting = (async () => {
      try {
        const m = await loadManifest();
        if (!m.classifier) {
          lastError = "MiniLM ONNX artifacts not vendored (models/ort.json: classifier=false)";
          return false;
        }
        const ort = await loadOrt();
        await loadTokenizer();
        const encUrl = base() + ENCODER_PATH;
        const headUrl = base() + HEAD_PATH;
        const [a, b] = await Promise.all([
          fetch(encUrl, { cache: "force-cache" }),
          fetch(headUrl, { cache: "force-cache" }),
        ]);
        if (!a.ok) throw new Error("encoder missing (HTTP " + a.status + ")");
        if (!b.ok) throw new Error("head missing (HTTP " + b.status + ")");
        const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
        _enc = await ort.InferenceSession.create(encUrl, opts);
        _head = await ort.InferenceSession.create(headUrl, opts);
        lastEngine = "onnx";
        lastError = null;
        return true;
      } catch (e) {
        lastError = (e && e.message) || String(e);
        _enc = null; _head = null; _tokenizer = null;
        return false;
      } finally {
        _booting = null;
      }
    })();
    return _booting;
  }

  async function available() {
    return (await init()) === true;
  }

  /**
   * classify(): run the model. Returns
   *   {category, confidence, probs:{banking,medical,legal,general}, engine:"onnx"}
   * or null when unavailable — the caller then uses pipeline.classify.
   */
  async function classify(text, opts) {
    const o = opts || {};
    if (!o.force) {
      const ok = await init();
      if (!ok) { lastEngine = "keywords"; return null; }
    }
    try {
      const { inputIds, attentionMask, seq } = encode(text);
      const idT = new _ort.Tensor("int64", BigInt64Array.from(inputIds, (v) => BigInt(v)), [1, seq]);
      const maskT = new _ort.Tensor("int64", BigInt64Array.from(attentionMask, (v) => BigInt(v)), [1, seq]);

      const encOut = await _enc.run({ input_ids: idT, attention_mask: maskT });
      const hKey = Object.keys(encOut).find((k) => /last_hidden|output|hidden/i.test(k)) || Object.keys(encOut)[0];
      const hidden = encOut[hKey];              // [1, seq, 384]
      const dims = hidden.dims || [1, seq, 384];
      const dim = dims[2] || 384;
      const emb = meanPoolL2(hidden.data, attentionMask, seq, dim);

      const headOut = await _head.run({
        embedding: new _ort.Tensor("float32", emb, [1, dim]),
      });
      const lKey = Object.keys(headOut).find((k) => /logit|output/i.test(k)) || Object.keys(headOut)[0];
      const probs = softmax(headOut[lKey].data);

      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      const p = {};
      for (let i = 0; i < CATEGORIES.length; i++) p[CATEGORIES[i]] = probs[i] || 0;
      lastEngine = "onnx";
      return { category: CATEGORIES[best], confidence: probs[best], probs: p, engine: "onnx" };
    } catch (e) {
      lastError = (e && e.message) || String(e);
      lastEngine = "keywords";
      return null;
    }
  }

  /**
   * classifyWithFallback(): always returns something. Prefers the model,
   * falls back to the pipeline keyword classifier. The result is tagged with
   * its source so the UI can be honest about which one ran.
   */
  async function classifyWithFallback(text) {
    const model = await classify(text);
    if (model) return model;
    const p = SL();
    if (!p) return { category: "general", confidence: 0.1, engine: "keywords", probs: null };
    const kw = p.classify(text);
    const kb = p.classify_doc(text);
    const category = kw.category === kb ? kw.category
      : (kw.confidence >= 0.5 ? kw.category : kb);
    return { category, confidence: kw.confidence, engine: "keywords", probs: null };
  }

  function status() {
    return {
      engine: lastEngine,
      error: lastError,
      encoderPath: ENCODER_PATH,
      headPath: HEAD_PATH,
      ready: !!( _enc && _head && _tokenizer ),
    };
  }

  return {
    CATEGORIES, MAX_TOKENS, ENCODER_PATH, HEAD_PATH, TOKENIZER_PATH,
    init, available, classify, classifyWithFallback, status,
    // exported for selftest.html
    softmax, meanPoolL2, basicTokenize, wordpiece, encode,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SightLineClassifier;
if (typeof window !== "undefined") window.SightLineClassifier = SightLineClassifier;
