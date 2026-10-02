/**
 * SightLine — on-device document pipeline (browser port).
 *
 * THIS FILE IS THE FALLBACK PATH. It has zero external dependencies beyond a
 * <canvas> and tesseract.js, and it is the path that MUST always work, offline,
 * on every browser including ones with no WebGPU/WASM-NN support. The ONNX
 * accelerators (restorer.js, classifier.js) are strictly optional upgrades that
 * hand their results back here for the same downstream stages.
 *
 * Python modules mirrored here:
 *   ml/src/evaluate.py  -> norm(), word_accuracy(), fields_found(), classify_doc(),
 *                         FIELDS, ARMS (preprocessing arm names)
 *   ml/src/capture.py   -> PRESCRIPTION / BANK_STATEMENT / LEASE_AGREEMENT text
 *                         (shipped as TEST_VECTORS for selftest.html)
 *   ml/src/corpus.py   -> CATEGORIES = [banking, medical, legal, general]
 *   /tmp/ad/SightLine-Mobile/js/pipeline.js -> the tested preprocess + OCR
 *                         implementation that this port reuses verbatim.
 *
 * Stages: preprocess (canvas) -> tesseract.js multi-pass OCR -> normalize ->
 *         classify -> extract fields -> NER -> summary.
 */

"use strict";

const SightLine = (() => {

  // ── Config ────────────────────────────────────────────────────────────

  /** Class order MUST match corpus.CATEGORIES — the ONNX head's index map. */
  const CATEGORIES = ["banking", "medical", "legal", "general"];

  const CATEGORY_LABELS = {
    banking: "Banking / Financial Document",
    medical: "Medical Prescription",
    legal: "Legal Document",
    general: "General Document",
  };

  /**
   * Ground-truth key fields per document type — mirrors evaluate.FIELDS.
   * A doc counts correct ONLY if all of its fields survive OCR, so this list
   * is deliberately minimal and safety-critical (dosages!).
   */
  const FIELDS = {
    prescription: ["500mg", "400mg"],
    banking: ["40218877", "40-11-04"],
    legal: ["2500", "2024-CV-00456"],
  };

  const DEFAULT_MAX_DIM = 3200;

  // ── evaluate.py metrics (port) ────────────────────────────────────────

  /** norm(): strip everything that is not [a-z0-9], lowercase. */
  function norm(s) {
    return String(s == null ? "" : s).toLowerCase().replace(/[^a-z0-9]/g, "");
  }

  /**
   * word_accuracy(): fraction of ground-truth words that appear in the OCR
   * output after norm(). Mirrors evaluate.word_accuracy exactly — note this is
   * a set-membership hit rate, not a Levenshtein alignment.
   */
  function word_accuracy(gt, hyp) {
    const g = String(gt || "").split(/\s+/).filter(Boolean);
    const h = String(hyp || "").split(/\s+/).filter(Boolean);
    if (!g.length) return 0.0;
    const hset = new Set(h.map(norm));
    let hit = 0;
    for (const w of g) if (hset.has(norm(w))) hit++;
    return hit / g.length;
  }

  /**
   * fields_found(): which ground-truth fields survive in the OCR text.
   * Returns the subset found (all-or-nothing scoring happens in the caller).
   */
  function fields_found(text, docType) {
    const n = norm(text);
    return (FIELDS[docType] || []).filter((f) => n.includes(norm(f)));
  }

  /** All fields of a doc type recovered? Mirrors evaluate.run's `field` metric. */
  function fields_complete(text, docType) {
    const want = FIELDS[docType];
    if (!want) return false;
    return fields_found(text, docType).length === want.length;
  }

  /** exact_dosage_rate numerator for a non-prescription doc. */
  function exact_dosage_rate(text, docType) {
    if (docType !== "prescription") return 1.0;
    return fields_complete(text, docType) ? 1.0 : 0.0;
  }

  /**
   * classify_doc(): keyword-density classifier — the no-model fallback and the
   * sanity check the Python eval uses. Port of evaluate.classify_doc.
   */
  function classify_doc(text) {
    const t = String(text || "").toLowerCase();
    const scores = {
      medical: countAll(t, ["mg", "tablet", "dose", "patient", "prescription", "capsule"]),
      banking: countAll(t, ["balance", "account", "sort code", "iban", "debit", "credit", "gbp", "atm"]),
      legal: countAll(t, ["agreement", "clause", "tenant", "landlord", "court", "deed", "witnesseth"]),
      general: countAll(t, ["notice", "meeting", "library", "parcel", "recipe", "school"]),
    };
    let best = "general", bestN = -1;
    for (const c of CATEGORIES) {
      if (scores[c] > bestN) { bestN = scores[c]; best = c; }
    }
    return bestN > 0 ? best : "general";
  }

  function countAll(haystack, needles) {
    let n = 0;
    for (const k of needles) {
      let i = 0;
      while ((i = haystack.indexOf(k, i)) !== -1) { n++; i += k.length; }
    }
    return n;
  }

  /** Per-category keyword counts — used by the UI to explain the decision. */
  function classify_doc_scores(text) {
    const t = String(text || "").toLowerCase();
    return {
      medical: countAll(t, ["mg", "tablet", "dose", "patient", "prescription", "capsule"]),
      banking: countAll(t, ["balance", "account", "sort code", "iban", "debit", "credit", "gbp", "atm"]),
      legal: countAll(t, ["agreement", "clause", "tenant", "landlord", "court", "deed", "witnesseth"]),
      general: countAll(t, ["notice", "meeting", "library", "parcel", "recipe", "school"]),
    };
  }

  // ── OCR-confusion normalizer (classifier.py fix) ──────────────────────

  /**
   * Deterministic post-fixes for the Tesseract errors that actually destroy
   * structured fields on blurry camera captures:
   *   "m9"       -> "mg"      (9 misread for g)
   *   "m1"       -> "ml"      (1 misread for l)
   *   "S00mg"    -> "500mg"   (leading S before a dose is a misread 5)
   *   "50mg mg"  -> "50mg"    (dupe unit from chained fixes)
   * These are what make dosage extraction survive at 48% image confidence.
   *
   * RULE ORDER IS LOAD-BEARING. The m9/m1 rules must run BEFORE the S rule: the
   * S lookahead requires the token to already end in "mg", so on raw "S00m9"
   * the S rule cannot fire and the output stays "S00mg" — not a dose. Resolving
   * m9 first lets the S rule see "S00mg" and recover "500mg". The reference
   * implementation had these in the opposite order and silently never worked;
   * tools/normalizer_diff.js proves this ordering recovers ground-truth fields
   * (evaluate.FIELDS) with zero regressions across 273 cases.
   *
   * The de-duplication rule anchors on the leading digits and rejoins them
   * ("50mg mg" -> "50mg"). The reference used "\s*\b(mg|ml)\s+\1", whose \b can
   * never match between the "0" and the "m" of "50mg", so it never fired; and
   * simply dropping the \b would emit "50 mg", changing the dose token and
   * breaking fields_found().
   */
  function normalizeOCRText(text) {
    return String(text || "")
      .replace(/m9\b/gi, "mg")
      .replace(/m1\b/gi, "ml")
      .replace(/\bS(?=\d{2,4}(?:m[gq]|m1|1m|mg|ml)?\b)/g, "5")
      .replace(/(\d+)\s*(mg|ml)\s+\2\b/gi, "$1$2");
  }

  // ── Richer classifier (the SightLine-Mobile one, better on live OCR) ───

  const CATEGORY_KEYWORDS = {
    banking: ["account number", "sort code", "balance", "deposit", "withdrawal", "iban",
      "statement", "bank", "hsbc", "barclays", "lloyds", "natwest", "santander",
      "card ending", "direct debit", "standing order", "mortgage", "credit card",
      "transaction", "payment", "£", "$", "eur", "usd"],
    medical: ["prescription", "rx", "dosage", "mg", "ml", "tablet", "capsule", "patient",
      "diagnosis", "medication", "warning", "take one", "twice daily", "three times",
      "daily", "pharmacy", "doctor", "nhs", "drug", "dose", "treatment", "symptoms",
      "keep out of reach", "drowsiness"],
    legal: ["contract", "agreement", "clause", "hereby", "witnesseth", "party", "parties",
      "tenant", "landlord", "lease", "liability", "terms and conditions", "binding",
      "court", "case no", "plaintiff", "defendant", "judgment", "warrant", "affidavit",
      "notarized", "power of attorney", "testament"],
  };

  const DRUGS = ["Amoxicillin", "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine",
    "Ibuprofen", "Paracetamol", "Aspirin", "Omeprazole", "Simvastatin", "Warfarin",
    "Insulin", "Prednisolone", "Sertraline", "Citalopram", "Ramipril", "Bisoprolol",
    "Salbutamol", "Codeine", "Tramadol", "Diclofenac", "Naproxen"];

  /** Keyword-hit classifier. Returns {category, confidence, hits}. */
  function classify(text) {
    const t = String(text || "").toLowerCase();
    let best = { category: "general", confidence: 0.1, hits: 0 };
    for (const cat of Object.keys(CATEGORY_KEYWORDS)) {
      let hits = 0;
      for (const kw of CATEGORY_KEYWORDS[cat]) if (t.includes(kw)) hits++;
      const conf = Math.min(0.95, hits * 0.09 + 0.05);
      if (hits > 0 && conf > best.confidence) best = { category: cat, confidence: conf, hits };
    }
    return best;
  }

  // ── Field extraction ──────────────────────────────────────────────────

  /**
   * Label-prefixed field patterns.
   *
   * Two deliberate divergences from the reference regex layer, both because the
   * reference silently drops a field that evaluate.FIELDS treats as ground
   * truth. Verified against capture.py's own document text:
   *
   *  - Patient: the reference matched the label /patient|name/ case-SENSITIVELY,
   *    but documents capitalise the label ("Patient: Jane Doe"), so it never
   *    fired and the patient name was never reported. The label is now matched
   *    with a character class rather than the /i flag, because /i would also
   *    relax the captured name and turn "Take one capsule" into a person.
   *
   *  - Case Number: the reference allowed only "no|number" after "case", but
   *    capture.py emits "Case Reference: 2024-CV-00456" — a document whose
   *    ground-truth case number the pattern could not extract. "reference|ref"
   *    and a trailing "." are now accepted.
   *
   * Both fixes only widen what is found. They never change a value.
   */
  const PATTERNS = {
    medical: {
      "Medication": /(?:Rx\s*:\s*|medication\s*:\s*|prescription\s+for\s+)([A-Z][a-z]+(?:\s\d{1,4}\s?mg)?)/gi,
      "Dosage": /(\d{1,4}[\dOo]?)\s*(m[gq]|m1|1m|mL|mI|tab1ets?|tablets?|capsul[ce]s?)/gi,
      "Frequency": /(every\s*\d+\s*(?:hours?|hrs)|\d+\s*times?\s*(?:a\s*)?day|once daily|twice daily|three times daily|at bedtime)/gi,
      "Warning": /((?:WARNING|Caution|Do not|Avoid|May cause|Consult|Keep out)[^.\n]+)/gi,
      "Patient": /(?:[Pp]atient|[Nn]ame|[Aa]ccount [Hh]older)\s*[:#]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)/g,
    },
    banking: {
      "Account Number": /account\s*(?:no|number)?\s*[:#]?\s*(\d{6,12})/gi,
      "Sort Code": /sort\s*code\s*[:#]?\s*(\d{2}[-\s]?\d{2}[-\s]?\d{2})/gi,
      "IBAN": /(GB\d{2}\s?[A-Z]{4}\s?[\dA-Z]{4}\s?\d{4}[\dA-Z]?\s?\d{4})/gi,
      "Balance": /(?:balance|available)\s*[:#]?\s*(?:£|\$|€)?\s*([\d,]+\.\d{2})/gi,
      "Card Ending": /(?:card\s+)?ending\s*(?:in\s*)?(\d{4})/gi,
      "Amount": /(£|\$|€)\s*([\d,]+\.\d{2})/gi,
    },
    legal: {
      "Case Number": /(?:case|matter)\s*(?:no|number|reference|ref)?\s*[:#.]?\s*([\w-]+-\d+)/gi,
      "Date": /(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{1,2}\/\d{1,2}\/\d{2,4})/gi,
      "Clause": /(?:clause|section)\s+(\d+(?:\.\d+)?)/gi,
      "Parties": /between\s+([A-Z][\w\s]{2,40}?)\s+and\s+([A-Z][\w\s]{2,40})/gi,
    },
    general: {
      "Date": /(\d{1,2}\/\d{1,2}\/\d{2,4})/g,
      "Phone": /(\+?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4})/g,
      "Email": /([\w.+-]+@[\w-]+\.[\w.-]+)/g,
      "Time": /(\d{1,2}:\d{2}\s*(?:am|pm)?)/gi,
    },
  };

  function extractFields(text, category) {
    const fields = [];
    const pats = PATTERNS[category] || PATTERNS.general;
    for (const label of Object.keys(pats)) {
      const re = pats[label];
      const rx = new RegExp(re.source, re.flags.indexOf("g") === -1 ? re.flags + "g" : re.flags);
      let m, count = 0;
      while ((m = rx.exec(text)) !== null && count < 6) {
        const value = m.slice(1).filter(Boolean).join(" ").trim();
        if (value && value.length > 1) {
          fields.push({ label, value, confidence: 0.88 });
          count++;
        }
        if (m.index === rx.lastIndex) rx.lastIndex++;  // zero-length match guard
      }
    }
    return fields;
  }

  // ── NER ───────────────────────────────────────────────────────────────

  const ORGS = "HSBC|Barclays|Lloyds|NatWest|Santander|NHS|Halifax|TSB|Monzo|Revolut";

  function extractEntities(text) {
    const entities = [];
    const push = (label, value) => {
      if (value && String(value).trim().length > 1) {
        entities.push({ label, text: String(value).trim(), confidence: 0.9 });
      }
    };
    const src = String(text || "");
    for (const drug of DRUGS) {
      if (new RegExp(`\\b${drug}\\b`, "i").test(src)) push("DRUG", drug);
    }
    let m;
    const dosageRe = /\b(\d{1,4}\s*(?:mg|ml|mcg|micrograms?|tablets?|capsules?))\b/gi;
    while ((m = dosageRe.exec(src))) push("DOSAGE", m[1]);
    const moneyRe = /(£|\$|€)\s*([\d,]+\.\d{2})|\b([\d,]+\.\d{2})\s*(?:GBP|USD|EUR)/gi;
    while ((m = moneyRe.exec(src))) push("MONEY", (m[2] ? m[1] + m[2] : m[3] + " " + m[4]).trim());
    const dateRe = /\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b/gi;
    while ((m = dateRe.exec(src))) push("DATE", m[1]);
    const orgRe = new RegExp(`\\b(${ORGS})\\b`, "gi");
    while ((m = orgRe.exec(src))) push("ORG", m[1]);
    const personRe = /(?:patient|name|account holder)\s*[:#]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)/g;
    while ((m = personRe.exec(src))) push("PERSON", m[1]);
    const seen = new Set();
    return entities.filter((e) => {
      const k = e.label + ":" + e.text;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }

  // ── Summary (natural language, fed to speech) ─────────────────────────

  function buildSummary(category, fields) {
    const byLabel = {};
    for (const f of fields || []) (byLabel[f.label] = byLabel[f.label] || []).push(f.value);
    const g = (l) => byLabel[l] || [];
    const parts = [];
    if (category === "medical") {
      parts.push("This is a medical prescription.");
      if (g("Patient").length) parts.push("Patient: " + g("Patient")[0] + ".");
      if (g("Medication").length) parts.push("Medications: " + g("Medication").slice(0, 4).join(", ") + ".");
      if (g("Dosage").length) parts.push("Dosages: " + g("Dosage").slice(0, 4).join(", ") + ".");
      if (g("Frequency").length) parts.push("Frequency: " + g("Frequency").slice(0, 2).join(", ") + ".");
      if (g("Warning").length) parts.push("Warning: " + g("Warning").slice(0, 2).join(" "));
    } else if (category === "banking") {
      parts.push("This is a banking document.");
      if (g("Account Number").length) parts.push("Account number: " + g("Account Number")[0] + ".");
      if (g("Card Ending").length) parts.push("Card ending " + g("Card Ending")[0] + ".");
      if (g("Balance").length) parts.push("Balance: " + g("Balance")[0] + ".");
      if (g("Amount").length) parts.push("Amounts: " + g("Amount").slice(0, 3).join(", ") + ".");
    } else if (category === "legal") {
      parts.push("This is a legal document.");
      if (g("Case Number").length) parts.push("Case reference: " + g("Case Number")[0] + ".");
      if (g("Date").length) parts.push("Date: " + g("Date")[0] + ".");
      if (g("Clause").length) parts.push("Clauses: " + g("Clause").slice(0, 3).join(", ") + ".");
    } else {
      parts.push("This is a document.");
      if (g("Date").length) parts.push("Date: " + g("Date")[0] + ".");
      if (g("Time").length) parts.push("Time: " + g("Time")[0] + ".");
    }
    return parts.join(" ");
  }

  // ── Preprocess: mirrors evaluate.ARMS on canvas ───────────────────────

  function _ctx2d(canvas) {
    return canvas.getContext("2d", { willReadFrequently: true });
  }

  function makeCanvas(w, h) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, w | 0);
    c.height = Math.max(1, h | 0);
    return c;
  }

  /**
   * assessQuality(): Laplacian variance (blur) + mean (dark) + high-variance
   * (salt-pepper noise). Same thresholds as the tested desktop port.
   */
  function assessQuality(grayCanvas) {
    const w = grayCanvas.width, h = grayCanvas.height;
    const scale = Math.max(1, Math.floor(Math.max(w, h) / 1200));
    const sw = Math.max(8, Math.floor(w / scale)), sh = Math.max(8, Math.floor(h / scale));
    const c = makeCanvas(sw, sh);
    const ctx = _ctx2d(c);
    ctx.drawImage(grayCanvas, 0, 0, sw, sh);
    const px = ctx.getImageData(0, 0, sw, sh).data;

    const lum = new Float32Array(sw * sh);
    let sum = 0;
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      const l = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      lum[j] = l; sum += l;
    }
    const mean = sum / lum.length;

    let varSum = 0, n = 0;
    for (let y = 1; y < sh - 1; y++) {
      for (let x = 1; x < sw - 1; x++) {
        const i = y * sw + x;
        const lap = lum[i - 1] + lum[i + 1] + lum[i - sw] + lum[i + sw] - 4 * lum[i];
        varSum += lap * lap; n++;
      }
    }
    const lapVar = n ? varSum / n : 0;
    const blurry = lapVar < 100;
    const dark = mean < 50;
    const noisy = lapVar > 800;
    const warnings = [];
    if (blurry) warnings.push("Image may be blurry. Hold steadier or move closer.");
    if (dark) warnings.push("Image is dark. Try better lighting.");
    return { quality: Math.min(1, lapVar / 500), blurry, dark, noisy, lapVar, mean, warnings };
  }

  /** Otsu threshold from a 256-bin histogram. */
  function otsu(hist, total) {
    let sumTotal = 0;
    for (let t = 0; t < 256; t++) sumTotal += t * hist[t];
    let wB = 0, sumB = 0, maxVar = 0, thresh = 127;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sumTotal - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > maxVar) { maxVar = between; thresh = t; }
    }
    return thresh;
  }

  /**
   * estimateSkew(): two-stage projection-profile sweep — coarse ±10° @ 1°,
   * refine ±1° @ 0.25°. Cheaper than minAreaRect and adequate for documents.
   */
  function estimateSkew(grayCanvas) {
    const w = grayCanvas.width, h = grayCanvas.height;
    if (w < 16 || h < 16) return 0;
    const ctx = _ctx2d(grayCanvas);
    const px = ctx.getImageData(0, 0, w, h).data;
    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      lum[j] = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) | 0;
    }
    const hist = new Uint32Array(256);
    for (let i = 0; i < lum.length; i++) hist[lum[i]]++;
    const thresh = otsu(hist, lum.length);

    let bestAngle = 0, bestScore = -1;
    const sweep = (lo, hi, step) => {
      for (let deg = lo; deg <= hi + 1e-9; deg += step) {
        const tan = Math.tan((deg * Math.PI) / 180);
        let score = 0;
        for (let y = 2; y < h - 2; y += 4) {
          for (let x = 8; x < w - 8; x += 6) {
            const yy = y + Math.round(tan * x);
            if (yy < 1 || yy >= h - 1) continue;
            if (lum[yy * w + x] < thresh) score++;
          }
        }
        if (score > bestScore) { bestScore = score; bestAngle = deg; }
      }
    };
    sweep(-10, 10, 1);
    sweep(bestAngle - 1, bestAngle + 1, 0.25);
    return Math.abs(bestAngle) < 0.5 ? 0 : bestAngle;
  }

  /** 3x3 median on luma — the classical denoise from evaluate.arm_classical. */
  function median3(canvas) {
    const w = canvas.width, h = canvas.height;
    const src = _ctx2d(canvas).getImageData(0, 0, w, h);
    const s = src.data;
    const out = _ctx2d(makeCanvas(w, h)).createImageData(w, h);
    const o = out.data;
    const v = new Array(9);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        for (let k = 0; k < 9; k++) {
          v[k] = s[i + ((k / 3 | 0) - 1) * w * 4 + ((k % 3) - 1) * 4];
        }
        v.sort((a, b) => a - b);
        o[i] = o[i + 1] = o[i + 2] = v[4];
        o[i + 3] = 255;
      }
    }
    // replicate border
    for (let x = 0; x < w; x++) {
      for (const y of [0, h - 1]) {
        const i = (y * w + x) * 4;
        o[i] = o[i + 1] = o[i + 2] = s[i];
        o[i + 3] = 255;
      }
    }
    for (let y = 0; y < h; y++) {
      for (const x of [0, w - 1]) {
        const i = (y * w + x) * 4;
        o[i] = o[i + 1] = o[i + 2] = s[i];
        o[i + 3] = 255;
      }
    }
    const c = makeCanvas(w, h);
    _ctx2d(c).putImageData(out, 0, 0);
    return c;
  }

  /**
   * unsharp(): separable box-blur approximation of cv2.GaussianBlur(sigma=2)
   * followed by cv2.addWeighted(g, 1.8, blur, -0.8, 0). Two 1-D box passes
   * approximate a Gaussian closely enough for text-edge restoration and cost
   * O(n) instead of O(n*k^2), which matters on a phone.
   */
  function unsharp(canvas, amount = 1.8, radius = 2) {
    const w = canvas.width, h = canvas.height;
    const src = _ctx2d(canvas).getImageData(0, 0, w, h);
    const a = src.data;
    const lum = new Float32Array(w * h);
    for (let i = 0, j = 0; i < a.length; i += 4, j++) {
      lum[j] = 0.299 * a[i] + 0.587 * a[i + 1] + 0.114 * a[i + 2];
    }
    // horizontal box
    const tmp = new Float32Array(w * h);
    const n = 2 * radius + 1;
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let x = -radius; x <= radius; x++) acc += lum[y * w + clampi(x, 0, w - 1)];
      for (let x = 0; x < w; x++) {
        tmp[y * w + x] = acc / n;
        const outIdx = clampi(x - radius, 0, w - 1);
        const inIdx = clampi(x + radius + 1, 0, w - 1);
        acc += lum[y * w + inIdx] - lum[y * w + outIdx];
      }
    }
    // vertical box
    const blr = new Float32Array(w * h);
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let y = -radius; y <= radius; y++) acc += tmp[clampi(y, 0, h - 1) * w + x];
      for (let y = 0; y < h; y++) {
        blr[y * w + x] = acc / n;
        const outIdx = clampi(y - radius, 0, h - 1);
        const inIdx = clampi(y + radius + 1, 0, h - 1);
        acc += tmp[inIdx * w + x] - tmp[outIdx * w + x];
      }
    }
    for (let i = 0, j = 0; i < a.length; i += 4, j++) {
      const val = lum[j] * amount - blr[j] * (amount - 1);
      const c8 = val < 0 ? 0 : val > 255 ? 255 : val | 0;
      a[i] = a[i + 1] = a[i + 2] = c8;
    }
    const c = makeCanvas(w, h);
    _ctx2d(c).putImageData(src, 0, 0);
    return c;
  }

  function clampi(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  /** Tile-CLAHE, 8x8 grid, clipLimit 3.0 — mirrors evaluate.arm_restorer_clahe. */
  function clahe(grayCanvas, clipLimit = 3.0, grid = 8) {
    const w = grayCanvas.width, h = grayCanvas.height;
    const img = _ctx2d(grayCanvas).getImageData(0, 0, w, h);
    const p = img.data;
    const tileW = Math.max(32, Math.floor(w / grid));
    const tileH = Math.max(32, Math.floor(h / grid));
    for (let ty = 0; ty < h; ty += tileH) {
      for (let tx = 0; tx < w; tx += tileW) {
        const ex = Math.min(tx + tileW, w), ey = Math.min(ty + tileH, h);
        const hist = new Uint32Array(256);
        let cnt = 0;
        for (let yy = ty; yy < ey; yy++) {
          for (let xx = tx; xx < ex; xx++) { hist[p[(yy * w + xx) * 4]]++; cnt++; }
        }
        const clip = Math.max(1, (cnt * clipLimit / 256) | 0);
        let excess = 0;
        for (let v = 0; v < 256; v++) {
          if (hist[v] > clip) { excess += hist[v] - clip; hist[v] = clip; }
        }
        const add = (excess / 256) | 0;
        const lut = new Uint8Array(256);
        let cdf = 0;
        for (let v = 0; v < 256; v++) {
          cdf += hist[v] + add;
          lut[v] = Math.min(255, (cdf * 255 / cnt) | 0);
        }
        for (let yy = ty; yy < ey; yy++) {
          for (let xx = tx; xx < ex; xx++) {
            const idx = (yy * w + xx) * 4;
            const v = lut[p[idx]];
            p[idx] = p[idx + 1] = p[idx + 2] = v;
          }
        }
      }
    }
    const c = makeCanvas(w, h);
    _ctx2d(c).putImageData(img, 0, 0);
    return c;
  }

  /** Otsu binarize — the `restorer_clahe` arm's final step. */
  function binarize(grayCanvas) {
    const w = grayCanvas.width, h = grayCanvas.height;
    const img = _ctx2d(grayCanvas).getImageData(0, 0, w, h);
    const p = img.data;
    const hist = new Uint32Array(256);
    for (let i = 0; i < p.length; i += 4) hist[p[i]]++;
    const t = otsu(hist, w * h);
    for (let i = 0; i < p.length; i += 4) {
      const v = p[i] > t ? 255 : 0;
      p[i] = p[i + 1] = p[i + 2] = v;
    }
    const c = makeCanvas(w, h);
    _ctx2d(c).putImageData(img, 0, 0);
    return c;
  }

  function toGrayCanvas(source, w, h) {
    const c = makeCanvas(w, h);
    const ctx = _ctx2d(c);
    ctx.drawImage(source, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const l = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      d[i] = d[i + 1] = d[i + 2] = l;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  function rotate(canvas, deg) {
    const c = makeCanvas(canvas.width, canvas.height);
    const ctx = _ctx2d(c);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.save();
    ctx.translate(c.width / 2, c.height / 2);
    ctx.rotate((-deg * Math.PI) / 180);
    ctx.drawImage(canvas, -c.width / 2, -c.height / 2);
    ctx.restore();
    return c;
  }

  /**
   * preprocess(): load -> downscale -> grayscale -> quality -> deskew ->
   * restore -> CLAHE -> candidates.
   *
   * `restoreMode` decides which restoration is applied:
   *
   *   "adaptive"  (default) noise -> 3x3 median, blur -> unsharp, and NOTHING
   *               when neither is detected. This mirrors the desktop port, which
   *               assumes a neural restorer covers the remaining damage.
   *   "classical" median + unsharp unconditionally — evaluate.py's arm_classical.
   *   "none"      deskew only.
   *
   * WHY THE DEFAULT IS NOT ENOUGH WHEN THERE IS NO MODEL. A moderately blurred
   * capture lands around lapVar 135, just over the "blurry" threshold of 100, so
   * adaptive mode applies nothing at all and the raw blur goes straight into
   * OCR. Measured on rendered samples that cost 3/3 field recoveries on
   * handheld_light. With restorer.onnx absent, app.js selects "classical" so
   * the fallback matches the Python baseline rather than a weaker imitation of
   * it. See tools/e2e_ocr.js.
   *
   * Returns { canvas, candidates, quality, warnings, skew, width, height, arms }
   * where `arms` names each candidate with the evaluate.py arm it mirrors.
   */
  function preprocess(source, maxDim, opts) {
    const restoreMode = (opts && opts.restoreMode) || "adaptive";
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    if (!sw || !sh) throw new Error("No image dimensions — invalid input");
    const scale = Math.min(1, maxDim / Math.max(sw, sh));
    const w = Math.max(1, Math.round(sw * scale));
    const h = Math.max(1, Math.round(sh * scale));

    const gray = toGrayCanvas(source, w, h);
    const q = assessQuality(gray);
    const angle = estimateSkew(gray);
    const deskewed = angle !== 0 ? rotate(gray, angle) : gray;

    // Median kills salt-pepper but destroys blurred strokes; unsharp rescues
    // blur. Applied in that order when both are present.
    let restored = deskewed;
    if (restoreMode === "classical") {
      restored = unsharp(median3(deskewed), 1.8, 2);
    } else if (restoreMode === "adaptive") {
      if (q.noisy) restored = median3(restored);
      if (q.blurry || q.noisy) restored = unsharp(restored, 1.8, 2);
    }

    const enhanced = clahe(restored);
    const binary = binarize(enhanced);

    const candidates = [enhanced, binary];
    const arms = [
      restoreMode === "classical" ? "classical" : "restorer_clahe",
      "restorer_clahe",
    ];

    return {
      canvas: enhanced,
      candidates,
      arms,
      restoreMode,
      quality: q.quality,
      blurry: q.blurry,
      dark: q.dark,
      noisy: q.noisy,
      lapVar: q.lapVar,
      warnings: q.warnings,
      skew: angle,
      width: w,
      height: h,
    };
  }

  // ── OCR: tesseract.js multi-pass ──────────────────────────────────────

  let _worker = null;
  let _workerPromise = null;

  function vendorBase() {
    // Resolve vendor paths relative to this script so the app works from any
    // mount point (subdirectory deploys) — mirrors the sw precache layout.
    const scripts = document.getElementsByTagName("script");
    let base = "./";
    for (const s of scripts) {
      const src = s.src || "";
      const i = src.indexOf("/js/");
      if (i !== -1) { base = src.slice(0, i + 1); break; }
    }
    return base;
  }

  async function initWorker(onProgress) {
    if (_worker) return _worker;
    if (_workerPromise) return _workerPromise;
    if (typeof Tesseract === "undefined") {
      throw new Error("tesseract.js not loaded (vendor/tesseract/tesseract.min.js)");
    }
    const b = vendorBase();
    _workerPromise = Tesseract.createWorker("eng", 1, {
      workerPath: b + "vendor/tesseract/worker.min.js",
      corePath: b + "vendor/tesseract",
      langPath: b + "vendor/tessdata",
      gzip: true,
      logger: (m) => { if (onProgress) onProgress(m); },
    }).then((w) => { _worker = w; _workerPromise = null; return w; })
      .catch((e) => { _workerPromise = null; throw e; });
    return _workerPromise;
  }

  /**
   * structure_score(): how much READABLE STRUCTURE the text contains.
   *
   * WHY THIS EXISTS
   * ---------------
   * runOCR() picks between segmentation modes by Tesseract confidence alone,
   * and confidence is a poor proxy for the thing this product actually needs:
   * a legible case number.
   *
   * Measured, on legal/off_axis with restoration enabled:
   *
   *     restored  psm=6  conf 81%   -> finds BOTH fields   <- never chosen
   *     restored  psm=11 conf 69%   -> finds NEITHER      <- was chosen
   *
   * The psm=6 result was MORE confident and recovered more, and still lost,
   * because runOCR's early-exit threshold (>75%) never fired, so selection fell
   * through to "highest confidence across everything tried" and psm=11 — whose
   * sparse-text mode treats the page as disconnected fragments, so a case
   * reference split across two lines never reassembles — edged ahead on a
   * narrower margin than its structure warranted.
   *
   * Confidence measures how sure Tesseract is about the glyphs it emitted.
   * Structure measures whether the RESULT is usable. For a blind user reading a
   * dosage aloud, those are not the same question, and the second one is what
   * the product is graded on.
   *
   * The patterns below are the safety-critical tokens this app extracts. They
   * are counted, not scored against ground truth, so this stays a property of
   * the OCR output alone and never peeks at the answer key.
   */
  function structure_score(text) {
    const t = String(text || "");
    if (!t.trim()) return 0;
    let n = 0;
    // Dosages: "500mg", "500 mg", "2.5mg"
    const doses = t.match(/\d+(?:\.\d+)?\s*mg\b/gi);
    if (doses) n += doses.length;
    // Sort codes: "40-11-04"
    const sortCodes = t.match(/\b\d{2}-\d{2}-\d{2}\b/g);
    if (sortCodes) n += sortCodes.length;
    // Case / reference numbers: "2024-CV-00456"
    const cases = t.match(/\b\d{4}-[A-Z]{2}-\d{3,8}\b/g);
    if (cases) n += cases.length * 2;      // weighted: most safety-critical
    // Bare account numbers: 6-8 digit runs not part of a date/dose
    const accts = t.match(/(?<![\d-])\d{6,8}(?![\d-])/g);
    if (accts) n += accts.length;
    // Currency amounts: "£2,500.00" / "$1,250" / "GBP 2,500.00" / "EUR 99".
    // The ISO code form matters: bank statements write "GBP"/"USD" far more
    // often than a symbol, and a bare "GBP 2,500.00" scored 0 without it.
    const money = t.match(/(?:[£$€]|[A-Z]{3})\s?\d[\d,]*(?:\.\d{2})?/g);
    if (money) n += money.length;
    return n;
  }

  /**
   * isGarbage(): reject OCR output that is noise, not text. >500 words is
   * almost always a failed segmentation; >60% of words being 1-2 chars is
   * speckle.
   */
  function isGarbage(text, wordCount) {
    if (!text || String(text).trim().length < 10) return true;
    if (wordCount > 500) return true;
    const words = String(text).split(/\s+/).filter(Boolean);
    if (!words.length) return true;
    const short = words.filter((w) => w.length <= 2).length;
    return short / words.length > 0.6;
  }

  /**
   * runOCR(): PSM 6 -> 3 -> 11 across every deduped candidate.
   *
   * SELECTION IS STRUCTURE-AWARE, NOT CONFIDENCE-ONLY. Ranking uses
   * better(a, b) below; confidence is a tiebreak, not the objective. See
   * structure_score() for the measurement that forced this: a psm=6 result at
   * 81% confidence recovering BOTH fields lost to a psm=11 result at 69%
   * recovering NEITHER, purely because confidence was the only criterion.
   *
   * Early-exit at >=75% confidence AND structure >= 1. Requiring structure too
   * is deliberate: a confident read of a blank or badly-fragmented page should
   * not short-circuit the remaining passes.
   */
  async function runOCR(candidates, onProgress) {
    const worker = await initWorker(onProgress);
    const passes = [
      { psm: "6", label: "block" },
      { psm: "3", label: "auto" },
      { psm: "11", label: "sparse" },
    ];
    let best = null;
    const seen = new Set();
    const unique = [];
    for (const img of candidates) {
      if (seen.has(img)) continue;
      seen.add(img);
      unique.push(img);
    }

    /**
     * Is candidate `a` a better result than the incumbent `b`?
     *
     * Rules, in order:
     *   1. more structured tokens wins — this is the product metric
     *   2. then higher confidence
     *   3. then earlier PSM (6 before 3 before 11), because a uniform block is
     *      the correct reading of a document; sparse-text is a fallback for
     *      pages that genuinely are not blocks
     *
     * Rule 1 is a hard preference rather than a bonus. A result that recovers
     * a case number is usable and one that does not is not, regardless of how
     * confident Tesseract was about either.
     */
    const better = (a, b) => {
      if (!b) return true;
      if (a.structure !== b.structure) return a.structure > b.structure;
      if (a.confidence !== b.confidence) return a.confidence > b.confidence;
      return a.psmRank < b.psmRank;
    };

    for (const img of unique) {
      for (let pi = 0; pi < passes.length; pi++) {
        const p = passes[pi];
        await worker.setParameters({ tessedit_pageseg_mode: p.psm });
        const res = await worker.recognize(img);
        const data = res.data;
        const garbage = isGarbage(data.text, data.words ? data.words.length : 0);
        const conf = data.confidence || 0;
        if (onProgress) onProgress({ status: "pass " + p.label + " conf " + conf.toFixed(0) + "%" });
        if (!garbage && conf > 40) {
          const cand = {
            text: data.text,
            confidence: conf / 100,
            words: data.words || [],
            psm: p.psm,
            structure: structure_score(data.text),
            psmRank: pi,
          };
          if (better(cand, best)) best = cand;
          if (best.confidence > 0.75 && best.structure >= 1) return best;
        }
      }
    }
    if (!best) {
      await worker.setParameters({ tessedit_pageseg_mode: "6" });
      const res = await worker.recognize(candidates[0]);
      best = {
        text: res.data.text,
        confidence: (res.data.confidence || 0) / 100,
        words: res.data.words || [],
        psm: "6",
        structure: structure_score(res.data.text),
        psmRank: 0,
      };
    }
    return best;
  }

  async function dispose() {
    if (_worker) {
      try { await _worker.terminate(); } catch (e) { /* already gone */ }
      _worker = null;
      _workerPromise = null;
    }
  }

  // ── Understanding: shared by every downstream caller ──────────────────

  /**
   * understand(): the post-OCR chain. Every path (fallback, ONNX classifier)
   * funnels through this so the result shape is identical.
   *
   * @param {string} rawText
   * @param {{category?:string, confidence?:number, source?:string}} [override]
   *   Pre-computed classification (e.g. from classifier.js). Falls back to the
   *   keyword classifier when absent.
   */
  function understand(rawText, override) {
    const normalized = normalizeOCRText(rawText);
    let cls;
    let source;
    if (override && override.category && CATEGORIES.indexOf(override.category) !== -1) {
      cls = { category: override.category, confidence: override.confidence || 0.5 };
      source = override.source || "model";
    } else {
      const kw = classify(normalized);
      const kb = classify_doc(normalized);
      // If the two keyword views disagree, take the higher-confidence one;
      // ties go to the richer (longer-phrase) classifier.
      if (kw.category === kb) {
        cls = { category: kw.category, confidence: Math.max(kw.confidence, 0.5) };
      } else {
        cls = kw.confidence >= 0.5
          ? { category: kw.category, confidence: kw.confidence }
          : { category: kb, confidence: 0.5 };
      }
      source = "keywords";
    }
    const fields = extractFields(normalized, cls.category);
    const entities = extractEntities(normalized);
    const summary = buildSummary(cls.category, fields);
    return {
      ocrText: rawText,
      normalized,
      category: cls.category,
      categoryLabel: CATEGORY_LABELS[cls.category],
      classificationConfidence: cls.confidence,
      classificationSource: source,
      fields,
      entities,
      summary,
    };
  }

  /**
   * scan(): the full fallback pipeline — preprocess -> OCR -> understand.
   */
  async function scan(source, opts) {
    const o = opts || {};
    const onProgress = o.onProgress || function () {};
    onProgress({ status: "preprocessing", progress: 5 });
    const pp = preprocess(source, o.maxDim || DEFAULT_MAX_DIM);
    onProgress({ status: "image " + pp.width + "x" + pp.height + (pp.blurry ? " (blurry)" : ""), progress: 15 });

    onProgress({ status: "loading OCR engine", progress: 20 });
    const ocr = await runOCR(pp.candidates, (m) => {
      onProgress({ status: m.status || "OCR", progress: 20 + Math.min(60, (m.progress || 0) * 0.6) });
    });
    onProgress({ status: "OCR complete", progress: 85 });

    const u = understand(ocr.text);
    onProgress({ status: "done", progress: 100 });

    return Object.assign(u, {
      ocrConfidence: ocr.confidence,
      warnings: pp.warnings,
      quality: pp.quality,
      imageDims: pp.width + "x" + pp.height,
      skew: pp.skew,
      engine: "classical",
      timestamp: new Date().toISOString(),
    });
  }

  return {
    // config
    CATEGORIES, CATEGORY_LABELS, FIELDS,
    // evaluate.py port
    norm, word_accuracy, fields_found, fields_complete, exact_dosage_rate,
    classify_doc, classify_doc_scores,
    // pipeline
    normalizeOCRText, classify, extractFields, extractEntities, buildSummary,
    understand, scan,
    // preprocess primitives (exposed for restorer.js + selftest)
    assessQuality, estimateSkew, otsu, median3, unsharp, clahe, binarize,
    toGrayCanvas, makeCanvas, _ctx2d, preprocess,
    // ocr
    initWorker, runOCR, isGarbage, structure_score, dispose, vendorBase,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SightLine;
// Publish on globalThis as well as window. In a browser window === globalThis so
// this is a no-op difference; under CommonJS (tests, tools) window is absent or
// distinct, and restorer.js/classifier.js resolve the pipeline through
// globalThis.SightLine. Setting only window left them without a handle.
if (typeof globalThis !== "undefined") globalThis.SightLine = SightLine;
