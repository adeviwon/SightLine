/** SightLine offline assets. Both reading methods and the shared runtime are
 * required at install, including the enhanced reader's module worker. Restorer
 * and MiniLM remain optional accelerators. A partial required install is refused.
 */

// Both lanes bumped this on the same day: the spoken-reading work (v1.14.1) and the
// word-LM decoder. One number has to cover both or installed clients keep serving
// whichever bundle they cached first — patch bump on top of the higher base.
// Bumped for the word-LM's metadata pass: app/data/word_trie.json gained the
// scoring constants the browser now READS instead of hardcoding, and
// app/js/word_beam.js changed to consume them. A precached ASSET change needs the
// bump or installed clients keep the old pair from cache (and a stale trie's
// defaults would silently decode with the pre-sweep numbers).
const VERSION = "v1.15.0-doses-word-lm";
const CACHE = "sightline-" + VERSION;

/** Everything required for a scan to complete with no network. */
const PRECACHE = [
  "./",
  "./index.html",
  "./selftest.html",
  "./manifest.json",
  "./README.md",
  // Availability manifest for the optional ONNX path. Precached so the app can
  // answer "are the models present?" with zero network requests.
  "./models/ort.json",
  "./js/pipeline.js",
  "./js/medicine_sections.js",
  "./js/gating.js",
  "./js/restorer.js",
  "./js/recognizer.js",
  "./js/word_beam.js",
  "./js/classifier.js",
  "./js/tts.js",
  "./js/speech_plan.js",
  "./js/device_check.js",
  "./assets/device.css",
  "./js/say_cantonese.js",
  "./data/drug_lexicon.json",
  // The word-LM lexicon trie. PRECACHED, not fetched on demand: a cold offline
  // install that cannot reach it would silently fall back to the greedy decoder
  // — the older, measurably worse read (67.97% vs 70.67% exact on held-out
  // crops) with nothing in the UI to say so.
  "./data/word_trie.json",
  "./js/app.js",
  "./js/history.js",
  "./js/i18n.js",
  "./js/cantonese_summary.js",
  "./js/field_localization.js",
  "./js/receipt.mjs",
  "./js/aria/fields.mjs",
  "./js/aria/taxonomy.mjs",
  "./js/aria/config.mjs",
  // The OCR ENGINE's own modules. pipeline.js dynamic-imports these on the
  // first scan; without them in the precache a scan on a device that is already
  // offline fails at the import, which is what the "cold install" check in
  // tools/ios_full_check.js caught: install online, airplane mode, first scan
  // -> "Failed to fetch dynamically imported module ocr_engine.js".
  "./js/ocr_engine.js",
  "./js/layout.js",
  // The label-anchored extraction path (field_spot + field_guard) and the module
  // that composes them. pipeline.js injects these three as script tags, which
  // works ONLINE but leaves an offline cold start falling back to the regex path
  // -- silently, and the fallback is the weaker extractor. Precaching them makes
  // the opt-in path offline-complete. (The engine is OFF by default: recall vs
  // safety is a product call, not a plumbing one.)
  "./js/field_spot.js",
  "./js/field_guard.js",
  "./js/field_engine.js",
  "./js/ORCHESTRATION.md",
  "./assets/icon-180.png",
  "./assets/icon-192.png",
  "./assets/icon-512.png",
  "./assets/favicon.ico",
  "./assets/reader.css",
  "./THIRD_PARTY.md",
  "./js/ppocr.js",
  "./js/ppocr.worker.js",
  "./js/ppocr_core.js",
  "./js/ppocr_math.js",
  "./models/ppocrv5-mobile-det.onnx",
  "./models/ppocrv5-mobile-rec.onnx",
  "./models/ppocrv5_dict.txt",
  "./models/ppocr-provenance.json",
  "./models/PP-OCR-LICENSE.txt",
  "./models/crnn.onnx",
  "./vendor/ort/ort.min.mjs",
  "./vendor/ort/ort-wasm-simd-threaded.mjs",
  "./vendor/ort/ort-wasm-simd-threaded.wasm",
  // PDF rendering. Required offline payload size is measured by precache_check;
  // PP-OCR and the original CRNN share the required WASM runtime above.
  "./vendor/pdfjs/pdf.min.js",
  "./vendor/pdfjs/pdf.worker.min.js",
];

/**
 * Optional accelerators: fetched and cached on first use, never required.
 * Their absence downgrades the app to the classical path, which is complete.
 */
const OPTIONAL = [
  "./models/restorer.onnx",
  "./models/minilm_encoder.onnx",
  "./models/minilm_head.onnx",
  "./models/tokenizer.json",
  // Filenames must match prune_ort.sh's KEEP list exactly. The .wasm binary is
  // the one that matters most: without it ORT silently falls back to looking
  // for it on a CDN, which would break the offline guarantee.
  "./vendor/ort/ort.min.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // Bound installation traffic while the first OCR scan may fetch the same
    // large files. Keep every required asset and every failure visible.
    const results = new Array(PRECACHE.length);
    let next = 0;
    async function saveEntries() {
      while(next < PRECACHE.length) {
        const index = next++;
        try { await cache.add(new Request(PRECACHE[index], {cache:"reload"})); results[index] = {status:"fulfilled"}; }
        catch(reason) { results[index] = {status:"rejected",reason}; }
      }
    }
    await Promise.all([saveEntries(),saveEntries()]);
    // Large reading files can fail during a burst of concurrent requests.
    // Retry rejected assets once, sequentially. Required files still MUST all
    // succeed; a persistent failure never becomes an offline-ready claim.
    for (let index = 0; index < results.length; index++) {
      if (results[index].status !== "rejected") continue;
      try {
        await cache.add(new Request(PRECACHE[index], { cache: "reload" }));
        results[index] = { status: "fulfilled" };
      } catch (reason) { results[index] = { status: "rejected", reason }; }
    }
    const failed = results.filter((r) => r.status === "rejected").length;
    const ok = PRECACHE.length - failed;
    if (failed > 0) {
      console.warn("[sw] precache: " + failed + "/" + PRECACHE.length + " failed");
    }

    // Report to any open client so the badge reflects reality immediately.
    const clients = await self.clients.matchAll({ includeUncontrolled: true, type: "window" });
    for (const c of clients) {
      c.postMessage({ type: "precache-report", ok: ok, total: PRECACHE.length, failed: failed,
        // Asset-only diagnostics for tests/helpers; no document contents.
        failures: results.flatMap((result, index) => result.status === "rejected"
          ? [{ asset: PRECACHE[index], reason: String(result.reason) }] : []),
      });
    }
    if (failed) throw new Error("Required reading files could not be saved.");
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys.filter((k) => k.startsWith("sightline-") && k !== CACHE).map((k) => caches.delete(k))
    );
    if (self.registration.navigationPreload) {
      try { await self.registration.navigationPreload.enable(); } catch (e) { /* optional */ }
    }
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  let url;
  try { url = new URL(req.url); } catch (e) { return; }

  // Hard offline boundary: nothing cross-origin is intercepted or stored.
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE);

    // Cache-first. ignoreSearch so ?v= busts hit the same cached body.
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return hit;

    try {
      const resp = await fetch(req);
      if (resp && resp.ok && resp.type === "basic") {
        // Store asynchronously; never make the caller wait on the write.
        cache.put(req, resp.clone()).catch(() => { /* quota — degrade gracefully */ });
      }
      return resp;
    } catch (err) {
      // Offline and uncached: navigations get the app shell so a deep link
      // (or a relaunch from the home screen) still boots.
      if (req.mode === "navigate") {
        const shell = await cache.match("./index.html");
        if (shell) return shell;
      }
      const optional = await cache.match(req, { ignoreSearch: true });
      if (optional) return optional;
      return new Response("Offline and not cached: " + url.pathname, {
        status: 504,
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }
  })());
});

// Explicit precache of the optional accelerators, so a user can pay the
// download cost once and then be fully offline including the model path.
self.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type === "precache-optional" && event.source) {
    event.waitUntil((async () => {
      const cache = await caches.open(CACHE);
      const results = await Promise.allSettled(
        OPTIONAL.map((url) => cache.add(new Request(url, { cache: "reload" })))
      );
      const ok = results.filter((r) => r.status === "fulfilled").length;
      event.source.postMessage({
        type: "optional-report", ok: ok, total: OPTIONAL.length,
        failed: OPTIONAL.length - ok,
      });
    })());
  }
});


// A reload must verify actual cache entries, rather than equating connectivity
// or an activated worker with complete offline readiness.
self.addEventListener("message", event => {
  if (event.data?.type !== "precache-status" || !event.source) return;
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    const entries = await Promise.all(PRECACHE.map(path => cache.match(path)));
    const ok = entries.filter(Boolean).length;
    event.source.postMessage({ type: "precache-report", ok, total: PRECACHE.length, failed: PRECACHE.length - ok });
  })());
});
