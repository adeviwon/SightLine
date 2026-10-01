/**
 * SightLine service worker — offline-first, cache-first, same-origin only.
 *
 * PRECACHE RATIONALE. The 42 MB of vendored wasm + traineddata is what makes
 * the app work in airplane mode, so the install step must be exhaustive: a
 * partially-cached install is worse than none, because the app looks offline
 * and then fails at the first OCR pass. Every entry below is added
 * individually (not cache.addAll, which is all-or-nothing) and the failure
 * count is reported back to the page via postMessage so the badge can be
 * honest about what is actually available.
 *
 * RUNTIME BEHAVIOUR
 *   - cache-first for every same-origin GET (the app is fully static)
 *   - navigations fall back to index.html so a deep link works offline
 *   - cross-origin requests are passed straight through and never cached, so
 *     the offline boundary is a hard guarantee, not a convention
 *   - models/*.onnx and vendor/ort/* are cached opportunistically on first use
 *     rather than precached: they are optional accelerators and must not block
 *     or fail the install.
 *
 * Bump VERSION to invalidate. The activate handler deletes every other
 * sightline-* cache, so old wasm never lingers.
 */

const VERSION = "v1.0.0";
const CACHE = "sightline-" + VERSION;

/** Everything required for a scan to complete with no network. */
const PRECACHE = [
  "./",
  "./index.html",
  "./selftest.html",
  "./manifest.json",
  "./README.md",
  "./js/pipeline.js",
  "./js/restorer.js",
  "./js/classifier.js",
  "./js/tts.js",
  "./js/app.js",
  "./assets/icon-180.png",
  "./assets/icon-192.png",
  "./assets/icon-512.png",
  // OCR engine + language data. ~40 MB. This is the offline guarantee.
  "./vendor/tesseract/tesseract.min.js",
  "./vendor/tesseract/worker.min.js",
  "./vendor/tesseract/tesseract-core.wasm.js",
  "./vendor/tesseract/tesseract-core.wasm",
  "./vendor/tesseract/tesseract-core-simd.wasm.js",
  "./vendor/tesseract/tesseract-core-simd.wasm",
  "./vendor/tesseract/tesseract-core-lstm.wasm.js",
  "./vendor/tesseract/tesseract-core-lstm.wasm",
  "./vendor/tesseract/tesseract-core-simd-lstm.wasm.js",
  "./vendor/tesseract/tesseract-core-simd-lstm.wasm",
  "./vendor/tessdata/eng.traineddata.gz",
  // PDF rendering
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
  "./vendor/ort/ort.wasm.bundle.min.mjs",
  "./vendor/ort/ort.wasm.wasm",
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    const results = await Promise.allSettled(
      PRECACHE.map((url) => cache.add(new Request(url, { cache: "reload" })))
    );
    const failed = results.filter((r) => r.status === "rejected").length;
    const ok = PRECACHE.length - failed;
    if (failed > 0) {
      console.warn("[sw] precache: " + failed + "/" + PRECACHE.length + " failed");
    }

    // Report to any open client so the badge reflects reality immediately.
    const clients = await self.clients.matchAll({ includeUncontrolled: true, type: "window" });
    for (const c of clients) {
      c.postMessage({ type: "precache-report", ok: ok, total: PRECACHE.length, failed: failed });
    }
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
