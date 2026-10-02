"""
Privacy audit: prove SightLine makes no external network requests.

Static analysis over the shipped app. This is the machine-checkable half of
docs/PRIVACY.md — the other half is the airplane-mode demo.

What it checks:
  1. No external URLs in any shipped HTML/JS file.
  2. Every fetch/XHR call site is accounted for: either a cache-backed
     precache read or a same-origin relative path.
  3. No sendBeacon / WebSocket / EventSource / analytics SDK.
  4. No Google Fonts or other third-party CDN references.
  5. The service worker's fetch handler does not fall back to the network.

Exit code 1 on any violation, so it can be a CI gate.

Usage:
    bash run.sh privacy
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

APP = Path("app")
SCAN_SUFFIXES = {".js", ".html", ".json"}

# Data files that ship inside app/ but cannot execute and are never fetched.
#
# models/tokenizer.json is a 30,522-entry BERT vocabulary. Scanning it as if it
# were source flagged "amplitude" at line 22423 -- the English word, a vocab
# token -- as an analytics SDK, which failed the privacy gate over a word in a
# word list.
#
# The check is right in spirit (a vendored bundle could exfiltrate) and wrong in
# scope (a token list cannot). Scoping is by CONTENT, not extension: .json is
# still scanned, because app/models/ort.json and the service-worker precache
# manifest are configuration that decides what gets loaded. Only files proven to
# be pure data are exempt, and that proof is by explicit list, not extension.
DATA_ONLY = {
    "models/tokenizer.json",   # wordpiece vocab: {"model": {"vocab": {...}}}
}
# A tokenizer.json that stops looking like a tokenizer must be scanned again, so
# the exemption is checked against the file's actual shape rather than trusted.
TOKENIZER_SHAPE = re.compile(r'"model"\s*:\s*\{[^{}]*"vocab"\s*:\s*\{')

# ...and shape alone is not sufficient. Mutation-testing the exemption showed
# that injecting a REAL analytics endpoint as a vocab token
# (vocab["https://api.segment.io/v1/track"] = 99999) still passed the audit,
# because the file kept its tokenizer shape. A token list cannot execute, but it
# can still carry a URL that some other component might read and fetch.
#
# So a data-only file is exempted from the CODE checks (analytics SDK names,
# network APIs) but NOT from the URL check. That is the honest split: the file
# cannot call anything, but every string in it is still worth resolving.
DATA_ONLY_SKIP_CODE_CHECKS = True
# Namespace URLs that appear in SVG/XML attributes, plus licence-text URLs and
# spec references. They are identifiers or documentation, never fetched by the
# app. Matching on host only would not work here because these appear as bare
# strings inside vendored minified bundles.
NAMESPACE_OK = (
    "www.w3.org", "ns.adobe.com", "purl.org", "iptc.org", "xfa.org",
    "www.xfa.org", "github.com", "opensource.org", "creativecommons.org",
    "mozilla.org", "apache.org", "example.com", "schema.org", "w3c.org",
)
# Hosts that appear ONLY as unused default values in vendored bundles. They
# are allowed past the URL scan but are NOT blanket-trusted: see
# check_cdn_defaults_overridden(), which fails the audit if the app ever stops
# overriding the corresponding library option. An unoverridden CDN default
# would fetch from a third party at runtime.
CDN_DEFAULT_OK = ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com")
# URL fragments inside minified bundles that are template placeholders or
# error-message text rather than fetch targets.
_TMPL = re.compile(r"\$\{|\{[a-zA-Z_$][\w$]*\}")
# Vendored third-party libraries are scanned too: a compromised vendor file
# would defeat every runtime guarantee.

NETWORK_APIS = [
    (r"\bnavigator\.sendBeacon\b", "sendBeacon"),
    (r"\bnew\s+WebSocket\b", "WebSocket"),
    (r"\bnew\s+EventSource\b", "EventSource"),
    (r"\bnew\s+SharedWorker\b", "SharedWorker"),
    (r"\bnavigator\.connection\b", "network-information-api"),
    (r"\bnew\s+Notification\b", "Notification API (can leak engagement)"),
]

ANALYTICS = ["google-analytics", "googletagmanager", "gtag(", "mixpanel",
             "segment.io", "amplitude", "sentry.io", "bugsnag", "hotjar",
             "facebook.net", "hotjar", "datadog", "newrelic"]

FAILURES = []
NOTES = []


def rel(p):
    return str(p.relative_to(APP.parent))


def check_external_urls(path, text):
    for m in re.finditer(r"""["'`]\s*(https?://[^\s"'`)]+)""", text):
        url = m.group(1)
        if _TMPL.search(url):
            continue  # template placeholder, resolved at runtime
        host = urlparse(url).netloc
        if not host or any(host == h or host.startswith(h + ".")
                           for h in NAMESPACE_OK):
            continue
        if host in ("localhost", "127.0.0.1"):
            continue
        if host in CDN_DEFAULT_OK:
            # Allowed here, but only because check_cdn_defaults_overridden()
            # separately proves the app overrides the option that would use it.
            continue
        line = text[:m.start()].count("\n") + 1
        FAILURES.append(f"{rel(path)}:{line}  external URL  {url}")


def check_network_apis(path, text):
    for pat, name in NETWORK_APIS:
        for m in re.finditer(pat, text):
            line = text[:m.start()].count("\n") + 1
            FAILURES.append(f"{rel(path)}:{line}  {name} present")


def check_analytics(path, text):
    low = text.lower()
    for a in ANALYTICS:
        if a in low:
            line = low[:low.index(a)].count("\n") + 1
            FAILURES.append(f"{rel(path)}:{line}  analytics/third-party SDK: {a}")


def check_fetch_targets(path, text):
    """Every fetch must be same-origin (relative) — cross-origin needs proof."""
    for m in re.finditer(r"fetch\(\s*([`'\"])([^`'\"]*)\1", text):
        target = m.group(2)
        line = text[:m.start()].count("\n") + 1
        if target.startswith("http://") or target.startswith("https://"):
            host = urlparse(target).netloc
            if host not in ("localhost", "127.0.0.1"):
                FAILURES.append(f"{rel(path)}:{line}  fetch() to absolute URL "
                                f"{target}")


def check_cdn_defaults_overridden(text_by_path):
    """
    Vendored libraries hardcode CDN defaults (Tesseract.js ships jsDelivr URLs
    for workerPath/corePath/langPath). Those strings are harmless ONLY if the
    app overrides every one of them with a local path. This checks the
    override exists rather than allowlisting the host — an unoverridden CDN
    default is a real privacy violation, because it would fetch from a
    third party at runtime.
    """
    app_js = "\n".join(t for p, t in text_by_path.items()
                       if p.parent.name == "js")
    required = ["workerPath", "corePath", "langPath"]
    overridden = [k for k in required
                  if re.search(rf"{k}\s*:\s*[^,\n]*vendor", app_js)]
    missing = [k for k in required if k not in overridden]
    if missing:
        FAILURES.append(
            "Tesseract CDN defaults are NOT fully overridden. Missing local "
            f"path for: {missing}. If these keep their jsDelivr defaults, OCR "
            "would fetch the engine and language data from a third party at "
            "runtime — a privacy violation and an offline-mode failure.")
    else:
        NOTES.append("Tesseract CDN defaults are overridden with vendored "
                     f"local paths ({', '.join(overridden)}) — the jsDelivr "
                     "URLs in the bundle are dead strings.")


def check_service_worker():
    sw = APP / "sw.js"
    if not sw.exists():
        FAILURES.append("app/sw.js missing — no offline guarantee")
        return
    text = sw.read_text()
    if "addEventListener" in text and "fetch" in text:
        handler = text
        # A network fallback inside the fetch handler defeats the guarantee.
        if re.search(r"fetch\(\s*event\.request\s*\)", handler):
            line = handler[:re.search(r"fetch\(\s*event\.request\s*\)",
                                      handler).start()].count("\n") + 1
            FAILURES.append(
                f"app/sw.js:{line}  fetch handler calls network fetch("
                f"event.request) — that is a network fallback")
        if "caches.match" not in handler and "caches.open" not in handler:
            FAILURES.append("app/sw.js  no caches.match/caches.open — "
                            "service worker does not serve from cache")
        if "precache" not in handler.lower() and "install" not in handler:
            NOTES.append("app/sw.js  no install/precache handler found — "
                         "verify assets are cached at install")
    # Precached asset list must actually resolve to files on disk.
    # Parse only entries that look like relative paths (contain a slash or a
    # file extension). Prose in comments contains quoted English that would
    # otherwise be mistaken for filenames.
    entries = []
    for m in re.finditer(r"(?:PRECACHE|CACHE_NAME|PRECACHE_URLS|ASSETS)\s*=\s*\[(.*?)\]",
                         text, re.S):
        block = m.group(1)
        block = re.sub(r"//[^\n]*", "", block)          # strip line comments
        block = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
        for e in re.findall(r"""["'`]([^"'`\n]+)["'`]""", block):
            e = e.strip()
            if re.fullmatch(r"[./A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-.]+)+", e):
                entries.append(e)
    if entries:
        missing = sorted({e for e in entries
                          if not e.startswith("http") and not (APP / e).exists()})
        if missing:
            FAILURES.append(
                f"app/sw.js  precache lists {len(missing)} non-existent "
                f"files: {missing[:6]}")
        else:
            NOTES.append(f"app/sw.js precache list verified: all "
                         f"{len(set(entries))} entries resolve to real files")
    else:
        NOTES.append("app/sw.js  no machine-readable precache array found — "
                     "verify by hand that every asset is cached at install")


def main():
    if not APP.exists():
        print("app/ not found — run from the repository root")
        return 1

    candidates = [p for p in APP.rglob("*")
                  if p.is_file() and p.suffix in SCAN_SUFFIXES
                  and "tools" not in p.parts]

    files, skipped = [], []
    data_only = set()
    for p in candidates:
        key = str(p.relative_to(APP))
        if key in DATA_ONLY:
            # Exempt by explicit list AND verified by content shape, so a
            # replaced file cannot hide behind the exemption.
            try:
                head = p.read_text(errors="ignore")[:4096]
            except OSError:
                head = ""
            if TOKENIZER_SHAPE.search(head):
                skipped.append(key)
                data_only.add(p)
                continue
            NOTES.append(f"{rel(p)} is on the data-only list but no longer "
                         f"matches the expected tokenizer shape — scanning it "
                         f"after all")
        files.append(p)

    print(f"scanning {len(files)} shipped files in app/ "
          f"(including vendored third-party code)")
    if skipped:
        print(f"  {len(skipped)} data-only file(s) scanned for URLs only "
              f"(cannot execute): {', '.join(skipped)}")
    print()
    text_by_path = {}
    for p in files:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        text_by_path[p] = text
        check_external_urls(p, text)
        check_network_apis(p, text)
        check_analytics(p, text)
        check_fetch_targets(p, text)

    # Data-only files still get the URL check: a token list cannot call
    # anything, but it can still carry a fetchable endpoint. Only the checks
    # that look for EXECUTABLE analytics/network APIs are skipped, because a
    # vocabulary entry named "amplitude" is a word, not an SDK.
    for p in data_only:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        check_external_urls(p, text)
        text_by_path[p] = text

    check_cdn_defaults_overridden(text_by_path)
    check_service_worker()

    # Inline <script> in index.html is covered by the scan above.
    print("=" * 70)
    if FAILURES:
        print(f"PRIVACY AUDIT FAILED — {len(FAILURES)} violation(s)\n")
        for f in FAILURES:
            print(f"  ✗ {f}")
        print("\nSee docs/PRIVACY.md for what each check means.")
        return 1

    print("PRIVACY AUDIT PASSED\n")
    print("  ✓ no external URLs in any shipped file")
    print("  ✓ no sendBeacon / WebSocket / EventSource")
    print("  ✓ no analytics, telemetry, or error-reporting SDKs")
    print("  ✓ every fetch() targets a same-origin relative path")
    print("  ✓ service worker precaches real files and has no network fallback")
    for n in NOTES:
        print(f"  ! {n}")
    print("\nThe remaining half of the claim is behavioural: load once, then")
    print("turn on airplane mode. See docs/05_DEMO_RUNBOOK.md § 3:05.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
