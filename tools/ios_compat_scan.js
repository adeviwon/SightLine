/**
 * iOS compatibility gate: scan the SHIPPED JS for syntax that older Safari
 * cannot even parse.
 *
 * Why a scan and not a lint rule: the interesting failures are PARSE-TIME, not
 * stylistic. A regex lookbehind in a classic script makes WebKit < 16.4 throw a
 * SyntaxError for the whole file, so `window.TTS` never exists and every
 * read-aloud call fails — while every test on a current engine still passes.
 * No ESLint target expresses "Safari 16.4", and a bundler cannot transpile a
 * lookbehind away. A twenty-line scanner that names the file and line does.
 *
 * Run: node tools/ios_compat_scan.js      (exit 0 = clean, 1 = found)
 * Also runnable from the repo root: bash run.sh ios-check
 *
 * What this does NOT check: runtime feature support (WebAssembly SIMD, COOP/
 * COEP, service-worker storage policy). Those are probed at runtime —
 * recognizer.js simdSupported() and tools/ios_webkit_check.js.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const APP = path.resolve(__dirname, "..");

// Files the browser actually loads. vendor/ is excluded: those are third-party
// builds whose own compatibility floor is their own business (and pdf.js is
// the thing we would have to re-vendor if it ever broke).
const dirs = ["js", "."];
const skip = /^(vendor|tools|node_modules)\//;

const RULES = [
  {
    name: "regex lookbehind",
    why: "Safari/iOS < 16.4 — PARSE-TIME SyntaxError, kills the whole script",
    // (?<=  and  (?<!   — NOT (?<name>, which is a named capture group and has
    // been supported since Safari 11.1. Flagging that would be a false positive
    // on perfectly compatible code.
    re: /\(\?<[=!]/,
  },
];

function files() {
  const out = [];
  for (const d of dirs) {
    const full = path.join(APP, d);
    if (!fs.existsSync(full)) continue;
    for (const name of fs.readdirSync(full)) {
      const rel = d === "." ? name : d + "/" + name;
      if (skip.test(rel)) continue;
      if (!/\.(js|mjs|html)$/.test(name)) continue;
      if (d === "." && name !== "index.html" && name !== "selftest.html") continue;
      out.push(rel);
    }
  }
  return out.sort();
}

let bad = 0;
const scanned = [];
for (const rel of files()) {
  const text = fs.readFileSync(path.join(APP, rel), "utf8");
  scanned.push(rel);
  const lines = text.split("\n");
  lines.forEach((line, i) => {
    for (const rule of RULES) {
      if (rule.re.test(line)) {
        bad++;
        console.log(`${rel}:${i + 1}: ${rule.name} — ${rule.why}`);
        console.log("    " + line.trim().slice(0, 120));
      }
    }
  });
}

console.log(`\nios-compat: scanned ${scanned.length} shipped files, ${bad} problem(s)`);
if (!bad) console.log("  no parse-time Safari-16.4 blockers found");
process.exit(bad ? 1 : 0);
