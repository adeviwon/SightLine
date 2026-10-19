/**
 * Where is Chromium?
 *
 * Every browser harness in this directory used to hardcode
 * `/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`. That
 * revision is decided by whichever Playwright installed it, so a Playwright
 * update (or a `playwright install` of another engine, which prunes revisions it
 * does not own) deletes the binary and every harness dies with ENOENT on a
 * browser that will never exist again — which is exactly what happened.
 *
 * Resolution order:
 *   1. CHROME_BIN, when set (CI, a different machine)
 *   2. the NEWEST chromium-* under the Playwright browsers path
 *
 * Usage:  const CHROME = require("./chrome_path.js")();
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

module.exports = function chromePath() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH
    || path.join(os.homedir(), ".cache", "ms-playwright");
  try {
    const versions = fs.readdirSync(root)
      .filter((d) => /^chromium-\d+$/.test(d))
      .sort((a, b) => Number(b.split("-")[1]) - Number(a.split("-")[1]));
    for (const v of versions) {
      for (const rel of ["chrome-linux64/chrome", "chrome-linux/chrome"]) {
        const bin = path.join(root, v, rel);
        if (fs.existsSync(bin)) return bin;
      }
    }
  } catch (e) { /* fall through */ }
  // Last resort: keep the historical path so the error message names a real
  // location instead of being a bare "not found".
  return "/home/ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome";
};
