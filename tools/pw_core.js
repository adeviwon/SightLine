/**
 * Where is playwright-core?
 *
 * `require("playwright-core")` only resolves when it happens to be installed in
 * a node_modules above the script — which it is not in this repo (there is no
 * package.json and no npm install step by design). The package does exist on
 * this machine inside an npx cache and other projects, so resolve it by looking
 * in the known places instead of assuming Node's module paths.
 *
 * Resolution order:
 *   1. PW_CORE — an explicit directory (CI, another machine)
 *   2. the local npx cache
 *   3. a sibling project that has it vendored
 *
 * Usage:  const { chromium } = require("./pw_core.js")();
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

const CANDIDATES = [
  process.env.PW_CORE,
  "/home/ubuntu/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core",
  "/home/ubuntu/OpenMAIC/node_modules/.pnpm/playwright-core@1.58.2/node_modules/playwright-core",
].filter(Boolean);

module.exports = function pwCore() {
  for (const dir of CANDIDATES) {
    if (dir && fs.existsSync(path.join(dir, "package.json"))) return require(dir);
  }
  // Not found: let Node produce its own error, but name what was tried.
  try {
    return require("playwright-core");
  } catch (e) {
    const searched = CANDIDATES.concat([path.join(os.homedir(), ".npm/_npx/*/node_modules/playwright-core")]);
    throw new Error(
      "playwright-core not found. Looked in:\n  " + searched.join("\n  ")
      + "\nSet PW_CORE=<dir containing playwright-core>.");
  }
};
