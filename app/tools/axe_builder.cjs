/**
 * axe_builder.cjs — a drop-in for `@axe-core/playwright` built on a VENDORED
 * axe-core, so the accessibility gate runs with no package.json and no install.
 *
 * WHY THIS EXISTS
 * ---------------
 * `ui_redesign_check.cjs` hard-required `@axe-core/playwright`, which is not
 * installed and has no resolver here (`pw_core.js` resolves playwright-core from
 * a cache because this repo deliberately ships no package.json). So `bash run.sh
 * test-ui` died at require time and the WCAG gate it exists for reported
 * nothing — a gate that cannot run is decoration.
 *
 * The wrapper is not what does the work: `@axe-core/playwright` injects the
 * axe-core bundle into the page and calls `axe.run`. So vendor axe-core itself
 * (`vendor/axe.min.js`, 636 KB, MIT) and call it directly — one fewer dependency
 * than upstream had, and the run works with no network.
 *
 * Only the surface this repo uses is implemented (`withTags` / `analyze`).
 * `AXE_CORE` still overrides the module path if someone wants the real wrapper.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const AXE_JS = path.join(__dirname, "vendor", "axe.min.js");
// axe's own version, read from the vendored bundle's header comment rather than
// hardcoded, so an upgrade cannot leave the reported version lying.
function axeVersion() {
  try {
    const head = fs.readFileSync(AXE_JS, "utf8").slice(0, 400);
    const m = head.match(/axe\.js\s+([0-9][0-9.]*)/i) || head.match(/([0-9]+\.[0-9]+\.[0-9]+)/);
    return m ? m[1] : "unknown";
  } catch {
    return "missing";
  }
}

class AxeBuilder {
  constructor({ page } = {}) {
    if (!page) throw new Error("AxeBuilder({ page }) needs a page");
    this.page = page;
    this.tags = [];
    this.rulesDisabled = [];
    this.context = null;
  }

  withTags(tags) {
    this.tags = tags || [];
    return this;
  }

  disableRules(rules) {
    this.rulesDisabled = (rules || []).map((r) => (typeof r === "string" ? r : r.id));
    return this;
  }

  include(selector) {
    this.context = selector;
    return this;
  }

  async analyze() {
    if (!fs.existsSync(AXE_JS)) {
      throw new Error("vendored axe-core missing at " + AXE_JS +
                      " — the accessibility gate cannot run without it");
    }
    await this.page.addScriptTag({ path: AXE_JS });
    const opts = { runOnly: { type: "tag", values: this.tags } };
    if (this.rulesDisabled.length) opts.rules = Object.fromEntries(
      this.rulesDisabled.map((id) => [id, { enabled: false }]));
    const results = await this.page.evaluate(
      ([o, ctx]) => window.axe.run(ctx || document, o),
      [opts, this.context]);
    results.axeVersion = axeVersion();
    return results;
  }
}

module.exports = AxeBuilder;
// `@axe-core/playwright` is consumed as `.default`, so expose both — the caller
// keeps its one-line require either way.
module.exports.default = AxeBuilder;
module.exports.axeVersion = axeVersion;
module.exports.AXE_JS = AXE_JS;
