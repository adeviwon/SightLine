"use strict";
// Verify every path listed in sw.js actually exists, and that every shipped
// file a page needs is listed. A precache entry pointing at a missing file is
// a silent offline failure; an unlisted file is an online-only dependency.
const fs = require("fs");
const path = require("path");
const APP = path.resolve(__dirname, "..");

const src = fs.readFileSync(path.join(APP, "sw.js"), "utf8");
const block = src.slice(src.indexOf("const PRECACHE"), src.indexOf("];", src.indexOf("const PRECACHE")));
const listed = block.split("\n")
  .map((l) => l.trim())
  .filter((l) => l.startsWith('"./'))
  .map((l) => {
    // Trim the surrounding quotes only. The naive version also ate a leading
    // "/" from a partially-matched line, which turned every path into
    // "/x\" and reported 29 missing files that all exist.
    return l.replace(/^"/, "").replace(/",?$/, "").replace(/^\.\//, "");
  });

let missing = 0;
for (const rel of listed) {
  const p = path.join(APP, rel);
  if (!fs.existsSync(p)) { console.log("  MISSING  " + rel); missing++; }
}
console.log("precache entries: " + listed.length + ", missing: " + missing);

// Anything shipped outside vendor/ that the app loads but the SW does not list.
const shipped = [];
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.name === "vendor" || e.name === "tools" || e.name === "node_modules") continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walk(full);
    else shipped.push(path.relative(APP, full));
  }
})(APP);

const unlisted = shipped.filter((f) => !listed.includes(f));
console.log("shipped files: " + shipped.length);
console.log(unlisted.length ? "  not precached:\n    " + unlisted.join("\n    ")
  : "  every shipped file is precached");
process.exit(missing === 0 ? 0 : 1);
