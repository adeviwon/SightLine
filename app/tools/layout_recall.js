/**
 * Does the slice-based detector actually beat the 62.6% projection-profile
 * recall? Run the real browser code against real photos and count bands.
 *
 *   node tools/layout_recall.js /tmp/pages
 *
 * We compare BAND COUNT against TRANSCRIPT LINE COUNT. That is a proxy, not
 * recall -- matching bands to lines needs the recogniser -- but it is the
 * number that changed before: the old detector returned 0.71 bands per
 * transcript line, and the fixed Python one 0.79. If this returns ~1.0 with a
 * tighter spread, lines are no longer being merged or dropped wholesale.
 *
 * The honest limitation is stated in the output: over-detection (a speck read
 * as a line) inflates this just as much as perfect detection does. It bounds
 * the problem in one direction only, which is why it is a screening test and
 * not the number that goes in the docs.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");

function loadLayout() {
  const src = fs.readFileSync(path.join(ROOT, "app/js/layout.js"), "utf8");
  const mod = src.replace(/^export default .*$/gm, "").replace(/^export /gm, "");
  const wanted = ["inkMask", "detectLines", "cropForRecognizer", "medianBlur"];
  return new Function("window", mod + "\n; return {" + wanted.join(",") + "};")(
    { });
}

function main() {
  const dir = process.argv[2] || "/tmp/pages";
  const mf = path.join(dir, "manifest.json");
  if (!fs.existsSync(mf)) {
    console.log(`no manifest at ${mf}\nrun: bash run.sh py ml/src/dump_pages.py ${dir} 20`);
    process.exit(1);
  }
  const pages = JSON.parse(fs.readFileSync(mf, "utf8"));
  const L = loadLayout();

  let totBands = 0, totWant = 0;
  const ratios = [];
  const heights = [];

  for (const p of pages) {
    const buf = fs.readFileSync(p.raw);
    const gray = new Uint8Array(buf.buffer, buf.byteOffset, p.w * p.h);
    const t0 = Date.now();
    const bands = L.detectLines(gray, p.w, p.h);
    const ms = Date.now() - t0;
    totBands += bands.length;
    totWant += p.want;
    const r = bands.length / Math.max(p.want, 1);
    ratios.push(r);
    for (const b of bands) heights.push(b.y1 - b.y0);

    const hs = bands.map(b => b.y1 - b.y0);
    const merged = hs.filter(h => h > 60).length;
    console.log(
      `${p.name.padEnd(20)} ${String(p.h).padStart(4)}px  ` +
      `want ${String(p.want).padStart(3)}  bands ${String(bands.length).padStart(3)}  ` +
      `ratio ${r.toFixed(2)}  ${String(ms).padStart(4)}ms` +
      (merged ? `  [${merged} merged h>60px]` : "")
    );
  }

  const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  const sorted = ratios.slice().sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  const H = heights.sort((a, b) => a - b);
  const pc = q => H[Math.min(H.length - 1, Math.floor(H.length * q / 100))];

  console.log("\n=== summary ===");
  console.log(`pages            ${pages.length}`);
  console.log(`transcript lines ${totWant}`);
  console.log(`bands detected   ${totBands}`);
  console.log(`bands/line       ${(totBands / totWant).toFixed(3)}` +
              `  (old Python detector: 0.79; before the grayscale fix: 0.71)`);
  console.log(`  mean ${mean.toFixed(3)}  median ${med.toFixed(3)}` +
              `  p10 ${sorted[2]?.toFixed(2)}  p90 ${sorted[sorted.length - 3]?.toFixed(2)}`);
  console.log(`band height      p10 ${pc(10)}px  p50 ${pc(50)}px  p90 ${pc(90)}px` +
              `   (>90 = merged block, not one line)`);
  console.log(`\nNOTE: band COUNT is a proxy. Over-detection inflates it as much as`);
  console.log(`perfect detection. The real recall needs recogniser output matched`);
  console.log(`to transcripts -- see ml/src/eval_ocr.py, which reports it as`);
  console.log(`detection_recall alongside line accuracy.`);

  // A merged block is the specific failure we are trying to remove, so count
  // them explicitly rather than leaving it to the ratio.
  const big = heights.filter(h => h > 90).length;
  console.log(`\nmerged bands (>90px tall): ${big}` +
              `  ${big ? "SOME LINES ARE STILL MERGED" : "none -- lines are separated"}`);
}

main();