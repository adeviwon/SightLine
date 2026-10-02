/**
 * Verify the JS recogniser's CTC decoder against the Python reference.
 *
 *   node tools/recognizer_parity.js
 *
 * The decoder is the safety-critical part: getting repeat handling wrong
 * corrupts every price, dosage and account number containing a doubled digit
 * (500 -> 50), and that failure is invisible in aggregate accuracy. So this
 * compares the two implementations on the cases where they disagree if either
 * is wrong -- and it reads the ACTUAL Python values rather than hardcoding a
 * second copy of the expected answers, so the two cannot drift together.
 */

const { execFileSync } = require("child_process");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const VPY = path.join(ROOT, ".venv", "bin", "python");

/** Pull the reference decoder and constants out of Python, once. */
function pythonRef(cases) {
  const payload = JSON.stringify(cases);
  const script = `
import json, sys
sys.path.insert(0, "ml/src")
import realdata as R
cases = json.loads(sys.argv[1])
out = []
for seq, limit in cases:
    text = R.decode_greedy(seq, limit) if limit is not None else R.decode_greedy(seq)
    out.append(text)
print(json.dumps({
    "charset": R.CHARSET,
    "blank": R.BLANK,
    "decoded": out,
}))
`;
  const raw = execFileSync(VPY, ["-c", script, payload], {
    cwd: ROOT,
    encoding: "utf8",
    env: { ...process.env, PYTHONPATH: "", VIRTUAL_ENV: "", PYTHONHOME: "" },
  });
  return JSON.parse(raw);
}

(async () => {
  const src = require("fs").readFileSync(
    path.join(ROOT, "app/js/recognizer.js"), "utf8");
  // Strip the ORT import and the ESM export keywords so the pure functions can
  // be exercised without WASM. The `export default` line must be removed
  // WHOLELY -- stripping only the keyword leaves a bare `default`, which is a
  // syntax error, not a harmless leftover.
  const mod = src
    .replace(/^import .*$/gm, "")
    .replace(/^export default .*$/gm, "")
    .replace(/^export /gm, "");
  // The module declares its bindings as `export const`/`export function`, so
  // stripping the keyword leaves bare declarations in the function scope rather
  // than properties on module.exports. Collect them explicitly instead of
  // relying on module.exports, which would stay empty.
  const wanted = ["CHARSET", "BLANK", "CROP_H", "EXPORT_W", "WIDTH_DIVISOR",
                  "decodeGreedy", "timestepsFor", "recognise", "init"];
  const sandbox = { window: undefined, navigator: { hardwareConcurrency: 4 },
                    console };
    const JS = new Function("window", "navigator", "console",
                            mod + "\n; return {" + wanted.join(",") + "};")
      (sandbox.window, sandbox.navigator, sandbox.console);

  let fails = 0, checks = 0;
  const check = (label, cond, detail) => {
    checks++;
    if (cond) {
      console.log(`  ok    ${label}`);
    } else {
      fails++;
      console.log(`  FAIL  ${label}${detail ? "  " + detail : ""}`);
    }
  };

  console.log("charset + constants agree with Python:");
  const cases = [];
  const seqs = [
    [[10, 0, 1, 0, 1], null],           // "900" -- blank-separated repeats
    [[10, 0, 1, 1], null],              // "90"  -- CTC collapse
    [[30, 25, 30, 11, 22, 37, 10, 0, 1, 0, 1], null],  // "TOTAL 900"
    [[0, 0, 0], null],                  // all blank
    [[], null],                         // empty
    [[5, 5, 5, 0, 5], null],            // repeats then break
    [[10, 0, 1, 0, 1, 0, 2, 0, 2], 4],  // limit truncates
  ];
  for (const s of seqs) cases.push([s[0], s[1]]);

  const ref = pythonRef(cases);
  check("charset matches Python", ref.charset === JS.CHARSET,
        `py=${ref.charset} js=${JS.CHARSET}`);
  check("blank index matches", ref.blank === JS.BLANK,
        `py=${ref.blank} js=${JS.BLANK}`);
  check("charset is 37 symbols", JS.CHARSET.length === 37, `got ${JS.CHARSET.length}`);
  check("charset has no punctuation",
        !/[.,;:!?()[\]{}@#$%&*+/\\"'=-]/.test(JS.CHARSET));

  console.log("\ngreedy decode agrees case-for-case:");
  for (let i = 0; i < cases.length; i++) {
    const [seq, limit] = cases[i];
    const js = limit === null
      ? JS.decodeGreedy(seq, undefined)
      : JS.decodeGreedy(seq, limit);
    const py = ref.decoded[i];
    check(`case ${i} ${JSON.stringify(seq)}${limit ? ` limit=${limit}` : ""}`,
          js === py, `js=${JSON.stringify(js)} py=${JSON.stringify(py)}`);
  }

  console.log("\ntimestep contract:");
  // Mirrors ocr_model.py: conv output width is floor(W/8) for even W.
  for (const w of [32, 64, 100, 128, 160, 200, 256, 320]) {
    const t = JS.timestepsFor(w);
    // W/8 with MaxPool2d flooring. For even W this is exact; for odd W the
    // Python side floors too, so compare against floor(w/8).
    const expected = Math.max(1, Math.floor(w / 8));
    check(`timestepsFor(${w}) = ${expected}`, t === expected,
          `got ${t}`);
  }
  check("EXPORT_W is 256", JS.EXPORT_W === 256, `got ${JS.EXPORT_W}`);
  check("CROP_H is 32", JS.CROP_H === 32, `got ${JS.CROP_H}`);

  console.log("\nthe specific case that motivated the gate:");
  check("'900' survives (blank-separated)",
        JS.decodeGreedy([10, 0, 1, 0, 1]) === "900",
        `got ${JSON.stringify(JS.decodeGreedy([10, 0, 1, 0, 1]))}`);
  check("adjacent '00' collapses (CTC-correct, NOT a bug)",
        JS.decodeGreedy([10, 0, 1, 1]) === "90",
        `got ${JSON.stringify(JS.decodeGreedy([10, 0, 1, 1]))}`);
  check("limit prevents reading padding",
        JS.decodeGreedy([10, 0, 1, 0, 1, 0, 7, 0, 7], 5) === "900",
        `got ${JSON.stringify(JS.decodeGreedy([10, 0, 1, 0, 1, 0, 7, 0, 7], 5))}`);

  console.log(`\n${checks - fails}/${checks} checks passed`);
  if (fails) {
    console.log("RECOGNIZER PARITY FAILURE — do not ship this decoder");
    process.exit(1);
  }
  console.log("RECOGNIZER PARITY OK");
})();