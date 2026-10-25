/**
 * word_lm_parity.js — does the BROWSER's word-LM decoder return the same string
 * as the Python one, on the shipped recogniser's own logits?
 *
 *     node app/tools/word_lm_parity.js
 *
 * Exits non-zero on any mismatch, and prints the crop and both strings so a
 * failure names a word instead of an array index.
 *
 * WHAT IS ASSERTED, AND WITH WHAT TOLERANCE
 * -----------------------------------------
 * 1. `beamDecode(logprobs)` must equal the Python `expected` string EXACTLY, for
 *    every row. Both sides get identical input, so no tolerance is honest here.
 * 2. The browser's log-softmax over the raw `logits` must match the Python
 *    logprobs within 1e-5 max abs difference (torch float32 vs Math.log in
 *    float64). An exact demand would fail on rounding, not on a bug — and the
 *    end-to-end check that matters is #3:
 * 3. log-softmax(raw logits) then beamDecode must still reproduce `expected`.
 *    This is the path the APP runs, so it is the path that decides.
 *
 * Skips (exit 1, not 0) if the fixtures are missing: a gate that passes because
 * its input is absent is decoration.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const FIX = path.join(__dirname, "fixtures", "wordlm_vectors.json");
const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
const BLANK = 0;

async function main() {
  if (!fs.existsSync(FIX)) {
    console.error(`FAIL no fixtures at ${FIX} — run:`);
    console.error("  bash run.sh py ml/src/gen_wordlm_vectors.py --n 60");
    process.exit(1);
  }
  const vectors = JSON.parse(fs.readFileSync(FIX, "utf8"));
  const mod = await import(path.join(ROOT, "app", "js", "word_beam.js"));
  const { beamDecode, WordTrie, logSoftmaxRow } = mod;
  const trie = new WordTrie(JSON.parse(
    fs.readFileSync(path.join(ROOT, "app", "data", "word_trie.json"), "utf8")));
  const { beam, alpha } = vectors.meta;

  const b64 = (s) => new Uint8Array(Buffer.from(s, "base64"));
  const asF64 = (u8) => new Float64Array(u8.buffer, u8.byteOffset, u8.byteLength / 8);
  const asF32 = (u8) => new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);

  let fails = 0, changed = 0, maxLp = 0;
  for (let i = 0; i < vectors.rows.length; i++) {
    const r = vectors.rows[i];
    const lps = asF64(b64(r.logprobs));
    const got = beamDecode(lps, r.T, r.C, CHARSET,
                           { beam, blank: BLANK, lm: trie, alpha }).text;
    if (got !== r.expected) {
      fails++;
      console.error(`FAIL row ${i} label=${JSON.stringify(r.label)}\n` +
                    `  expected ${JSON.stringify(r.expected)}\n` +
                    `  browser  ${JSON.stringify(got)}`);
    }
    if (r.changed) changed++;

    // the transform the app applies before decoding
    const logits = asF32(b64(r.logits));
    const row = new Float64Array(r.C);
    const mine = new Float64Array(r.T * r.C);
    for (let t = 0; t < r.T; t++) {
      logSoftmaxRow(logits, t * r.C, r.C, row);
      for (let c = 0; c < r.C; c++) {
        const d = Math.abs(row[c] - lps[t * r.C + c]);
        if (d > maxLp) maxLp = d;
        mine[t * r.C + c] = row[c];
      }
    }
    const got2 = beamDecode(mine, r.T, r.C, CHARSET,
                            { beam, blank: BLANK, lm: trie, alpha }).text;
    if (got2 !== r.expected) {
      fails++;
      console.error(`FAIL row ${i} (logits path) label=${JSON.stringify(r.label)}\n` +
                    `  expected ${JSON.stringify(r.expected)}\n` +
                    `  browser  ${JSON.stringify(got2)}`);
    }
  }

  const n = vectors.rows.length;
  console.log(`word-LM parity: ${n} crops, checkpoint ${path.basename(vectors.meta.ckpt)}, ` +
              `beam=${beam} alpha=${alpha}, trie ${vectors.meta.trie_words} words`);
  console.log(`  decoder identical:      ${fails === 0 ? "YES" : "NO"} (${n}-${fails}/${n})`);
  console.log(`  log-softmax max |Δ|:    ${maxLp.toExponential(2)} (tolerance 1e-5)`);
  console.log(`  LM changed the read on: ${changed}/${n} crops`);
  if (!(maxLp < 1e-5)) {
    console.error(`FAIL log-softmax differs by ${maxLp} — the app's transform is not ` +
                  `the one the vectors were built with`);
    fails++;
  }
  if (fails) {
    console.error(`PARITY FAILED (${fails} checks)`);
    process.exit(1);
  }
  console.log("PARITY OK");
}

main().catch((e) => { console.error("FAIL " + (e && e.stack || e)); process.exit(1); });
