# SightLine — is the failure in the RESTORE or the READ?

Plan written 2026-10-20 16:59 CST. Repo: `/home/ubuntu/Projects/sightline`.
`main` = `63ebb1e`, `origin/gh-pages` = `f566df4`. A retrain is already running in
tmux (`retrain`) — do **not** restart it (see T5).

## Answer, in one line

**The READ.** The recogniser is the dominant cause; the restore is not the cause
but it is a measured *net negative*, and today the app can only ever feed the
recogniser its two worst-measured configurations.

## The evidence

| measurement | value | what it isolates |
|---|---|---|
| crop-level, our documents, **no app preprocessing at all** (tight boxes, held-out, `bash run.sh eval-render --split test`) | **4.4% exact, CER 0.53**, 8.3% within 1 edit | the READ. Nothing downstream of the crop is involved, and it still fails |
| same crops, error *kind* | `PRESCRIPTION` → `PRE5CR1P710N`, `DATE01102026` → `0A7E01102026` | the model learned SROIE receipt glyph shapes and **digit-ises our letters** (S→5, T→7, O→0, I→1, G→6). No denoiser or threshold can fix a character-identity error |
| arms table at document level on our rendered corpus, same checkpoint (`artifacts/eval_results.json`, `"engine": "crnn"`) | raw **5.6% field / 26.9% word** · classical 5.6/23.9 · restorer 7.4/26.4 · **restorer_clahe 3.7/23.5** | the RESTORE. The spread is 3.7 points and the *best* preprocessing is **none**; CLAHE is the **worst** |
| `app/tools/restore_ab.js` (independent harness, pre-existing) | "Restoration DECREASES total OCR yield — it is destroying legibility" | agrees with the arms table |
| `app/js/pipeline.js:709-713` (read today) | `const enhanced = clahe(restored); const binary = binarize(enhanced); const candidates = [enhanced, binary];` | the app applies CLAHE **unconditionally** and offers the engine only CLAHE or CLAHE+Otsu — **the best-measured input (raw) is not reachable at all** |
| `estimateSkew` (already fixed and shipped in v1.1.5) | returned −10 on blank, −8 on an axis-aligned page, −7 on a 4° page | a real restore-stage bug: every scan was rotated by a false angle. Fixed and pinned in `selftest.html` |

So: **restore is not why the words are wrong, and fixing restore will not make
them right.** The read has to be retrained on the fonts and layouts we serve
(T5, already running). Separately, the restore stage is costing us points today
and is a ~10-line fix (T2–T4) that uses a seam that already exists.

## Architecture / proposed approach

Two independent changes with separate gates. (1) Make the app offer the engine the
**unprocessed deskewed grayscale** as a candidate alongside the CLAHE and binary
ones, and let the existing `structure_score` ranking choose — no new machinery, the
`candidates` array is the seam. (2) Fix the **read** by training on our own
rendered documents (already launched), then re-export the ONNX and deploy only if
the held-out numbers move. Every step is verified by a number that exists before
the change.

---

## Step-by-step tasks

Run everything from `/home/ubuntu/Projects/sightline`. TDD per task: RED first,
then the minimal change, then GREEN, then commit.

### T1 — Record the two baselines in one place (3 min)

1. Run and keep the outputs:
   ```bash
   bash run.sh eval-render --split test      # the READ baseline
   bash run.sh bench-hackathon               # the end-to-end baseline (writes artifacts/hackathon_metrics.json)
   ```
2. **Expected:** `eval-render` prints `ALL 4.4% exact / CER 0.53 / 8.3% <=1 edit`
   over 781 crops; `bench-hackathon` prints
   `field recovery 0.0% · classification 33.3% · word acc 3.6% · CER 70.8%`
   (if `artifacts/pipeline_raw.json` is stale, run `bash run.sh bench-pipeline` first, ~4 min).
3. Write both into the commit message of T3. No file changes in this task.

### T2 — RED: the app must offer the unprocessed grayscale to the recogniser (5 min)

1. Append to `app/tools/ocr_input_contract.js` (create it if it does not exist —
   it is specified in the earlier plan
   `.hermes/plans/2026-10-20_164918-sightline-ocr-accuracy.md`, T3):

```js
// THE BEST-MEASURED INPUT WAS NOT REACHABLE. In the arms table
// (artifacts/eval_results.json, engine "crnn") the raw deskewed grayscale scores
// best (5.6% field / 26.9% word) and CLAHE worst (3.7 / 23.5), yet
// app/js/pipeline.js offered the engine only [clahe, clahe+otsu].
const pipelineSrc = fs.readFileSync(path.resolve(__dirname, "..", "js", "pipeline.js"), "utf8");
check("preprocess offers the raw deskewed grayscale as a candidate",
  /const candidates = \[raw, enhanced, binary\]/.test(pipelineSrc),
  "app/js/pipeline.js still builds candidates without the unprocessed image");
```

2. Run: `node app/tools/ocr_input_contract.js`
3. **Expected (RED):** `FAIL  preprocess offers the raw deskewed grayscale as a candidate`, exit 1.

### T3 — GREEN: add the raw candidate (5 min)

1. In `app/js/pipeline.js`, replace this block (currently lines ~703-713):

```js
    let restored = deskewed;
    if (restoreMode === "classical") {
      restored = unsharp(median3(deskewed), 1.8, 2);
    } else if (restoreMode === "adaptive") {
      if (q.noisy) restored = median3(restored);
      if (q.blurry || q.noisy) restored = unsharp(restored, 1.8, 2);
    }

    const enhanced = clahe(restored);
    const binary = binarize(enhanced);

    const candidates = [enhanced, binary];
```

with:

```js
    let restored = deskewed;
    if (restoreMode === "classical") {
      restored = unsharp(median3(deskewed), 1.8, 2);
    } else if (restoreMode === "adaptive") {
      if (q.noisy) restored = median3(restored);
      if (q.blurry || q.noisy) restored = unsharp(restored, 1.8, 2);
    }

    // THE UNPROCESSED IMAGE IS A CANDIDATE, and it used to be unreachable.
    // Measured on our own rendered documents with the shipped checkpoint
    // (artifacts/eval_results.json, engine "crnn"): raw 5.6% field accuracy /
    // 26.9% word, classical 5.6 / 23.9, restorer 7.4 / 26.4, restorer_clahe
    // 3.7 / 23.5. The app applied CLAHE to everything and offered the engine
    // only [clahe, clahe+otsu] -- its two WORST inputs. Deskew stays in all
    // three: rotation is geometry, and estimateSkew was fixed to stop rotating
    // by a false angle (see selftest.html, "Skew estimation").
    const raw = deskewed;
    const enhanced = clahe(restored);
    const binary = binarize(enhanced);

    const candidates = [raw, enhanced, binary];
```

2. Keep `canvas: enhanced` in the returned object unless
   `grep -n "\.canvas" app/js/pipeline.js app/js/app.js` shows the preview needs the
   *unprocessed* one; if it does, change it too and say so in the commit.
3. Also fix the arm labels, which currently claim the second candidate is
   `restorer_clahe` in both slots:

```js
    const arms = [
      restoreMode === "classical" ? "classical" : "restorer_clahe",
      "restorer_clahe",
    ];
```
becomes
```js
    // Reporting names, in candidate order: raw, the mode's restored image, then
    // the CLAHE+Otsu one. The old two-entry list mislabelled the second slot.
    const arms = [
      "raw",
      restoreMode === "classical" ? "classical" : "restorer_clahe",
      "restorer_clahe",
    ];
```
4. Verify:
   ```bash
   node app/tools/ocr_input_contract.js     # expect "OCR INPUT CONTRACT OK"
   node app/tools/crop_contract.js          # expect "CROP CONTRACT OK" (9/9)
   node app/tools/ios_full_check.js --engine=webkit --suites=boot   # expect 11/11
   ```
5. Commit:
   `git add -A && git commit -m "preprocess: offer the recogniser the raw image, the best-measured input"`

### T4 — Measure the restore change; keep it only if the number moves (8 min)

1. ```bash
   bash run.sh bench-pipeline        # ~4 min, 12 samples in WebKit
   bash run.sh bench-hackathon
   ```
2. **Decision rule:** compare `headline` against T1's baseline
   (`0.0% fields / 33.3% classification / 3.6% word / 70.8% CER`).
   - Positions recovered, or word accuracy up ≥1 point → keep, and record the
     before/after in the commit.
   - No movement → the ranking is picking CLAHE anyway. Check which candidate won
     (add `psmRank` to the bench record in `app/tools/ios_full_check.js`'s `bench`
     suite, one line: `psm_rank: r.lastResult ? r.lastResult.psmRank : null`), re-run,
     and if candidate 0 (raw) is never chosen, the fix belongs in the ranking, not
     the candidate list — stop, and say so rather than tuning blind.
3. Commit either way, with the numbers in the message.

### T5 — The READ fix is already running: collect it when it lands (10 min + waiting)

A tmux session named `retrain` is training the recogniser on SROIE **plus 25k
crops rendered from our own documents** (`ml/tools/retrain_rendered.sh`,
checkpoint backed up to `models/ocr/crnn_before_render.pt`).

1. Check progress (do not restart it):
   ```bash
   tmux capture-pane -p -t retrain | tail -5
   tail -f /tmp/retrain_render.log
   ```
2. When it finishes, score the new checkpoint against T1's numbers:
   ```bash
   bash run.sh eval-render --split test      # expect > 4.4% exact, and the S/T/O/I
                                             # digit-confusions to be gone from the misses
   bash run.sh eval-ocr                      # expect no regression on real photos
   ```
3. **Decision rule:** crop exact match above 4.4% **and** zero real-photo
   regression → keep `models/ocr/crnn.pt` and go to T7. Worse on either → restore
   the backup (`cp models/ocr/crnn_before_render.pt models/ocr/crnn.pt`) and report
   what the training did, with its log.

### T6 — Read-side tensor padding: paper, not mid-grey (10 min)

1. RED — append to `app/tools/ocr_input_contract.js`:

```js
// The tensor tail must be PAPER (+1), not mid-grey (0): the model was trained on
// paper-backed crops, and buildInput's Float32Array starts zeroed.
const recSrc = fs.readFileSync(path.resolve(__dirname, "..", "js", "recognizer.js"), "utf8");
const buildInputBody = recSrc.split("function buildInput")[1].split("return padded")[0];
check("buildInput pads with paper, not 0 (mid-grey)",
  /fill\(1\)/.test(buildInputBody),
  "app/js/recognizer.js leaves the padding at 0");
```
2. Run `node app/tools/ocr_input_contract.js` → **RED** on that check.
3. GREEN — in `app/js/recognizer.js`'s `buildInput`, change
   `padded.fill(0);           // clear any previous content` to
   `padded.fill(1);           // PAPER, not 0: 0 is mid-grey and the model was trained on paper`.
4. Re-run T4's bench; keep only if it does not regress, and record the number.

### T7 — Land it (20 min)

1. Only if T5 kept the retrained model: re-export and check the payload
   ```bash
   bash run.sh export
   bash run.sh bundle-check        # expect "BUNDLE CHECK PASSED"
   node app/tools/recognizer_parity.js   # expect 24/24
   ```
2. Update the numbers in `docs/14_HACKATHON_BENCHMARK.md`,
   `docs/03_RESULTS.md` §3, `HANDOFF.md` §5 and `docs/12_HACKATHON_SUBMISSION.md`
   §3. Never leave a stale figure: `grep -rn "4.4%\|3.6%" docs/ HANDOFF.md README.md`.
3. Bump `app/sw.js` `const VERSION = "v1.1.5";` → `v1.1.6` and the matching string
   in `app/tools/ios_full_check.js` (`sightline-v1.1.5`), because `app/js/*` is
   precached. Verify: `node app/tools/ios_full_check.js --engine=webkit --suites=boot` → 11/11.
4. Gates: `bash run.sh test` (expect `312 passed, 1 xfailed`),
   `bash run.sh bundle-check`, `node app/tools/crop_contract.js`,
   `node app/tools/ocr_input_contract.js`,
   `node app/tools/ios_full_check.js --engine=both`.
5. Push and deploy without checking out `gh-pages`:
   ```bash
   git push origin pwa:main
   export GIT_INDEX_FILE=$(mktemp); git read-tree origin/main:app; TREE=$(git write-tree)
   NEW=$(git commit-tree "$TREE" -p "$(git rev-parse origin/gh-pages)" -m "Deploy main:app")
   [ "$(git rev-parse "$NEW^{tree}")" = "$(git rev-parse origin/main:app)" ] && git push origin "$NEW":refs/heads/gh-pages
   ```
6. Verify live after the Pages build (~1–30 min):
   ```bash
   ph() { gh api repos/adeviwon/SightLine/pages/builds/latest | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'],(d.get('commit') or '')[:8])"; }
   curl -s "https://adeviwon.github.io/SightLine/sw.js?nocache=$(date +%s)" | grep -o 'const VERSION = "[^"]*"'
   ```
   Expect `built` + the new commit, and `v1.1.6`.

## Tests / validation

- **The read is validated by** `bash run.sh eval-render --split test` (crop-level,
  held out by page, labels drawn not detected) and `bash run.sh bench-hackathon`
  (end to end, WebKit, `headline` over 12 samples with its per-profile table).
- **The restore change is validated by** the same bench, because the candidate
  ranking only matters in the browser; plus `app/tools/ocr_input_contract.js`
  (the candidate list and the tensor padding) and `app/tools/crop_contract.js`
  (crop geometry).
- Never compare a crop-level number with a document-level one: 4.4% crop exact and
  3.6% document word accuracy are different measurements of different things.
- Every number that goes into a doc must come from an artifact
  (`artifacts/hackathon_metrics.json`, `artifacts/eval_results.json`,
  `artifacts/pipeline_raw.json`).

## Risks, tradeoffs, and open questions

- **The ranking may keep picking CLAHE.** `structure_score` rewards text-like
  output and can prefer a longer, wrong read over a shorter, right one. If T4 shows
  candidate 0 never wins, the fix is in the ranking (or force raw for the CRNN) —
  that is T4 step 2, and it must be a measurement, not a preference.
- **The retrain may overfit the renderer.** Mitigated by training on SROIE as well
  (both sources), and by validating on real photos (`bash run.sh eval-ocr`) before
  shipping. The failure mode to watch: crop accuracy up on renders, line accuracy
  on the 198 hand-photographed receipts down.
- **CPU-only training** is ~3 h for 30 epochs on 4 cores; the tmux session is the
  only safe place for it (a Hermes background process is killed at ~600 s).
- **`canvas: enhanced` may be load-bearing** for the preview/PDF path; T3 step 2
  checks before changing it.
- **Open question.** The learned restorer (`app/models/restorer.onnx`, 0.15 MB) is
  **not used anywhere in the pipeline** — `preprocess()` only does classical ops +
  CLAHE. If T4 keeps the raw candidate and CLAHE proves harmful, the candidate list
  should probably drop to `[raw]` and the restorer become an explicit, measured
  upgrade instead of dead weight. Decide with the arms table, not by preference.
- **Open question for the user.** If the retrained model still cannot clear a
  usable field-recovery bar on degraded captures, is the Tesseract arm (62.2%
  field accuracy, ~41 MB, offline, `--engine=tesseract` already in the evaluator) an
  acceptable shipped fallback while the CRNN keeps improving? There is no size
  budget any more, so this is now a pure accuracy/latency tradeoff.
