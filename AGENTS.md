# AGENTS.md — operational contract for any AI agent taking over this repo

You are taking over SightLine: an offline PWA that photographs a document,
detects text lines, recognises them with a CRNN we trained ourselves
(942,166 params, CTC), classifies the text with frozen MiniLM + a 49,796-param
head, and speaks the result. Everything runs in the browser; nothing leaves the
phone. The hackathon context (Imperial College London, Hong Kong) and the
five-step brief are answered in `docs/12_HACKATHON_SUBMISSION.md`.

**Read order:** this file → `HANDOFF.md` (human narrative + commands) →
`docs/12_HACKATHON_SUBMISSION.md` (the brief) → `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`
(architecture) → `docs/03_RESULTS.md` (numbers).

## Environment

- All entry points: `bash run.sh <command>`. It strips PYTHONPATH/VIRTUAL_ENV
  and uses the uv venv. Do not bypass it.
- **No GPU.** 4 CPU cores, ~7 GB RAM, ~250–300 s/epoch. A full 40-epoch train is
  ~2.8–3.3 h. Budget accordingly; run long jobs in background and verify the
  log afterward (a "completed" notification is not proof — check epoch lines).
- Datasets are **not committed** (169 MB + 524 MB). `bash run.sh fetch-data`
  fetches them (SROIE + Zenodo 13688441, both CC-BY-4.0) into `data/`.
- `/tmp` prunes in 24 h. Anything you leave there (checkpoints, caches) is
  temporary; the durable measurement scripts live in `ml/experiments/`.

## Verified state (as of this file's commit)

Everything below was measured, not assumed:

- `bash run.sh test` → **278 passed, 1 xfail(strict)** (a resolution-bound page
  pinned as xfail, not a lie).
- `bash run.sh bundle-check` → passes; **fails the build if any Tesseract
  component returns** (§4b). That gate is deliberate, not bureaucratic: the app
  once shipped Tesseract while its manifest claimed otherwise.
- Browser recogniser `app/models/crnn.onnx`: **4,029,084 bytes**, epoch-39
  export, padding-invariance verified on real crops ('1071', '01143008').
- Classifier, 5 seeds, held-out n=24: **accuracy 95.8% (sd 2.63), macro-F1
  95.7% (sd 2.72)** — `ml/experiments/step3_f1.py`. The stored eval once said
  100%; that was one lucky seed. The correction is in the submission doc on
  purpose.
- Handheld real photos (20 pages): recall **67.9–71.6%**, line-exact
  **1.8–2.4%**, CER **56.5–58.4%** depending on localisation. Detector variants
  are statistically indistinguishable (McNemar p = 1.000, n≈290 lines).
- Real-photo training cache: the committed `artifacts/ocr_handheld.npz` was
  **stale at 76 crops**; a rebuild yields **2,301** (1637/356/308):
  `bash run.sh py ml/src/build_handheld.py`

## Standing rules (each exists because breaking it hurt)

1. Tests green, bundle-check passing — never weaken a gate to make a run pass.
   If a gate is wrong, fix the gate and say why in the commit.
2. Splits are **by document identity, never by crop**. Random crop splits leak
   receipts across train/test and inflate scores.
3. Report detector recall separately from recogniser CER/line accuracy. A
   rising proxy (bands/line) with a falling product metric (CER) is a
   regression — we reverted such a "win" once.
4. Omit uncertain fields rather than guess; the product refuses fields it is
   not confident about. Never present a confident wrong number.
5. Record what you did **not** measure alongside what you did. The submission's
   credibility comes from its own corrections.
6. Do **not** claim medicine-label capability: no medicine corpus exists.
   Everything measured is receipts + synthetic documents. Transfer is an
   untested assumption (the submission says so verbatim).
7. Git: `origin` = `adeviwon/SightLine`, current work on `main`.
   **`master` holds an earlier, different prototype** (`src/offscan/` Python
   CLI, 15 commits). Preserve it; do not force-push over it, do not merge it.
8. Before trusting a 404 from the GitHub API, check who the token is:
   `gh api user --jq .login`.

## Known traps

- **Multi-cache training ends in `KeyError: 'receipts'`** when the handheld
  cache is concatenated (SROIE schema carries a `receipts` array, handheld does
  not). The KeyError hits *after* per-epoch saves: the checkpoints are still
  valid — recover from `models/ocr/crnn.pt` before anything overwrites it.
- `ml/experiments/eval_realphoto.py`: model output has no batch dim —
  `pred.argmax(-1)[0]` grabs a timestep. Use `argmax(-1).numpy().reshape(-1)`.
  The venv has no `Levenshtein`; the script uses a stdlib DP instead.
- Band↔label pairing is **positional**: 167/187 real-photo pages have a
  band/line count mismatch, so some of the 2,301 cache labels are wrong. That
  is a known, accepted cost — do not silently relabel; measure it instead.
- Three of twenty real-photo pages are **resolution-bound** (12–17 px lines vs
  a 32 px input): no detector can fix them. Don't chase them.

## Open work, ranked (with what "done" means)

1. **Medicine-label corpus** — consent photographs + line labels. Done = a
   cache in the same npz schema, split by document identity, with a documented
   consent/licence trail. This is the actual barrier to the product claim.
2. **Chinese in the output alphabet** — `realdata.CHARSET` is
   `'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '` (37 symbols, zero CJK, no
   lowercase). Done = CJK-capable charset + retrained model + per-language
   metrics that pass a release floor. Without this the tool cannot serve most
   of Hong Kong's over-65 population.
3. **Line detector** — recall 67.9–71.6%. Done = recall gain **and** CER
   non-regression **and** McNemar significance on aligned lines. A proxy
   improvement alone does not count.
4. **Real-photo A/B at full length** — the 8-epoch comparison (A: SROIE-only
   92.7% CER, B: +2,301 real crops 95.1% CER on 308 held-out crops) is
   ambiguous because both are weak; the shipped 40-epoch control through the
   same script is pending. Done = 40-epoch A and B, same held-out split,
   plus the control row. `ml/experiments/eval_realphoto.py` runs all three.