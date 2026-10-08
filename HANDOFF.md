# HANDOFF — picking this repo up

You are taking over SightLine: an offline PWA that photographs a document,
detects text lines, recognises them with a CRNN we trained ourselves, classifies
the text with MiniLM + a small head, and speaks the result. Everything runs in
the browser; nothing leaves the phone.

Read this first, then `README.md`, then `docs/12_HACKATHON_SUBMISSION.md`.

---

## 1. Run it in 60 seconds

The browser models are committed. Nothing below is needed to *use* the app:

```bash
git clone https://github.com/adeviwon/SightLine.git
cd SightLine
python3 -m http.server 8000 -d app      # then open http://localhost:8000
```

(Any static server works; the PWA needs `python3` only for this line. Camera
permission needs `localhost` or HTTPS — that is a browser rule, not ours.)

**Branch note:** `main` is the current project. This repo's `master` holds an
earlier, different prototype — a Python CLI under `src/offscan/` with its own
OCR/NER/TTS pipeline, from October 1st. It is superseded by this architecture,
but its `docs/VICTORY_PLAN.md` may still be useful framing.

## 2. Train the OCR model from scratch

Fresh clone → trained model → exported browser graph, in order. Budget ~3.5 h
on a 4-core CPU laptop; **no GPU anywhere in this project**.

```bash
bash run.sh fetch-data          # SROIE + CC-BY-4.0 handheld photos (~525 MB, gitignored)
bash run.sh build-data          # 52,554 crops, split BY DOCUMENT IDENTITY
bash run.sh train-ocr 40        # ~250 s/epoch -> ~2.8 h; writes models/ocr/crnn.pt
bash run.sh eval-ocr            # word-acc / CER on held-out receipts
bash run.sh export              # ONNX + parity checks + padding-invariance gate
bash run.sh bundle-check        # real inference through every shipped model
bash run.sh test                # 278 tests
```

Each command fails loudly if the previous step's output is missing. Do not skip
`bundle-check`: it exists because the app once shipped Tesseract while its
manifest claimed it did not.

**Where the models live:**

| file | what | committed? |
|---|---|---|
| `models/ocr/crnn.pt` | the trained recogniser (942,166 params, epoch 39) | yes |
| `app/models/crnn.onnx` | browser recogniser, 4,029,084 B | yes |
| `app/models/minilm_encoder.onnx` | MiniLM int8, 22.9 MB | yes |
| `app/models/minilm_head.onnx` | our 49,796-param head | yes |
| `artifacts/ocr_train.npz` | training cache (169 MB) | **no** — regenerable |

## 3. The real-photo extension (the open task)

We train on SROIE receipts. We ALSO have 193 hand-photographed pages (Zenodo
13688441, CC-BY-4.0) with whole-page transcripts — but no line boxes, so a
crop's label is only trustworthy if we know which transcript line it came from.

The builder pairs bands to transcript lines **positionally** and only keeps
pairs whose count matches the detector's output. Measured on those 193 pages:
**2,661 candidate pairs, 2,301 survive the quality gates** (median 22.7 px per
character, floor is 12). The committed cache `artifacts/ocr_handheld.npz` was
STALE at 76 crops — a rebuild yields 2,301:

```bash
bash run.sh py ml/src/build_handheld.py          # -> artifacts/ocr_handheld.npz
```

Then train on both caches (train_ocr concatenates them and remaps splits):

```bash
bash run.sh train-ocr 8 --data artifacts/ocr_train.npz,artifacts/handheld_rebuilt.npz
```

**The honest caveat:** 167/187 pages have a band/line count mismatch, so those
positional pairs are guesses and some fraction of the 2,301 labels are wrong.
`ml/experiments/gate_breakdown.py` re-derives every number above.

**The experiment in flight when this was written:** `ml/experiments/eval_realphoto.py`
compares 8-epoch SROIE-only (A) vs 8-epoch SROIE+real-photos (B) on 308
held-out real-photo crops. Equal epochs matters — comparing B against the
shipped 40-epoch model would conflate data with training time. Both checkpoints
were saved to `/tmp` (`crnn_A.pt`, `crnn_B.pt`), which is volatile; rerun if gone.

## 4. What would actually move the needle, ranked

1. **A medicine-label corpus.** Everything measured was measured on receipts.
   The hackathon submission says this in its own words — transfer to medicine
   labels is an assumption, not a result. Consent photographs + line labels is
   the whole task.
2. **Chinese in the output alphabet.** `realdata.CHARSET` is
   `'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '` — 37 symbols, **zero CJK**. A HK
   medicine tool that cannot read Chinese cannot serve most of its users. This
   is a data problem (~3–4k common characters), not an architecture problem.
3. **The line detector.** Real-photo recall is 67.9–71.6% and line-exact
   accuracy is 1.8–2.4%. Three of twenty pages are resolution-bound (12–17 px
   lines vs a 32 px input) and no detector can fix those. `docs/03_RESULTS.md`
   documents six attempted fixes; the best one improved the ceiling and made
   CER worse, so we reverted it, and McNemar on 290 lines says the variants are
   indistinguishable (p = 1.000).

## 5. Which numbers you can trust

Everything in `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md` appendix A is labelled:
**session-verified** (re-run it, it reproduces) vs **carried forward** (plausible
but from an earlier state of the code). The submission doc
(`docs/12_HACKATHON_SUBMISSION.md`) extends this with the classifier metrics:
**accuracy 95.8%, macro-F1 95.7% across 5 seeds** (`ml/experiments/step3_f1.py`)
— and records that the stored evaluation used to claim 100%, which was one
lucky seed. Trust the rerun.

## 6. House rules

- `bash run.sh <command>` for everything; it strips PYTHONPATH/VIRTUAL_ENV.
- Tests are the contract: `bash run.sh test` must stay green. `bundle-check`
  failing the build on Tesseract's return is deliberate, not bureaucratic.
- Datasets are licensed (CC-BY-4.0) and gitignored; `fetch_data.sh` fetches and
  cites. Never commit `data/` or `artifacts/*.npz`.
- When you measure something, write down what you measured and what you did
  NOT. The submission's credibility comes from its own corrections — the 100%
  seed, the reverted detector fix, the inverted safety gate. Keep that habit.