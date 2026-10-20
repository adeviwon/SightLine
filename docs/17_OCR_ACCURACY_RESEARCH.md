# 17 — OCR accuracy research: real-text corpora, the standard recipe, decoding, architecture

> **What this page is.** A cited decision document for the next accuracy attempt on
> the shipped recogniser (`ml/src/ocr_model.py`, 942,166 params, 37-symbol charset,
> CTC). It answers: which real-text corpora can we legally use, what does the
> literature say actually lifts a weak CRNN, what is a beam + LM decoder worth, which
> architecture change to try first, and what this project has already **measured as a
> dead end** so nobody re-runs it.
>
> **How to use it.** Every claim is either marked `[measured here]` with the file it
> came from, or cited with a URL. Nothing is asserted from memory.
>
> Companion tool: `ml/src/fetch_text_data.py` — prints URL / size / checksum / licence
> for a dataset and refuses to download anything whose licence it cannot confirm:
>
> ```bash
> bash run.sh py ml/src/fetch_text_data.py --dry-run --dataset mjsynth
> bash run.sh py ml/src/fetch_text_data.py --yes --dataset sroie
> ```

## 0. The three things to do next, in order

1. **Re-export the ONNX graph at width 320 and finish the `evaluate.py` → CRNN
   migration.** `[measured here]` The export is fixed at 256 px while training clamps
   word crops at 320 and `eval_ocr.py` keeps full width, so a full-width line arrives
   squeezed ~3.5× horizontally (HANDOFF §4, item 0). This is a defect with a known
   cause and no new data or licence. The migration is the second half: every field/
   word number in `artifacts/` and docs 03/04 is still Tesseract's.
2. **Grow a domain-matched corpus and pretrain → finetune on it.** `[measured here]`
   SROIE scans 11.95% CER vs hand-photographed 56.5% CER on the same checkpoint — a
   4.7× domain gap that is the largest weakness in the project. §2 says the licence-clean
   options are narrow (SROIE and the Zenodo handheld set today; TextOCR/CC-BY-4.0 and
   the MIT synthetic *generators* are the only additions worth having), and §3 says how
   to use them.
3. **A lexicon/LM decoder, offline in Python first.** §4 has the published numbers:
   a character-level LM inside beam search buys ~0.3 pp CER, a word-level LM buys
   ~3.5 pp CER and ~19 pp WER on IAM. For a 37-symbol alphabet whose safety-critical
   tokens are numbers and units, the "lexicon" is a domain token set, not a 0.5M-word
   English dictionary.

---

## 1. Where the recogniser is today (all measured in this repo)

| fact | value | source |
|---|---|---|
| architecture | CRNN: 7 conv/GroupNorm blocks (base 16ch) → 2×128 BiLSTM → 38 logits, CTC | `ml/src/ocr_model.py` |
| parameters | **942,166** (CNN 272,944 / BiLSTM 659,456 / head 9,766) | `[measured here]` `bash run.sh py -c` param count, this lane |
| ONNX payload | 3,840 KB, fixed 256 px wide, self-contained | `docs/03_RESULTS.md` §3, §5 |
| training data | 52,554 crops from 420 SROIE receipts, split by receipt (294/63/63) | `docs/03_RESULTS.md` §3 |
| training cost | ~250 s/epoch, CPU, **4 cores, no GPU anywhere** | `docs/03_RESULTS.md` §3 |
| held-out word accuracy / CER (SROIE) | **29.30% / 11.95%** (17,945 test words) | `docs/03_RESULTS.md` §3 |
| corpus-size + homography effect | +3.05 pp accuracy, −3.95 pp CER (epoch 20 → 39) | `docs/03_RESULTS.md` §3 |
| **domain gap** | SROIE scan 11.95% CER vs hand-photographed **56.5% CER** | `docs/03_RESULTS.md` §3 |
| field accuracy, shipped engine | **7.4%** (Tesseract arm on the same generator: 62.2%) | `docs/03_RESULTS.md` §3 |
| end-to-end, real app in WebKit | **12.5%** field recovery, 22.4% word accuracy, 57.5% CER (was 0.0/3.6/70.8 before the export-width fix) | `HANDOFF.md` §5, `docs/14` |
| detector ceiling on real photos | 67.9–71.6% line recall, 1.8–2.4% line exact | `docs/03_RESULTS.md` §3 |

Labels are charset-clean by construction: `render_corpus.train_label` upper-cases and
keeps only `realdata.CHARSET` = `0123456789A-Z␠` (37 symbols, blank index 0, 38 classes),
and `realdata.norm_label` maps glyph confusions (`O→0`, `S→5`, `B→8`) **for scoring
only** — never as a training target. Two label-space decisions are already load-bearing
and must not be undone without measurement:

- punctuation is deliberately *out* of the charset (`realdata.py`: "the highest-error
  class in practice and contributes nothing to any field this app extracts");
- `g`/`q` are deliberately *not* normalised to `9`, because doing so turns the dosage
  `500mg` into `500M9` and breaks the extractor's `\d+mg` pattern — a false negative on
  the most safety-critical token the app reads.

---

## 2. Candidate real-text corpora (licence first — it is a hard gate)

Sizes are as published by the hosting page; `sha256`/`md5` are printed by
`ml/src/fetch_text_data.py`. "Plausible for us" = a 900×580 English document reader,
labels A–Z, 0–9 and space, trained on 4 CPU cores, and the corpus's licence must be
acceptable for a shipped product.

| key | corpus | licence | size | where it comes from | what it contains | plausible for us? |
|---|---|---|---|---|---|---|
| `mjsynth` | MJSynth / Synthetic Word Dataset (Jaderberg et al. 2014) | **Non-commercial research only** — Oxford VGG dataset Terms of Access: 12-month licence, content must not be "incorporated in commercial products" | **~10 GiB** (9M word-box images, 90k English words) | [VGG data/text](https://www.robots.ox.ac.uk/~vgg/data/text/) | synthetic scene-text word crops; font render → border/shadow → background → projective distortion → blending | **No** — licence blocks shipping; synthetic *scene* imaging is the wrong process (`realdata.py`'s argument) |
| `synthtext` | SynthText in the Wild (Gupta et al. CVPR 2016) | same Oxford VGG group-2 terms (non-commercial, 12 months); download links have been **withdrawn** from the site | 800k images, ~8M word instances (~41 GB) | [VGG data/scenetext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) | synthetic words placed into natural scene images with word- and char-level boxes | **No** — licence + availability |
| `iam` | IAM Handwriting Database 3.0 | no open licence text on the database page; distributed for research on registration/agreement → **unconfirmed** | 657 writers, 1,539 pages, 13,353 lines, **115,320 words**, 300 dpi PNG | [FKI Bern](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | handwritten English sentences (LOB corpus) | **No** — handwriting is a different glyph distribution, tiny, licence unconfirmed |
| `iiit5k` | IIIT 5K-Words (Mishra et al. BMVC 2012) | **no licence statement** on the project page (research use implied) | **106 MB** tarball; 5,000 word images (2k train / 3k test) + 50-word and 0.5M-word lexicons | [CVIT IIIT-H page](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) | scene + born-digital cropped words, case-insensitive by construction | **No** — licence unconfirmed (fetcher refuses); scene/logo words, not document text |
| `svt` | Street View Text (Wang et al. 2011) | **no licence text** on the hosting page | 349 images, ~725 words | [UCSD vision](https://vision.ucsd.edu/index.php/datasets/street-view-text) | business signage harvested from Street View, low resolution | **No** — licence unconfirmed; tiny; signage |
| `cute80` | CUTE80 (Risnumawan et al. ESWA 2014) | third-party mirror states no licence, only requests a citation → **unconfirmed** | 80 images, 288 cropped words | [mirror](https://github.com/mohtashim-nawaz/Cute80-Dataset) | curved scene text, no lexicon | **No** — licence unconfirmed; curved display text |
| `textocr` | TextOCR (Singh et al. CVPR 2021) | **CC-BY-4.0** | **272 MB** train + 39 MB val JSON, images 6.6 GB (+926 MB test) — 714,770 train / 107,802 val word annotations, 24,902 images | [textvqa.org](https://textvqa.org/textocr/dataset/) | scene text on OpenImages, case-sensitive, punctuation included, illegible→`"."` | **Partial — the only large licence-clean English word corpus**; needs case-fold, punctuation strip and `"."` filtering to fit our 37 symbols; scene fonts ≠ forms |
| `sroie` | ICDAR2019-SROIE (Huang et al. ICDAR 2019 RRC) | **CC-BY-4.0** as re-hosted by the HF mirror (original competition release has its own terms) | **510 MB** total: train parquet 318,620,215 B `sha256 b18c16b4…3887`, test parquet 191,045,976 B `sha256 04f8f31b…84c2`; 626 train + 361 test receipts | [HF mirror](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | real scanned receipts: per-**word** boxes + transcriptions + key entities | **Yes — what we train on today**; flatbed, but real thermal fade, barcodes, handwriting, ghost print |
| `handheld` | Express Expense (Auad et al. SIBGRAPI 2024), Zenodo 13688441 | **CC-BY-4.0** | **18,998,248 B** (`md5 b7a29ba1cfab963714782e1752ce8e62`), 198 receipts | [Zenodo](https://zenodo.org/records/13688441) | hand-photographed restaurant receipts, whitespace-aligned page transcripts, 4-corner homographies | **Yes** — the only licence-clean *photographed* set, and what `eval_ocr.py` set B scores on; crop-level labels from it were a measured negative (§5) |
| `union14m` | Union14M (Jiang et al. ICCV 2023) | code **MIT**, but the data is consolidated **from 17 public datasets** whose licences differ per source → **per-crop licence unconfirmed** | 4M labelled + 10M unlabelled scene-text crops | [GitHub](https://github.com/Mountchicken/Union14M) | scene text across difficulty levels, incl. a real-data "M" subset | **No** — untraceable provenance/licence; scene text |
| `hiertext` | HierText (Long et al. CVPR 2022) | **CC-BY-SA-4.0** annotations (share-alike); images from OpenImages | 11,639 images, ~1.2M word annotations (+ line and paragraph levels) | [GitHub](https://github.com/google-research-datasets/hiertext) | hierarchical scene/document text; has a *document* subset | **No** — share-alike is a stricter gate than non-commercial for a shipped app; the line/paragraph labels are the one thing we would want |
| `synthdog` | SynthDoG (Donut, Kim et al. ECCV 2022) | **MIT** | no download — generated locally; English alphabet built in | [clovaai/donut](https://github.com/clovaai/donut) | synthetic **documents** (forms, invoices) with text labels, configurable fonts/backgrounds/alphabet | **Yes/cheap** — documents, not scene words; MIT; runs on our CPU |
| `trdg` | TextRecognitionDataGenerator | **MIT** | no download — `pip install trdg` | [Belval/TRDG](https://github.com/Belval/TextRecognitionDataGenerator) | word/line crop generator with your own fonts, lexicon and distortion stack | **Yes/cheap** — closest match to our crop geometry |
| `render_corpus` | in-repo renderer (`ml/src/capture.py` + `render_corpus.py`) | our own code — no gate | 0 bytes downloaded; ~400 pages ≈ 10k crops in minutes on this box | this repo | pages rendered by `capture.render_document`, 9 capture profiles, **labels drawn from the layout** (no detector error can leak into a target) | **Yes — already built**; the only corpus whose label alignment is guaranteed |

Licence gate, stated plainly: **three** of the fifteen rows are usable for a product we
ship (`sroie`, `handheld`, plus our own renderer), two more are usable as *generators*
(`synthdog`, `trdg`, MIT — they generate rather than license data), one is
licence-clean but off-domain (`textocr`, CC-BY-4.0), and the historical scene-text
corpora (`mjsynth`, `synthtext`, `iam`, `svt`, `cute80`, `iiit5k`, `union14m`,
`hiertext`) all fail the gate — non-commercial terms, share-alike, or no licence text
at all. `fetch_text_data.py` refuses the unconfirmed ones outright.

---

## 3. The standard recipe for lifting a weak CRNN

### 3.1 Synthetic pretraining, then domain finetuning — and the measured caveat

Published practice is consistent: pretrain on large synthetic corpora, finetune on real
data. Baek et al. (ICCV 2019) train on MJSynth ∪ SynthText and report the effect of
finetuning on real data: **+2.2 pp on in-distribution benchmarks, −1.3 pp on
out-distribution ones**, concluding "finetuning over real data is effective when the
real-data is close to the test-time distribution. Otherwise, fine-tuning over real data
may do more harm than good"
([arXiv:1904.01906](https://arxiv.org/abs/1904.01906)).

That sentence is the published form of a negative result this repo already has
`[measured here]`: finetuning the shipped model on 2,301 crops bootstrapped from
hand-photographed receipts with *positional* labels dropped line recall 71.6% → 60.7%
and line-exact 2.4% → 1.2% (`HANDOFF.md` §3). The labels were wrong, not the idea.

Two useful corollaries from the same paper, both about *diversity over count*: MJSynth
alone 80.0%, SynthText alone 75.6%, both 84.1% — and 20% of each (2.9M crops) already
beat either alone. And Baek et al. (CVPR 2021) disprove the "real data is not enough"
folklore: real labels only, with **simple augmentation**, gets competitive results
([CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)).

For us, the synthetic half should be **our own renderer** (`render_corpus.py`, MIT-free,
exact labels) plus optionally SynthDoG/TRDG — not MJSynth/SynthText, which we cannot
license and whose imaging process (`realdata.py`) is the wrong one.

### 3.2 Augmentation families that matter for photographed documents

| family | why it matters for a photographed form | source |
|---|---|---|
| perspective / projective warp | off-axis handheld capture; already implemented with **real four-corner homographies from Zenodo** and measured worth **+3.05 pp / −3.95 pp CER** | `[measured here]` `build_dataset.py`, `docs/03_RESULTS.md` §3 |
| motion + defocus blur | the dominant failure on handheld photos | `[measured here]` `build_dataset.py` variants `motion`, `defocus` |
| illumination: dark, bright, glare, hand shadow | `hand_shadow` is the only profile where the restorer moved the Tesseract number (+60 pp) | `docs/03_RESULTS.md` §4 |
| JPEG / social re-compression | `jpeg` variant, quality 25–60 | `[measured here]` `build_dataset.py` |
| ink/paper/digital artefacts (fade, show-through, ringing, speckle, bleed) | the document-specific families this project does **not** cover yet | Augraphy, Groleau et al. ([arXiv:2208.14558](https://arxiv.org/abs/2208.14558)) |
| composition of 2–3 above | what a phone actually produces | `[measured here]` `build_dataset.py` `composite` |

Augmentation is applied **once at build time and cached** in this repo, deliberately, for
reproducibility (`build_dataset.py` header). Keep that: a corpus that changes per epoch
makes the split-leakage and reproducibility gates meaningless.

### 3.3 Curriculum order: clean → degraded

Curriculum learning (present easier examples first) is an established principle
(Bengio et al., ICML 2009, [paper](https://icml.cc/Conferences/2009/papers/119.pdf)),
and the OCR realisation of it that the literature actually validates is the
synthetic→real progression of §3.1: train on clean renders first, then introduce the
degradation profiles, and finetune last on the real capture distribution. Two
project-specific constraints:

- **`worst_case` stays out.** `evaluate.SUB_HUMAN` marks it unreadable to a human; a
  human cannot read it, so training on it teaches guessing (`render_corpus.py`).
- **Do not degrade below the information the photo contains.** Three of the twenty
  handheld pages carry 11.7–17.4 px per line against a 32 px input — 1.84–2.73× upscale
  — so no amount of degradation modelling recovers them (`docs/03_RESULTS.md` §3).

### 3.4 Label-space hygiene (the rules this repo already follows, plus gaps)

- **Case**: upper-case for training targets (`render_corpus.train_label`), and
  case-fold any external corpus (TextOCR annotations are case-sensitive) *before* it
  enters the 37-symbol alphabet.
- **Punctuation**: out of the charset by design; strip it from external labels rather
  than widening the output layer for a class the extractor tolerates.
- **Blank handling**: blank is index 0, and the fixed-width ONNX export is only safe
  because a trained CTC model emits blanks over right-padding — a gate verifies it
  (`docs/03_RESULTS.md` §3, four pad cases, all `ok`).
- **Scoring vs training labels must stay separate**: `norm_label` (O→0 for scoring) vs
  `train_label`/`clean_for_training` (what the model is asked to emit). Training on
  normalised labels teaches the model to corrupt company names (`realdata.py`).
- **Annotation noise**: keep raw labels, clean only the scoring copy
  (`realdata.py`), and filter annotations whose text is not in the alphabet
  (TextOCR's illegible `"."`).

---

## 4. Decoding: what a beam search + LM is actually worth

**Published deltas on CTC recognisers.** The reference measurement is Scheidl et al.,
ICFHR 2018, which decodes the *same* trained network with every decoder
([poster with both tables](https://iapr.org/archives/icfhr2018/SlidesPosters/Poster-55.pdf),
[CER/WER tables in the comparison paper](https://raw.githubusercontent.com/githubharald/CTCDecoder/master/doc/comparison.pdf)):

*IAM, CER% / WER%, "rudimentary LM" (train-set text + 370k word list) → "perfect LM":*

| decoder | rudimentary LM | perfect LM |
|---|---|---|
| best path (greedy) | 8.77 / 29.07 | 8.77 / 29.07 |
| vanilla beam search + **character** LM | 8.48 / 28.24 | 8.27 / 27.34 |
| word beam search (+ **word** LM), best mode | 8.61 / 22.86 | **5.21 / 9.78** |

Bentham: best path 6.76 / 18.00 → word beam search 4.05 / 7.36. Two readings, both
useful here:

1. **A character-level n-gram LM inside beam search is worth ~0.3–0.5 pp CER.** Do not
   spend the complexity there.
2. **A word-level LM (a lexicon + n-gram scores over word completions) is worth
   ~3.5 pp CER and ~19 pp WER** on IAM — and its beam width was **2**. Cost is linear in
   sequence length and quasi-linear in alphabet size and beam width, so a 37-symbol
   alphabet with beam width 2–10 is cheap.

The mechanism is also stated in the original CTC paper: CTC "does not explicitly model
inter-label dependencies", and prefix-search decoding over pure best-path gains only
~1 pp (TIMIT 31.47% → 30.51% LER); the language model is what supplies the dependency
([Graves et al., ICML 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)).

**Practical recipe for our 37-symbol alphabet.** The corpus this product reads is not
English prose: it is field keys, units, dates and account numbers. So:

- **Lexicon**: not a 0.5M-word dictionary (IIIT5K ships one; ours would be useless on
  `40218877`). Use domain tokens — `MG MCG ML G KG`, field keys as printed
  (`DOB`, `TOTAL`, `ORDER`), month/unit abbreviations — plus a **digit-shape prior**
  (digits are digits: `O`/`I`/`S` confusions are the ones that make a dosage wrong).
- **Beam width 2–10**, log-domain scores, LM weight normalised by label length (the
  comparison paper is explicit that un-normalised per-character LM scores bias toward
  short labelings).
- **Prune by product of CTC and LM score, not by CTC alone**, and keep greedy as the
  fallback when no beam beats it (this is exactly the "does it pay" check).
- **Measure before shipping.** Everything above is only worth it if it moves our
  held-out number: score the SROIE test split (`bash run.sh eval-ocr`) with greedy vs
  beam+LM and report both. A decode change that improves CER but not
  **exact-match word accuracy** does not help this product — `eval_ocr.py`'s header
  makes that argument (a falling CER with flat exact-match means one wrong character
  per word: `500mg` → `5O0mg`).
- **The browser is the constraint.** The shipped path is greedy decode in JS
  (`recognizer.js`), mirrored case-for-case against Python's `decode_greedy` by
  `app/tools/recognizer_parity.js` (24/24). An LM decoder is therefore a *Python-side
  experiment first*; port only if the measured gain justifies the second implementation
  and the parity gate.

---

## 5. Architecture options, with measured parameter counts

All counts below are `[measured here]` by instantiating `SightLineCRNN` with the given
kwargs and summing parameters (the constructor already exposes `hidden`, `lstm_layers`,
`channels`, `input_height`). Inference timings are one forward pass, batch 32, width 256,
**1 thread**, on a box that is concurrently running a 4-thread training job — read them
as relative weights, not benchmarks.

| option | params | vs ship | 32×256 input | notes |
|---|---|---|---|---|
| **ship**: ch16, h32, 2×128 BiLSTM | **942,166** | — | 32 timesteps | 3.84 MB ONNX; batch-32 ≈ 760 ms |
| 3-layer BiLSTM (128) | 1,337,430 | +42% | 32 | depth where the recurrent stage is cheap |
| 2×256 BiLSTM | 2,659,926 | +182% | 32 | the capacity jump with the best accuracy/cost reputation in CRNN literature |
| 3×256 BiLSTM | 4,236,886 | +350% | 32 | ≈ 951→1241 ms; ONNX payload would grow past the 4 MB we ship |
| wider CNN (base 32ch) | 1,889,926 | +101% | 32 | doubles CNN features, +50% time |
| one extra 3×3 conv 256→512 | +1,180,672 | +125% | 32 | must be inserted after the last pooling stage to keep width/8 |
| input height 32 → 48 | 942,166 (unchanged) | 0% | 32 | geometry still lands on 1 row; costs +38% time (1049 vs 758 ms) |

Reading the options:

- **Height 32 → 48 is not the lever, and the literature says why.** "Accuracy does not
  increase obviously when the height is larger than 32 pixels… 32 px height is a
  customary rule in scene text recognition" (TextZoom / Wang et al., ECCV 2020,
  [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550647.pdf)).
  Note the failure mode: a 32 px-tall input *scaled from* a 12–17 px line is already an
  upsampling artefact, and upsampling at the input layer cannot create strokes that
  were never captured (`docs/03_RESULTS.md` §3). Height helps only if the *capture*
  resolution rises.
- **Width is the cheap, already-diagnosed win.** The graph is fixed at 256 px while
  training clamps at 320 and evaluation uses full width; re-exporting at 320 is
  ranked item 0 in `HANDOFF.md` and needs no new data.
- **A transformer encoder is not worth it at this data scale.** Our entire corpus is
  52,554 crops from 420 documents (~250 s/epoch on 4 cores); transformer STR models are
  trained on 10⁶–10⁷ crops, and PARSeq reports "using synthetic training data", i.e. a
  14M-crop-scale diet, for its results ([arXiv:2207.06966](https://arxiv.org/abs/2207.06966)).
  Baek et al. also find attention's marginal contribution on these benchmarks largely
  amounts to an implicit character-level LM ([arXiv:1904.01906](https://arxiv.org/abs/1904.01906))
  — which §4 says is the part of decoding worth ~0.3 pp, obtainable far more cheaply
  from a decoder.

**What I would try first, in one paragraph.** Re-export and evaluate at width 320, then
run the same checkpoint with a 2-layer→3-layer BiLSTM at hidden 128 and, separately, a
wider CNN (base 32) — both are +40–100% params and ~+25–50% inference time, which the
payload and 4-core budget can absorb, and both are pure capacity additions on a model
whose held-out word accuracy is 29.30%. Keep input height at 32 and skip the transformer
until there is an order of magnitude more labelled crops. Score every variant on the same
17,945 SROIE test words *and* the rendered corpus, report exact match first and CER
second, and change one thing per run — the epoch-20-vs-39 comparison is the only clean
A/B this project has, and it is the template.

---

## 6. What would NOT help — measured dead ends and unlikely published methods

**Measured in this repo. Do not re-run these.**

1. **Six paper-estimate/segmentation heuristics for the line detector** — adaptive
   valley splitting (0.610 bands/line), Otsu global (0.195), morphological-close
   (0.198), Bradley adaptive (0.242), paper localisation (0.716), page-scaled
   window+margin 45 (0.817, *best*, and still reverted because CER went 56.5% → 61.5%).
   Records live in `ml/src/eval_ocr.py`; the variants are indistinguishable by
   McNemar on 290 lines (p = 1.000) — a sample-size problem, not a tuning problem.
   `test_sample_size_cannot_support_detector_claims` pins this.
2. **Finetuning on crops bootstrapped from hand-photographed pages with positional
   labels.** 71.6% → 60.7% recall, 2.4% → 1.2% line-exact; the `strict_count` variant
   recovers recall but is still no net win (`HANDOFF.md` §3). Needs a properly
   *line-labelled* corpus first.
3. **Chunking long lines into windows.** Tried, read worse (`HSBC UK BANK PLC` 98%
   squeezed vs `MSEC UNBANE L` 54% chunked), reverted; recorded in
   `app/tools/crop_contract.js`.
4. **Image restoration as an OCR boost on the shipped engine.** On the CRNN arms,
   `raw` ties or beats the restorer (`docs/03_RESULTS.md` §3); the Tesseract-era
   "restorer > classical" ordering does not survive the engine change.
5. **Tesseract PSM sweeps.** 336 configurations across 7 modes × all candidates × 12
   samples recovered **no field at all** on `handheld_light` (`docs/10_ACCURACY_GAP.md`
   §3). Tesseract is gone from the app; do not re-open this path.
6. **Detector micro-tuning in general.** The blocker is that the receipt must be
   localised before line finding, and the honest ceiling is data
   (20 hand-photographed pages), not heuristics (`HANDOFF.md` §4, item 3).

**Published methods unlikely to pay off at this scale.**

- **A character-level LM in beam search.** ~0.3–0.5 pp CER on IAM (§4). Not worth two
  decoders to maintain.
- **A large English lexicon.** The gains in §4 come from restricting output to *words*;
  our safety-critical tokens are numbers and unit strings that no dictionary contains.
  `iiit5k` even ships a 0.5M-word lexicon, and it would have nothing to say about
  `40218877`.
- **Training a transformer recogniser from scratch on 52k crops.** See §5; the field
  trains these on 10⁶–10⁷ crops.
- **Swapping in MJSynth/SynthText as the "real data" fix.** Two independent reasons: the
  licence (non-commercial, 12-month, no commercial products) and the imaging process —
  `realdata.py`'s header is a measured argument that synthetic scene rendering does not
  produce thermal fade, ghost print, pink cast or JPEG ringing.
- **Raising input height to fix the three resolution-bound pages.** 12–17 px lines
  cannot be upsampled back into existence; the fix is capture resolution
  (`docs/03_RESULTS.md` §3).

---

## 7. Sources

1. MJSynth / Synthetic Word Dataset — https://www.robots.ox.ac.uk/~vgg/data/text/
2. SynthText in the Wild — https://www.robots.ox.ac.uk/~vgg/data/scenetext/
3. Oxford VGG dataset Terms of Access (non-commercial, 12 months, group-2) — https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html
4. IAM Handwriting Database 3.0 — https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
5. IIIT 5K-Words — https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset
6. Street View Text — https://vision.ucsd.edu/index.php/datasets/street-view-text
7. CUTE80 mirror — https://github.com/mohtashim-nawaz/Cute80-Dataset
8. TextOCR dataset (CC-BY-4.0, sizes) — https://textvqa.org/textocr/dataset/
9. ICDAR2019-SROIE on Hugging Face (CC-BY-4.0, 987 rows, 510 MB) — https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
10. Express Expense / hand-photographed receipts (CC-BY-4.0) — https://zenodo.org/records/13688441
11. Union14M (ICCV 2023) — https://github.com/Mountchicken/Union14M · https://arxiv.org/abs/2307.08723
12. HierText (CC-BY-SA-4.0) — https://github.com/google-research-datasets/hiertext
13. Donut / SynthDoG (MIT) — https://github.com/clovaai/donut
14. TextRecognitionDataGenerator (MIT) — https://github.com/Belval/TextRecognitionDataGenerator
15. Baek et al., *What Is Wrong With Scene Text Recognition Model Comparisons?* (ICCV 2019) — https://arxiv.org/abs/1904.01906
16. Baek et al., *What If We Only Use Real Datasets for Scene Text Recognition?* (CVPR 2021) — https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf
17. Scheidl, Fiel, Sablatnig, *Word Beam Search* (ICFHR 2018) — https://iapr.org/archives/icfhr2018/SlidesPosters/Poster-55.pdf
18. Scheidl, *Comparison of CTC decoding algorithms* (beam search vs token passing, CER/WER tables) — https://raw.githubusercontent.com/githubharald/CTCDecoder/master/doc/comparison.pdf
19. Graves et al., *Connectionist Temporal Classification* (ICML 2006, TIMIT LER table) — https://www.cs.toronto.edu/~graves/icml_2006.pdf
20. Wang et al., *Scene Text Image Super-Resolution in the Wild / TextZoom* (ECCV 2020, "32 px height" finding) — https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550647.pdf
21. Groleau et al., *Augraphy: A Data Augmentation Library for Document Images* (2022/2023) — https://arxiv.org/abs/2208.14558
22. Bengio et al., *Curriculum Learning* (ICML 2009) — https://icml.cc/Conferences/2009/papers/119.pdf
23. Bautista & Atienza, *PARSeq* (ECCV 2022) — https://arxiv.org/abs/2207.06966

In-repo evidence used above: `ml/src/ocr_model.py`, `ml/src/realdata.py`,
`ml/src/render_corpus.py`, `ml/src/build_dataset.py`, `ml/src/eval_ocr.py`,
`ml/src/eval_render_corpus.py`, `docs/03_RESULTS.md`, `docs/10_ACCURACY_GAP.md`,
`HANDOFF.md`, `app/tools/crop_contract.js`, `app/tools/recognizer_parity.js`.
