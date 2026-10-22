# SightLine — Imperial College London Hackathon, Hong Kong & Macau

**Submission.** Structured to the five required steps. Every statistic is
sourced and every metric is measured on held-out data.

**Read this first if you only read one thing:** §6 — four train/serve contract
defects we found in our own code and fixed, each with the number it moved — and
§4, where we found two ways our own tool would exclude the people it is meant to
serve. We did not find any of them by being asked; we found them by reading our
own model config and our own serving path.

**Round 1 deliverables** (150-word proposal + 3–5 minute video, due
2026-10-25): the proposal, a shot-by-shot video list, judge intel and the
dated countdown live in `13_ROUND1_SUBMISSION_KIT.md`; the one-document
judges' brief is `11_JUDGE_TECHNICAL_DEEP_DIVE.md`.

---

## 1. Understand the problem

### The problem we chose

An elderly person in Hong Kong stands in their kitchen with a box of medicine. Two
things stop them reading it: **they can no longer see it clearly, and what is
printed on it is in English they never learned.** 46.4% of older Hong Kong people
take five or more medicines at once, so the label is not one instruction — it is
the only instruction they get, several times over.

**The task we set the AI:** given one photograph of a medicine label, read the drug
name, strength and dose, and **say it back in Cantonese** — on the phone, with no
network.

**Scope, narrowed deliberately (2026-10-22).** This is a medicine-label tool, not a
general document reader. "Reads any document" is a category that already exists;
"a medicine label your grandmother can *hear in Cantonese*, with the network off" is
one specific claim with a named user and two compounding barriers. The full refined
brief is [`docs/26_IDEA_REFINED.md`](26_IDEA_REFINED.md).

**Why this does not need Chinese OCR:** Hong Kong medicine labels carry the drug name,
strength and directions **in English**, which is the `0-9 A-Z` alphabet our recogniser
already has. The gap is on the *output* side — speaking Cantonese — which is a bounded
speech-and-lexicon change, not a 3–4,000-character alphabet expansion. **That change now
ships as a first build** (§2, "the Cantonese output path"): the drug name from a licensed
zh/en lexicon, the dose from the read text, the directions from a closed phrase set, and a
**refusal** for anything it cannot translate. Its three honest limits are in §6 and
[`docs/26_IDEA_REFINED.md`](26_IDEA_REFINED.md) §7 — not buried.

### Who is affected, how many, and what goes wrong

| # | Data point | Population | Source |
|---|---|---|---|
| 1 | **22% of Hong Kong's population is aged 65 or over** — 1.7 million people, up from 15% in 2014 (+55% in nine years) | All HK residents, 2023 | HKSAR Legislative Council Secretariat, *ISSH32/2024*, citing Census & Statistics Department |
| 2 | **74% of Hong Kong's 1.2 million chronic-disease patients were elderly** (2022–23) | HK chronic-disease population | Same source |
| 3 | **46.4% of older Hong Kong people were on five or more medicines simultaneously** — the **highest** of Hong Kong, Taiwan, South Korea, UK and Australia (2016), and **rising at 2.7% per year** | 1.62 million people across 5 countries | Lee H. et al., *Age and Ageing* 2023, doi:10.1093/ageing/afad014 |
| 4 | Medication errors cost an estimated **US$42 billion per year** globally; harm from medicines accounts for **nearly 50% of all preventable harm** in medical care | Global | World Health Organization, *Medication Without Harm* |
| 5 | Macau's elderly population reached **14.0%** in 2023 and **exceeded the youth population for the first time** | Macau residents, 2023 | Macao Government, DSEC demographic statistics 2023 |

**What goes wrong today.** Polypharmacy at 46.4% means a typical patient's
medication list is not "one medicine", it is a *set*. The information is not
missing — it is printed on the box. It goes wrong at the last step, when
somebody has to read a small, low-contrast, possibly-glared label under a
ceiling light, correctly, at 11pm, possibly with failing eyesight and a hand
that shakes. A wrong dose is not an inconvenience; it is a medical event.

### The data our AI needs, and where it comes from

| input | why | source |
|---|---|---|
| Photos of medicine labels | the model reads pixels, not text | **Not yet available.** We trained on ICDAR 2019 SROIE (real receipts) and a CC-BY-4.0 hand-photographed set (Zenodo 13688441). Honest position: **we have no medicine-label corpus at all.** This is our single biggest gap. |
| Medicine label text for supervision | to train recognition | Public formulary text is not a labelled image corpus. The realistic route is the HK Drug Registry + hospital discharge medication lists for the *text*, paired with photographs we would have to collect with consent. |
| Language and speech | so it can speak the answer | English TTS by default, plus a **first Cantonese output build** (`app/js/say_cantonese.js`): the drug name from a licensed zh/en lexicon, the dose from the read text, the directions from a closed phrase set. Refuses a name it cannot translate. Three limits in §6. |

**We will not pretend we have medicine data.** Everything in §3 was measured on
receipts and synthetic documents. Transfer to medicine labels is **an untested
assumption, not a result**, and saying so is more useful to a judge than a
number we cannot defend.

---

## 2. Build the AI tool

Two trained models, both running **entirely in the browser**. No server call.

### Model A — the recogniser (our main contribution)

A **CRNN with CTC loss**, trained from scratch in PyTorch.

- **Why CRNN+CTC.** We read variable-width line crops with **no per-character
  boxes** and must run on a phone CPU. CTC needs no alignment, and greedy CTC
  decoding is **non-autoregressive** — one forward pass, not 32. A transformer
  decoder would autoregress and miss the latency budget.
- **Shape.** CNN (7 blocks, 16→128 channels) → 2-layer BiLSTM (hidden 128) →
  linear → 38 classes. **942,166 parameters.** Training crops are built at
  their NATURAL width clamped to `realdata.MAX_W = 320`, and the corpus median
  IS that clamp; the shipped graph is **dynamic width** — input `[1,1,32,width]`,
  output `[1,T,38]`.
- **Training data.** ICDAR 2019 SROIE — **52,554 crops from 420 receipts**,
  11 augmentation variants including perspective warps driven by **199 real
  hand-held homographies** (Zenodo 13688441, CC-BY-4.0).
- **Split by document identity, never by crop.** 294/63/63 receipts. Randomly
  splitting crops puts images of the same receipt on both sides and inflates the
  score. This is the most common way to fake an OCR result.
- **Cost.** 40 epochs, ~250 s/epoch on **4 CPU cores, no GPU** (~2.8 h total).

### Model B — the document classifier

`sentence-transformers/all-MiniLM-L6-v2` (**frozen**) plus a **49,796-parameter
head** we trained. 4 classes: banking / medical / legal / general.

**Stated precisely because it is easy to overclaim:** MiniLM is a *text* encoder.
It consumes tokens, not pixels. It sits **after** our OCR and never sees an
image. The encoder is **frozen** — this is not end-to-end fine-tuning, and we do
not claim it is.

### The Cantonese output path (shipped — a first build, with limits)

The refined brief is "photograph the label, **hear it in Cantonese**". Our recogniser reads
the label in English, so the output half is a **Cantonese utterance built from facts we hold**,
not a translation of free text:

- **The drug name** is matched against a **licensed zh/en medicine lexicon** —
  `app/data/drug_lexicon.json`, **18,839 English → Cantonese name pairs (1.69 MB)**, sourced
  from **`twinkle-ai/tw-drug-labels-vision` (Taiwan FDA leaflet archive; 72,326 real
  medicine-label images), CC-BY-4.0**, harvested by `ml/tools/hf_lexicon.py`. Attribution is
  recorded inside the file (`_licence`).
- **The dose** comes from the read text; **the directions** come from a curated **closed set**
  of dispensing phrases ("TAKE ONE CAPSULE THREE TIMES DAILY WITH FOOD" → 每日三次，每次一粒，隨餐服用).
  A label is a structured record, which is the one case where a dictionary beats a model.
- **The safety property is the design** (`app/js/say_cantonese.js`): it speaks only what it can
  translate and **refuses the rest**. An unknown drug name is **never guessed, transliterated
  or approximated**; an unreadable label says — in Cantonese — "*I cannot see it clearly, I
  will not guess — please confirm with a pharmacist*". A wrong dose spoken confidently in
  Cantonese is as dangerous as one spoken in English.
- **Measured, in the real engine, against the live site:** `cantonese_verify.js --engine=webkit`
  **14/14**; the utterance logic `node app/js/say_cantonese.js` **10/10**; the lexicon contract
  `ml/tests/test_drug_lexicon.py` **7/7**.

**A bug this property caught, and the same class of error we had already fixed.** The lexicon
is built from real labels, so it contains entries **keyed on dosage forms** — `TABLET` is a key.
The first matcher read our own bench document, found the word `TABLET`, and **confidently
offered a specific medicine as "the drug"** — a plausible-looking match that is simply wrong,
exactly like the invented account number (§4). Fixed with a stop-list of dosage forms and
excipients plus a rule that a 1–2 word key containing a form word is not a name; a regression
test pins it.

**A bug about verification, not logic.** The 1.7 MB lexicon was **invisible to git**:
`.gitignore` had an unanchored `data/`, which swallowed `app/data/`, so the asset would have
deployed as a **404 on the live site** with every local test still passing. It was caught by
**checking the live URLs after deploying**, not by any test. `data/` is now anchored (`/data/`).

**Three honest limits**, stated wherever the feature is stated and not buried: (1) the sentences
have **not been reviewed by a Cantonese speaker**; (2) the **drug-name step is unvalidated on
real medicine labels** — on our own 12-document bench it matched **nothing** honestly, so the
lexicon's coverage of what we can actually read is not yet measured; (3) the **web engine this
deck was verified in has no Chinese voice**, so the app **shows the Cantonese text** rather than
faking a voice — on a phone the voice must be present in Settings, and the demo device is checked
before filming.

### Deployment

Exported to **ONNX** and run through `onnxruntime-web` **1.20.1**. **26.4 MB**
of models + tokenizer ship inside the app (27,735,952 B, per `bundle-check`),
plus the runtime's only non-JSEP wasm — `ort-wasm-simd-threaded.wasm`,
11,246,032 B = **10.73 MiB** — so the honest on-device total is **≈37.2 MiB**.
Both numbers are facts a judge can reproduce with `du -sb app` and
`bash run.sh bundle-check`; quote the sum, not the weights alone.

The shipped graph is **dynamic width**: input `[1,1,32,width]`, and
`app/js/recognizer.js` reads the graph's own width axis at init, so every crop
is served at its NATURAL width with **no padding at all** (`app/models/ort.json`).
Two notes that used to be here were wrong, and correcting them was worth more
than any model change:

- This file claimed the graph had to be a FIXED 256px because `aten::lstm`
  cannot export with a dynamic sequence length. That was a **mode artefact**:
  `set_export_mode(True)` traces an explicit per-timestep loop that bakes T,
  while the FUSED `nn.LSTM` lowers to a real ONNX LSTM op with a dynamic axis.
  The fixed 256px graph truncated the tail off **98.8%** of lines (only 40 of
  3,270 held-out crops are ≤256px).
- It claimed right-padding was safe "because a trained CTC model emits BLANK
  there". It is **not inert**: the bidirectional LSTM resumes its backward pass
  from the right edge, i.e. inside the padding, and that cost **11.5 points of
  crop-level exact match** — padded 19.0% / CER 0.308 against natural 30.5% /
  0.198. See `docs/18_CRNN_ACCURACY_STRATEGIES.md` §0.1.

**No OCR library.** Tesseract was removed: 41.1 MB deleted, `bundle-check` now
**fails the build** if any part of it returns. It used to ship *while our
manifest claimed otherwise* — see §6.

---

## 3. Test and evaluate

### The two required metrics — classifier, 5 seeds, held-out n=24

| metric | mean | range | sd |
|---|---|---|---|
| **Accuracy** | **95.8%** | 91.7 – 100.0% | 2.63 |
| **Macro-F1** | **95.7%** | 91.4 – 100.0% | 2.72 |

Per class (seed 123): banking/medical/legal/general F1 all 1.0, as is
five-fold CV mean accuracy — the same saturated task, the same seed. They
corroborate; they are not the headline and we never quote them as one.

> **A correction we want judges to see.** Our stored evaluation said
> **100% accuracy, macro-F1 1.0**. Re-running across five seeds gives 1.000,
> 0.958, 0.917, 0.958, 0.958 — mean **95.8%**, sd 2.63. The stored 100% was
> **one lucky seed**. Quoting it would have been easy and would have been the
> single most fragile number in the submission.

**What this metric is not.** n=24 is small, and the four categories are
lexically distinct (sort code / dose / whereas / invoice), so the task is close
to saturated. 95.8% is evidence that **the text-classification stage works
end-to-end on held-out templates**. It is **not** evidence about reading a
medicine label, and it is **not** end-to-end accuracy.

### The OCR, on held-out data — the number that actually matters

17,945 held-out words, both checkpoints scored on the identical test set:

| checkpoint | corpus | word accuracy | CER |
|---|---|---|---|
| epoch 20 | 30,021 crops / 260 receipts | 26.25% | 15.90% |
| **epoch 39** (pre-retrain baseline — the shipped graph is epoch 29, see docs/03) | **52,554 crops / 420 receipts** | **29.30%** | **11.95%** |

Read against the crop level, where the same recogniser looks much better: on the
trainer's own 7,875 held-out crops it records **28.0%** word accuracy
(`artifacts/ocr_training.json`), and paired on **300 held-out crops with only
the weights changing** the retrained checkpoint reaches **57.0% exact line
match at CER 0.156** (`app/tools/arm_ab.js`). The gap between the crop-level
numbers (28%, 57%) and the page-level 29.30% is what the end-to-end section
below decomposes: detection, serving geometry, and the metric each column is
named after. Nothing here is a generalisation claim about medicine labels —
see §1.

### The shipped application, end to end — the honest headline

The two rows above are the recogniser alone. What the product actually does —
detect lines, read them, classify the document, extract fields — measured by
driving the real app in WebKit (iOS Safari's engine) over 12 rendered documents,
3 document types × 4 capture profiles, one record per document:

| metric | value | source |
|---|---|---|
| word accuracy | **57.2%** | `artifacts/pipeline_raw_shipped.json` |
| CER | **0.217** | same |
| ground-truth fields present in the read text | **14 of 24** | same |
| field recovery, the app's extracted-field metric (`ml/src/bench_hackathon.py`) | **33.3%** — 8 of 24 | same |
| classification | **91.7%** | same |

  **Which fixture generation, because it matters.** These are measured on 12 synthetic
  rendered documents, WebKit, one record per document, on the **current** bench fixtures.
  The figures we previously published (45.9% / 0.303 / 9 of 24 / 20.8% / 83.3%) were measured
  on a **previous fixture generation** whose capture profiles were later re-tuned — the old
  ones were 17-67x harsher than real photographs. A controlled A/B (`pipeline_raw_oldmodel_currentfix.json`)
  splits the move: **+1.7 points is the fixture correction and +9.5 is the retrained model**.
  Never read 45.9% -> 57.2% as 11.3 points of model improvement. Full attribution:
  `docs/14_HACKATHON_BENCHMARK.md`.

The two field rows are deliberate: "present in the read text" is the weaker,
more direct claim (the recogniser produced the right string somewhere on the
page) and it is the one the Grand Final deck quotes; `field_recovery` is the
stricter one the app's own scorer reports, and it asks whether the value also
survived extraction. Both come from the same 12 records; quote which one you
mean.

All 12 scans completed with no crash and no failed scan (`"error": null` on
every record). **Read the metric, not just the number:** `word_accuracy` here is
a **normalised token recall** — the fraction of ground-truth words present in
the read text after `norm()` strips punctuation and case — **not** exact-word
match. Both definitions live in `ml/src/bench_hackathon.py` and are unit-tested
there; the browser side deliberately computes no metrics, so the two cannot
drift apart.

**The trajectory is the result.** Same 12 documents, same metric, each step a
defect we found in our own train/serve contract — with the step that regressed
kept in:

| word acc | CER | what changed |
|---|---|---|
| 3.6% | 0.708 | as shipped on 10-20: ONNX graph baked at a FIXED 256px against 320px training crops, the ranking could only choose among degraded arms, and the deskew correction was applied backwards |
| 22.4% | 0.575 | export width brought to the training clamp (320), same checkpoint — the app had been feeding the model the first 256px of **98.8%** of lines; the read changed on **200 of 200** held-out crops |
| 17.1% | 0.600 | **a step back on purpose:** dynamic-width graph *and* retrained weights went in together, and on these 12 documents the pair measured worse — even though at crop level the same retrain moved exact line match **3.3% → 57.0%** (CER 0.156). That disagreement is why the next fix was measured on the arm list, not the model |
| 35.0% | 0.465 | the untransformed image made reachable: `preprocess()` already offered the raw deskewed grayscale, and a later branch replaced the candidate list with the restorer's three degraded arms, so the arm that measures best could never win |
| **45.9%** | **0.303** | *(previous fixture generation)* deskew applied in the correct sense: the estimator measured the right angle and the correction **doubled** the tilt instead of removing it — `off_axis` 1.1% → 45.1%, CER 0.882 → 0.234 |

Raw records for every row, in that order: `artifacts/pipeline_raw.json`,
`pipeline_raw_final.json`, `pipeline_raw_dynamic.json`,
`pipeline_raw_rawarm.json`, `pipeline_raw_deskew.json` — **every row above was measured on
THE PREVIOUS FIXTURE GENERATION**; the current shipped build is `pipeline_raw_shipped.json`.
Per capture profile on
the deskew arm: `studio_clean` 52.3% (and 100% classified correctly),
`handheld_light` 46.5%, `jpeg_social` 39.8%, `off_axis` 45.1%. Reproduction,
per-profile detail and the caveats — including which numbers are *not*
generalisation claims — are in `docs/14_HACKATHON_BENCHMARK.md`:

```
bash run.sh bench-pipeline && bash run.sh bench-hackathon
```

The caveat that remains, stated plainly: this is **not** 95%, and on real
hand-photographed pages the ceiling is the **detector**, not the recogniser —
see the next section.

### One situation where it gets it wrong — and why

**A medicine label photographed in poor light at night.**

On our 20 hand-photographed real pages, the line detector found **67.9%** of
text lines and line-exact accuracy was **1.8%**. Real failures:

```
'GREEN FIELD'            -> 'NVIH'
'Order #: 69923 dine In' -> '15305 E PACIFIC COAS'
'He Pho Ga'              -> 'NO PIO 66'
```

**Why.** The detector builds a row-ink profile and splits where ink drops to a
gap. On a dark, glossy, curved label, ink appears in **100% of rows** on one
page (`1019-receipt`, profile minimum 3 of 93) — there is no gap anywhere, so
**26 transcript lines collapse into 1 band**. The recogniser is then handed the
wrong pixels.

We tried six fixes and measured all of them. One raised bands-per-line
0.736 → 0.817 and detection recall 67.9% → 71.1% while pushing **CER 56.5% →
61.5%** — so we **reverted it**. It raised the ceiling and made the text worse.

**Then we checked whether the difference was even real.** On 290 aligned lines
the best variant gained **2 lines and lost 0**: exact McNemar **p = 1.000**. Six
variants, all within ~4 pp, none significant. That is a **sample-size problem
wearing a tuning problem's clothes.**

Three of the twenty pages are **resolution-bound** (12–17 px per line against a
32 px recogniser input — 1.8–2.7× upscale) and **cannot be fixed by any
detector**; the information was never captured.

**For a dose this is disqualifying, and we say so.** Our design refuses to speak
a field it is not confident about — it omits rather than guesses. Measured end
to end, our own rendered documents now read at **57.2%** word accuracy (12 renders, WebKit,
current fixtures), but on
real hand-held photographs we are still at **1.8% line-exact**, where the honest
product is *"I could not read that label — please check it with a pharmacist"*,
and that is a **failure state, not a product**. That gap — clean renders against
real captures — is the size of the remaining work, and it is a data problem
before it is a modelling one.

---

## 4. Check for fairness

We found two groups our tool would treat unfairly. Both are verifiable from our
own configuration.

### Group 1 — Chinese-reading elderly patients (the majority in Hong Kong)

**The finding.** Our recogniser's output alphabet is:

```
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '     (37 symbols)
```

`has CJK: False`. **There is no Chinese character in the model's output
space at all.** A 繁體中文 medicine label cannot produce a single correct
character — not "poorly", *impossibly*. This is the **input** barrier, and it is unchanged by
the Cantonese work: the app now *speaks* Cantonese (§2), but reading a Chinese-**only** label
still needs the alphabet expanded. Our text-to-speech also defaults to `en-GB`, though a first
Cantonese output build now ships and is selectable.

**Why this is the most serious thing in this document:** Hong Kong's dispensed
medicine information is routinely bilingual, and a large share of over-65
patients read Chinese as their first language. **A Hong Kong medicine tool that
cannot read Chinese excludes most of the people it is built for.** An English-only
classifier scoring 95.8% tells us nothing about the patient in front of us.

**What we would do.** (1) Expand the output alphabet to traditional Chinese
before any field pilot — this is a data problem, ~3–4k common characters cover
most medicine labels. (2) Cantonese output **now ships as a first build** (§2); make it the default for the HK
build once the three limits there are closed. (3) Evaluate per-language and **fail the release gate** if any language
scores below a floor. (4) Until then, **do not deploy to patients** — this is
not a nice-to-have, it is the difference between serving the population and
serving a subset of it.

### Group 2 — Elderly patients with low vision or tremor

**The finding.** The target population is defined by needing help reading small
labels. Those are exactly the conditions under which a handheld photo fails:
the detector collapses on ink-saturated pages (§3), and one blur
(`handheld_light`) defeated **every** Tesseract configuration we ever tested.
Low vision and tremor are not edge cases here; they are **the use case**.

**What we would do.** (1) A **guided capture mode** that scores blur/ink
coverage *before* capture and re-shoots automatically, so the user never has to
diagnose their own photograph. (2) Voice-first interaction — the app speaks its
instructions, so it does not depend on reading small screen text. (3) Test
recruitment that explicitly includes low-vision and tremor participants, and
**publish per-group accuracy**, not just the mean. A mean over a young,
steady-handed test panel is not evidence for this population.

### Should AI be making this decision at all?

**No — and this is the most important sentence in the submission.**

Our model **must not** be the last check on a medication dose. It should do
what it is actually good at: *reduce reading difficulty*. The safe division of
labour:

- **AI does:** read the label, structure it, speak it, and **refuse** when
  uncertain.
- **A human does:** approve the dose. Pharmacist, doctor, or the patient with a
  second pair of eyes.

**Who is accountable when it goes wrong?** **The licensed clinician, never the
software.** A phone app is not a regulated medical device in Hong Kong, and a
manufacturer cannot accept clinical liability for a dose. Concretely: the app
**never displays a dose as authoritative**; it labels everything it says as
"read from your label, please verify"; it records what it was unsure about; and
an adverse event routes to a named clinician. If we cannot design the system so
that its worst failure is a refused answer rather than a wrong dose, we should
not ship it.

---

## 5. Deploy it

### Who would use it

| candidate | fit |
|---|---|
| **Hong Kong Hospital Authority — Hospital Authority Pharmacist Services / community pharmacy programme** | **Best fit.** Owns the dispensing relationship, already runs patient-facing medication safety programmes, and a 22%-over-65 population with 46.4% polypharmacy is exactly its problem. |
| **Centre for Health Protection** | Plausible for public health messaging and surveillance of misread labels. |
| **Social Welfare Department / community elderly centres** | Plausible distribution channel for the offline-first design. |
| **Macau Health Bureau** | Same demographic trajectory (elderly exceeded youth in 2023), smaller deployment. |

**Honest caveat: we have no partnership, no letter of intent, and no pilot.**
This is a table of who *should* own it.

### Cost to build and run for one year

Stated as assumptions, because a cost model without them is fiction.

| item | one-year cost | basis |
|---|---|---|
| **Model training compute** | **US$0** | We train on a 4-core CPU laptop, ~2.8 h per full retrain. No GPU, no cloud training. This is a real architectural benefit. |
| **Hosting** | **≈US$0–100/yr** | The app is a static PWA. No server, no database, no per-user backend. Any CDN or GitHub Pages serves it. |
| **Backend / inference** | **US$0** | All inference is on-device. There is no API to pay for at scale — the marginal cost of a million users is zero. **This is the strongest commercial argument.** |
| **Domain + PWA hosting** | US$10–40/yr | one domain, CDN free tier |
| **Maintenance & retraining** | **1 engineer, ~0.2 FTE** | 1 retrain/quarter is ~12 h of CPU time; the rest is incident response and revalidation. |
| **Data collection — medicine labels** | **the real cost** | Consent, photography, annotation and governance for a medicine-label corpus. Genuinely the dominant cost, and the thing we have not budgeted. |

**Headline: under US$1,000 for year one** — *excluding* the medicine-label
data collection, which we estimate at **several months** and which is the real
barrier. A cost model that reports only the cheap line is misleading, so we are
naming the expensive one.

### How we would explain this to the public

Plainly, and without the word AI:

> "SightLine helps you read the label on your medicine box out loud, in your
> own language. It works **on your phone**, so **your photo never leaves the
> device** — no internet, no account, nothing sent to us. It is a **reading
> aid, not a doctor**: it will tell you when it cannot read something clearly,
> and it will always ask you to check the dose with your pharmacist. If you are
> ever unsure, trust the pharmacist, not the app."

**Safety commitments we would publish:**

1. **It never guesses a dose.** Uncertain → it says so and stops.
2. **Your photo stays on your phone.** Verifiable: the app is a PWA with no
   network call after install; our models run locally.
3. **It always says "verify with a pharmacist".** Not buried in terms.
4. **It shows its uncertainty**, not a confident wrong number.
5. **Free, no account, no data collection.** No dark patterns, no ads.
6. **In Cantonese by default** for the HK build — or we do not ship (§4).

---

## 6. What we got wrong

Judges should know what broke. Each of these now has a check behind it, and most
of them fail the build.

| what happened | why it matters | now |
|---|---|---|
| **The served graph was exported at a FIXED 256px** while training built every crop at its natural width clamped to `MAX_W = 320` — and the corpus median IS that clamp | the app fed the model the first 256px of **98.8%** of lines; crop-level exact match 44.7% at natural width against 0.0% cut to 256, and the read changed on **200 of 200** held-out crops | graph re-exported at the training clamp, then made **dynamic width**; `ml/tests/test_graph_width_contract.py` asserts **equality** with the clamp (the old assertion was `width <= MAX_W`, which every narrower graph satisfies — it passed hardest on the worst value) |
| **Right-padding was assumed inert** because CTC emits BLANK in the tail — it is not: the bidirectional LSTM resumes its backward pass from the right edge, i.e. inside the pad | **11.5 points of crop-level exact match** (padded 19.0% / CER 0.308 vs natural 30.5% / 0.198) | dynamic-width graph — the app serves every crop at its **natural** width, no padding at all |
| **The image transform ran at serve and never at train** — and a later branch replaced `preprocess()`'s candidate list with the restorer's three degraded arms | the arm that measures **best** (raw deskewed grayscale) could never win the ranking; 34.5 points of exact line match, end-to-end word accuracy 17.1% → 35.0% | the untransformed candidate is prepended; the ranking rule was left alone, because the **list** was the bug, not the rule |
| **The deskew correction was applied in the opposite sense** — the estimator measured the right angle, and `preprocess()` rotated the page *away* from straight, doubling the tilt | it fired only on `off_axis`, the pages whose raw detection was already marginal: `off_axis` word accuracy **1.1%**, and two pages lost their lines entirely | one sign changed (`rotate(gray, -angle)`); `off_axis` now **45.1%** (CER 0.882 → 0.234) and the app-wide number **35.0% → 45.9%** (both on the previous fixture generation; re-benched
on the current fixtures the deskew step measures **47.7%**); `app/tools/deskew_verify.js` asserts the **round trip** (apply, re-estimate, require ~0) and exits non-zero |
| Our manifest shipped `"tesseract": false` **while 30 MB of Tesseract loaded at runtime** | the shipped artifact lied; a judge opening the bundle would have seen it | removed (41.1 MB); `bundle-check` **fails** if any of 4 routes back |
| A safety gate had its **condition inverted**, so it never ran — it skipped precisely when the model was good enough for the check to matter | the only check on the assumption the ONNX export rests on | fixed, passes, and **mutation-tested to confirm it can fail** |
| `eval_sroie_test()` called `most_common()` on a `defaultdict` | it had **never run to completion**; our headline held-out number was previously unmeasured | fixed |
| Our first regression gate **grepped for a constant name**; renaming it **passed the test** | a gate that can be defeated is worse than none | asserts resolved values; the rename mutation is caught |
| The stored classifier eval said **100%** | it was one lucky seed; true 5-seed mean is 95.8% | re-measured and published |
| Six detector variants "improved" things **inside the noise** (McNemar p = 1.000) | we nearly shipped a regression as a gain | significance now computed before any claim |
| The Cantonese matcher **named a medicine from the word `TABLET`** — the lexicon is keyed on real labels, so it contains dosage forms, and our own bench document contains that word | a plausible-looking match that is simply wrong — **the same class of error as the invented account number** (§4), in the medium where wrong matters | a stop-list of dosage forms and excipients (`NOT_A_NAME`) plus a rule that a 1–2 word key containing a form word is not a name; `node app/js/say_cantonese.js` pins it (10/10) |
| The 1.7 MB Cantonese lexicon was **invisible to git** — `.gitignore` had an unanchored `data/` that also swallowed `app/data/` — so it would have shipped as a **404 on the live site** with every local test passing | a shipped-but-missing asset is invisible to unit tests; only the deployed artifact tells the truth | `data/` anchored to `/data/`; caught by **checking the live URLs after deploying**, not by a test |

---

## 7. If you remember three things

1. **We trained the OCR.** 942,166 parameters, from scratch, exported to a
   **3.6 MB** self-contained, dynamic-width ONNX graph, decoding in the browser.
   No OCR dependency exists — and the build fails if one returns.
2. **We know exactly which part is broken.** Line detection caps accuracy at
   **67.9% recall**; line accuracy on real photos is **1.8%**; our best fix
   improved the ceiling and worsened the text, and we **reverted it**. We also
   proved the difference was not statistically significant.
3. **We found two ways our own tool would exclude the people it is for** — it
   cannot read Chinese, and it fails for low-vision users. A tool that cannot
   read a Hong Kong medicine label should not be presented as a Hong Kong
   medicine tool. It now *speaks* Cantonese — a first build that refuses a name
   it cannot translate — but it has not been reviewed by a Cantonese speaker, the
   drug-name step is unvalidated on real labels, and the proving engine has no
   Chinese voice. We would rather be judged on that than on a number.

### Reproduce it

```bash
bash fetch_data.sh          # SROIE + CC-BY-4.0 hand-photographed set
bash run.sh build-data      # 52,554 crops, split by document identity
bash run.sh train-ocr 40    # ~2.8 h, 4 CPU cores, no GPU
bash run.sh export          # ONNX + parity + the graph-width contract check
bash run.sh bundle-check    # real inference through every shipped model
bash run.sh test            # the full pytest suite (ml/tests)
```

Datasets are not redistributed; `fetch_data.sh` pulls them and the licences are
cited. Full detail: `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`,
`docs/03_RESULTS.md`, `docs/07_EVALUATION_METHODOLOGY.md`.