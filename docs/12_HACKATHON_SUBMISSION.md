# SightLine — Imperial College London Hackathon, Hong Kong & Macau

**Submission.** Structured to the five required steps. Every statistic is
sourced and every metric is measured on held-out data.

**Read this first if you only read one thing:** §6. We found two ways our own
tool would exclude the people it is meant to serve. We did not find them by
being asked; we found them by reading our own model config.

---

## 1. Understand the problem

### The problem we chose

A patient is prescribed several medicines. The dose is printed on a small label,
in a font they did not choose, on a box they open in poor light. Many of them
are over 65, take **five or more medicines at once**, and read Chinese.

**The task we set the AI:** given a photograph of a medicine label, read the
drug name, strength and dose aloud, and speak it in the patient's own language —
on the phone, with no network.

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
| Language and speech | so it can speak the answer | System TTS; Cantonese (zh-HK) is the gap in §6. |

**We will not pretend we have medicine data.** Everything in §3 was measured on
receipts and synthetic documents. Transfer to medicine labels is **an assumption
we have not tested**, and saying so is more useful to a judge than a number we
cannot defend.

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
  linear → 38 classes. **942,166 parameters.** Input `[1,1,32,256]`, output
  `[1,32,38]`.
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

### Deployment

Exported to **ONNX** and run through `onnxruntime-web`. **26.68 MB** total.

Why the fixed 256px width: `aten::lstm` cannot export with a dynamic sequence
length, and an explicit gate loop cannot unroll over a symbolic bound. Both
failed for real. So crops are scaled proportionally, right-padded to 256, and
the decoder trims to the true timestep count.

Padding is safe **because a trained CTC model emits BLANK there** — verified, not
assumed:

```
pad  32->256: ok (ref 'RM'       vs padded 'RM')
pad  64->256: ok (ref '1071'     vs padded '1071')
pad 128->256: ok (ref '01143008' vs padded '01143008')
pad 200->256: ok (ref 'M  C2  0' vs padded 'M  C2  0')
```

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

Per class (seed 123): banking F1 100%, medical 100%, legal 100%, general 100%.
Five-fold CV mean accuracy 1.0.

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
| **epoch 39** | **52,554 crops / 420 receipts** | **29.30%** | **11.95%** |

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
a field it is not confident about — it omits rather than guesses. At 1.8% line
accuracy on real photos, the honest product today is *"I could not read that
label — please check it with a pharmacist"*, and that is a **failure state, not
a product**.

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
character — not "poorly", *impossibly*. Our text-to-speech also defaults to
`en-GB`.

**Why this is the most serious thing in this document:** Hong Kong's dispensed
medicine information is routinely bilingual, and a large share of over-65
patients read Chinese as their first language. **A Hong Kong medicine tool that
cannot read Chinese excludes most of the people it is built for.** An English-only
classifier scoring 95.8% tells us nothing about the patient in front of us.

**What we would do.** (1) Expand the output alphabet to traditional Chinese
before any field pilot — this is a data problem, ~3–4k common characters cover
most medicine labels. (2) Add Cantonese (zh-HK) TTS as the default for the HK
build. (3) Evaluate per-language and **fail the release gate** if any language
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

Judges should know what broke. Each of these is now a check that fails the build.

| what happened | why it matters | now |
|---|---|---|
| Our manifest shipped `"tesseract": false` **while 30 MB of Tesseract loaded at runtime** | the shipped artifact lied; a judge opening the bundle would have seen it | removed (41.1 MB); `bundle-check` **fails** if any of 4 routes back |
| A safety gate had its **condition inverted**, so it never ran — it skipped precisely when the model was good enough for the check to matter | the only check on the assumption the ONNX export rests on | fixed, passes, and **mutation-tested to confirm it can fail** |
| `eval_sroie_test()` called `most_common()` on a `defaultdict` | it had **never run to completion**; our headline held-out number was previously unmeasured | fixed |
| Our first regression gate **grepped for a constant name**; renaming it **passed the test** | a gate that can be defeated is worse than none | asserts resolved values; the rename mutation is caught |
| The stored classifier eval said **100%** | it was one lucky seed; true 5-seed mean is 95.8% | re-measured and published |
| Six detector variants "improved" things **inside the noise** (McNemar p = 1.000) | we nearly shipped a regression as a gain | significance now computed before any claim |

---

## 7. If you remember three things

1. **We trained the OCR.** 942,166 parameters, from scratch, exported to a 4 MB
   self-contained ONNX graph, decoding in the browser. No OCR dependency exists
   — and the build fails if one returns.
2. **We know exactly which part is broken.** Line detection caps accuracy at
   **67.9% recall**; line accuracy on real photos is **1.8%**; our best fix
   improved the ceiling and worsened the text, and we **reverted it**. We also
   proved the difference was not statistically significant.
3. **We found two ways our own tool would exclude the people it is for** — it
   cannot read Chinese, and it fails for low-vision users. A tool that cannot
   read a Hong Kong medicine label should not be presented as a Hong Kong
   medicine tool. We would rather be judged on that than on a number.

### Reproduce it

```bash
bash fetch_data.sh          # SROIE + CC-BY-4.0 hand-photographed set
bash run.sh build-data      # 52,554 crops, split by document identity
bash run.sh train-ocr 40    # ~2.8 h, 4 CPU cores, no GPU
bash run.sh export          # ONNX + parity + padding-invariant checks
bash run.sh bundle-check    # real inference through every shipped model
bash run.sh test            # 278 tests
```

Datasets are not redistributed; `fetch_data.sh` pulls them and the licences are
cited. Full detail: `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`,
`docs/03_RESULTS.md`, `docs/07_EVALUATION_METHODOLOGY.md`.