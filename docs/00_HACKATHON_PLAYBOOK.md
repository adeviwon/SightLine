# How to win the Imperial Hackathon with SightLine

This is the strategy document. The build is **done** — 278 tests green, the
PWA live at <https://adeviwon.github.io/SightLine/> — so what remains is
**Round 1** (150-word proposal + 3–5 minute video, due **2026-10-25**) and the
**Grand Final** (2026-11-14). The operational countdown lives in
[`13_ROUND1_SUBMISSION_KIT.md`](13_ROUND1_SUBMISSION_KIT.md); this file keeps
the pitch, the judge playbook, and the deck strategy.

---

## Part 1 — The 60-second pitch

Rehearse this until it is boring. Judges hear forty pitches; yours must land
in the first fifteen seconds or you have lost them.

> **SightLine reads the medicine label out loud — and the goal is to say it
> back in Cantonese — with the internet switched off.**
>
> In Hong Kong alone — where this hackathon is judged — **199,600 people, 2.7%
> of the population, report long-term seeing difficulty** (HKSAR C&SD Special
> Topics Report No. 63, 2021), three in four of them over 65. For an older
> Hongkonger the label is unreadable twice over: the print is too small to see,
> and it is printed in English they never learned. Every app that helps them
> uploads that label to a cloud server, because the OCR runs on a GPU somewhere
> else. For medication, that's a privacy line they shouldn't have to cross to
> be independent.
>
> **Honest state of the build:** the **Cantonese output build ships** — `app/js/say_cantonese.js`
> plus an 18,839-pair CC-BY-4.0 zh/en medicine-name lexicon, speaking the drug name and dose in
> Cantonese and **refusing a name it cannot translate** rather than guessing. The gap was always on
> the output side: no Chinese OCR is needed, because HK labels print the drug, strength and
> directions in English. **Every accuracy number in this repo is still measured on the ENGLISH
> read** — that has not changed — and the feature carries three limits that are stated, not buried:
> no Cantonese speaker has reviewed the sentence frames, the drug-name step is now **measured on
> 150 real medicine labels** — 10 recovered, **8 verbatim**, 8 lexicon hits, a floor on a Taiwanese
> corpus, not Hong Kong labels — and an engine with no zh-HK voice shows the text rather than faking
> one. The full statement is [`26_IDEA_REFINED.md`](26_IDEA_REFINED.md) §7-8.
>
> We trained **two neural networks to read the page**. The first is a
> restoration CNN that undoes the motion blur and sensor noise you get holding
> a phone one-handed over a label. The second is the one that matters: a
> **CRNN text recogniser we trained ourselves, on 52,554 real word crops from
> 420 receipts** — with perspective warps driven by 199 real hand-held
> homographies. We wrote the character-level model from scratch. We did not
> use Tesseract.
>
> We benchmarked Tesseract against itself first — 336 configurations, every
> page-segmentation mode — and it recovered *no* safety-critical field on those
> real handheld photos. That's why we replaced it.
>
> A fine-tuned MiniLM then reads the recovered text and works out what kind of
> document it is. **MiniLM can't see pixels** — that's a text encoder — so it
> sits downstream of our recogniser rather than replacing it. Being precise
> about that is the point: it means both vision models are ours.
>
> Everything runs on-device. Turn on airplane mode and it still works.

**The four numbers to say out loud** (re-run the gates to confirm before you
present):

1. **Classifier accuracy, mean over 5 seeds, held-out n=24 — 95.8% (sd 2.63);
   macro-F1 95.7% (sd 2.72)**, per `ml/experiments/step3_f1.py`. Stable and
   honest; never quote the 100% seed — the submission discloses that
   correction on purpose.
2. **Held-out scan OCR, epoch 39 (the PRE-retrain baseline's row — the shipped graph is epoch 29; see docs/03's provenance note), 17,945 test words — word accuracy 29.30%,
   CER 11.95%** (`docs/03_RESULTS.md` §3). This answers "did you actually train
   something that reads?"
3. **Real hand-held photos (20 pages) — recall 67.9%, line-exact 1.8%,
   CER 56.5%** (`docs/03_RESULTS.md` §3) — and the app **refuses** a field it is
   not confident about. Say the gap before a judge does. *(Corrected 2026-10-21:
   this used to quote the localisation variant's 71.6% / 2.4% / 58.4%, whose
   whole gain is two lines and statistically insignificant — McNemar p = 1.000.
   Those are the 20-page set; `docs/14` also records the 193-page set-B reproduction (71.6% / 2.4% / 58.4%).)*
4. **Payload — 26.45 MiB of models + tokenizer, ≈37.2 MiB installed on device.**
   *(Corrected 2026-10-21: the old "26.68 MB" counted the weights only and
   excluded the 10.73 MiB ONNX-runtime WASM the app cannot infer without.)* A
   fact about the download, not a design constraint.

> ### No payload budget
>
> There is **no size cap** on this project, and there is no gate enforcing one:
> `ml/src/bundle_check.py` asserts that every shipped model LOADS and runs, never
> that the payload is small. Sizes are reported because a judge may ask; they are
> not a reason to reject an approach.
>
> Decisions that were previously justified by size are therefore open again, and
> must now be settled on accuracy: the removal of Tesseract (~41 MB),
> the refusal of OpenCV.js (~9 MB) for preprocessing, and INT8 quantisation of the
> MiniLM encoder (86.8 MB → 22.16 MB; `docs/01_ARCHITECTURE.md`). Re-open them
> with a measurement, not a
> feeling — `bash run.sh bench-hackathon` for the end-to-end number and
> `bash run.sh eval-render` for the recogniser's.

**Do not quote a single held-out score without its context.** Field extraction
is all-or-nothing by design, which is a much harder bar than per-character
accuracy, and the two are reported separately. See
[`09_INTERPRETING_THE_NUMBERS.md`](09_INTERPRETING_THE_NUMBERS.md) — knowing
this before a judge asks is worth more than a rounder headline.

---

## The three questions a judge will actually ask

**"You said you don't use Tesseract. What do you use instead?"**

> A CRNN with a CTC head — a convolutional stack feeding a bidirectional LSTM,
> trained by us on 52,554 real word crops from 420 scanned receipts.
> 942,166 parameters, **3.61 MiB** as a single dynamic-width ONNX file
> (`app/models/crnn.onnx`, 3,785,916 B), running in the browser's WASM. CTC means greedy decoding is one softmax
> per timestep — no beam search, no lexicon — which is what makes it viable on
> a phone.

**"Couldn't you just fine-tune an off-the-shelf OCR model?"**

> We could have, and that would have been faster. We couldn't get at
> word-level supervision aligned to *our* degradation profile, and we needed
> the model small enough to ship in a PWA. Training from scratch on real data
> gave us both, plus the ability to say honestly that it is ours.

**"What's your accuracy?"** — this is where most teams overclaim.

> The classifier: **95.8% mean over five seeds** (macro-F1 95.7%). End-to-end
> field extraction, where
> a wrong digit means we omit the field rather than read it aloud: much lower —
> on real hand-held photos, 1.8% line-exact — and here is exactly where it
> breaks. We measure that on real photos, and the
> three failure modes are named.

Volunteering the gap before you're asked is worth more than the number was.

---

## Part 2 — Why this wins, against the other 40 teams

Most hackathon projects fall into one of four buckets. Yours must be
demonstrably outside all four.

| Common project | Why it loses | How SightLine is different |
|---|---|---|
| "We called the OpenAI API" | Costs money, needs network, privacy-dead | Two trained models, ONNX, zero network |
| "We used a pretrained model" | Downloading weights ≠ training one | We show training curves, a data pipeline, and ONNX parity proof |
| "It's a mobile app" | Expected, undifferentiated | The hard part is the ML, and we can defend it |
| "It has a nice UI" | Not the judging criterion | UI is a supporting actor; accuracy is the lead |

**Your three defensible claims.** Be ready to defend each for 5 minutes:

1. **The models are actually ours.** `ml/src/train_restorer.py` and
   `ml/src/train_classifier.py` are readable in one sitting. Judges can run
   `bash run.sh test` (278 tests) and `bash run.sh bundle-check` (real
   inference through every shipped model) from a fresh clone.
2. **The evaluation is honest.** Splits by document identity, never by crop.
   Five-seed classifier metrics with the 100%-seed correction published, not
   buried. McNemar significance computed before any detector claim. See
   [`07_EVALUATION_METHODOLOGY.md`](07_EVALUATION_METHODOLOGY.md).
3. **The offline claim is verifiable.** Not a bullet point — a demo where you
   turn on airplane mode mid-presentation.

---

## Part 3 — The final 8 days (2026-10-17 → 2026-10-25)

Round 1 needs exactly two artifacts: the **150-word proposal** and the
**3–5 minute video**. The dated, operational countdown — one item per day,
each with a single done-criterion — lives in
[`13_ROUND1_SUBMISSION_KIT.md`](13_ROUND1_SUBMISSION_KIT.md). The strategy,
compressed:

- **Days 1–2 (Oct 17–18): freeze the numbers.** Re-run both gates, then touch
  nothing. Done = `bash run.sh test` and `bash run.sh bundle-check` pass and
  every figure in docs/12–13 matches them.
- **Days 3–4 (Oct 19–20): script and device.** VO ≤3:30 from the shot list;
  filming phone has the PWA installed and airplane-mode verified.
  Done = a timed read-through and one spoken scan in airplane mode.
- **Days 5–6 (Oct 21–22): film and cut.** Five takes plus a screen-recording
  backup. Done = every shot-list beat covered, and a captioned cut watched
  start-to-finish twice.
- **Day 7 (Oct 23–24): verify and upload.** Every on-screen number traced to
  `docs/03_RESULTS.md`; upload early. Done = both deliverables live in a
  private window 24 h before the deadline.
- **Day 8 (Oct 25): submit with margin.** Done = confirmation received.

**The single most common way to lose:** a demo that only works on the
presenter's phone with the app already open. Rehearse the full cold start on
the filming phone, and keep the screen recording as the backup track.

---

## Part 4 — Grand Final slide deck (9 slides, ≤10 allowed; due 2026-11-12)

**1. Problem** — 2.1 billion people with vision impairment *(global estimate —
unverified project note, no artifact in this repo; the HK figure in Part 1 is
ours)*; for the user we chose, the label is unreadable twice over — too small to
see, and printed in English they never learned; independence vs.
privacy is a forced trade-off in every existing app. One sentence, no more.

**2. Solution** — the pipeline diagram. Camera → restore → read → understand → speak.
Show the actual architecture image from `docs/01_ARCHITECTURE.md`.

**3. Why it's hard** — the failure mode. A shaky hand at 30 cm produces motion
blur that destroys the strokes of a digit. Show the `handheld_heavy` sample
from the contact sheet. This slide earns the ML slide.

**4. The models** — SightLineNet (37,793 params, trained) + MiniLM (frozen
22.7M encoder, 49,796-param head). Training curves. Params on the slide.

If asked why the restorer is so small: 126K params trained at 124 s/epoch on
4 CPU threads is 80+ minutes for one model. 37,793 params trains in ~35
minutes and scores the same — we measured, then chose. That is the answer.

**5. The data** — 52,554 crops from 420 ICDAR SROIE receipts plus CC-BY-4.0
hand-photographed pages (Zenodo 13688441), split by document identity, never
by crop. This is the slide that separates you from
teams that ran `train_test_split` once.

**6. The results** — the table from `docs/03_RESULTS.md`. Include the confidence
intervals. Include the sub-human rows.

**7. It runs on a phone** — the payload table, the ONNX parity proof, the
airplane-mode screenshot.

**8. Privacy** — verifiable from source. Zero network. Why this matters for
a medicine label specifically.

**9. What's next** — one line. Then stop talking.

---

## Part 5 — Questions you will be asked

### "What's your accuracy?"
Don't assert. Point at the file:
> "Three numbers, kept apart. The classifier: **95.8% accuracy, 95.7%
> macro-F1, mean over five seeds** on held-out text (`ml/experiments/step3_f1.py`)
> — the stored eval once said 100%, which was one lucky seed, and we published
> the correction. Held-out scan OCR: **29.30% word accuracy, 11.95% CER** on
> 17,945 words (`docs/03_RESULTS.md` §3). On real hand-held photos the ceiling is
> the line detector: **67.9% recall, 1.8% line-exact** (`docs/03_RESULTS.md` §3)
> — so the app refuses a field it is not confident about rather than reading a
> wrong dose aloud. Reproduce with `bash run.sh test`."

### "Your real-photo numbers are terrible. Why ship at all?"
Concede, then reframe:
> "On those 20 pages, yes — and we say so on purpose. Three of the twenty are
> resolution-bound (12–17 px lines against a 32 px input); no detector can fix
> them, the information was never captured. The rest is a sample-size problem:
> six detector variants were statistically indistinguishable (McNemar
> p = 1.000). What ships today is the refusal behaviour: 'I could not read
> that — please check with a pharmacist.' An omitted field is safe; a guessed
> dose is not. The fix is more hand-photographed, line-labelled pages — that
> is the next data task, not a tuning task."

### "Why MiniLM? Why not fine-tune the whole model?"
> "Because a frozen encoder can't catastrophically forget, it trains in
> seconds on CPU, and the trained head exports to a 0.2 MB ONNX file. We can
> show the parity check proving the ONNX graph matches PyTorch to
> max |Δ| = 1.94e-05."

### "What if the model is wrong about a dosage?"
> "That's why every stage carries a confidence score, and why low confidence
> triggers a retake prompt instead of a guess. A blind user who is told
> 'I couldn't read this, please retake' is safe. One who is told '400mg' when
> the label says 500mg is not. Safety is a UX problem, not just an accuracy
> problem."

---

## Part 6 — Submission checklist

```
□  README renders on GitHub (relative links work)
□  `bash run.sh test` (278 passed) and `bash run.sh bundle-check` pass from a fresh clone
□  Every number in README/docs matches docs/03_RESULTS.md and the appendix of docs/11
□  Demo video uploaded, link works in a private window
□  Live demo or video — whichever is required, both ready
□  Repo is public, has a licence
□  Two screenshots in the README (app UI + results table)
□  Team member names and roles listed
□  Submitted before the deadline — with 2 hours of margin
```

---

## Part 7 — If you are behind schedule

Round 1 is two artifacts, so the cut list is short. Cut in this order; never
cut in the reverse order.

1. Policy-anchor breadth in the video — cite the **2025** Policy Address Paras
   **75 / 226 / 266** on screen only (`docs/13_ROUND1_SUBMISSION_KIT.md`; the
   Open Challenge rule names the **2025** edition — we had been citing 2026).
   Keep the 2026 follow-through paragraphs for Q&A, off-screen. −half a day.
2. A second demo device — one filming phone plus the screen-recording backup
   is enough for Round 1. −half a day.
3. Music, colour grade, caption styling. Cut last, never first. −half a day.

**Never cut:**
- The airplane-mode demo. It is the single most memorable moment you have.
- The refusal beat ("could not read that — please check with a pharmacist").
- The honest numbers. A team caught rounding up loses more than a team with
  a 29% word-accuracy number and a precise explanation of where it breaks.
