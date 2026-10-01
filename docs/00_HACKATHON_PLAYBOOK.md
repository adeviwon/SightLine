# How to win the Imperial Hackathon with SightLine

This is the working document. It assumes you have ~36 hours, a team of 3–5,
and one goal: **win the Open Challenge track.**

Everything here is ordered by what actually moves a judging panel.

---

## Part 1 — The 60-second pitch

Rehearse this until it is boring. Judges hear forty pitches; yours must land
in the first fifteen seconds or you have lost them.

> **SightLine reads your bank statements, prescriptions, and legal letters
> out loud — with the internet switched off.**
>
> Thirty-one million blind people can't read their own medication label.
> Every app that helps them uploads the document to a cloud server, because
> the OCR runs on a GPU somewhere else. For a medical prescription, that's a
> privacy line they shouldn't have to cross to be independent.
>
> We trained two models to run on the phone. A PyTorch restoration network
> that undoes the motion blur and sensor noise that happens when you hold a
> phone over a document with one hand — which is exactly the situation our
> users are in every time. And a MiniLM sentence transformer that reads the
> recovered text and works out what kind of document it is.
>
> We measured it on ten simulated capture conditions, including the ones
> where a human can't read the page either, and we report those honestly
> rather than hiding them.
>
> Everything runs on-device. Turn on airplane mode and it still works.

**The three numbers to say out loud** (from `docs/03_RESULTS.md`, re-run to
confirm before you present):

1. **Cross-validation accuracy** for the classifier — the mean over 20
   disjoint folds. Stable, and the number that belongs on a slide.
2. **End-to-end field accuracy** per capture profile, showing where it works
   and where it honestly fails.
3. **Model payload size** — the "it fits on a phone" proof.

**Do not quote a single held-out test score without its context.** The test
split is ~24 documents, so a single number swings by 8 points depending on the
seed. See [`09_INTERPRETING_THE_NUMBERS.md`](09_INTERPRETING_THE_NUMBERS.md) —
knowing this before a judge asks is worth more than a rounder headline.

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
   `bash run.sh all` on a spare laptop and get the same numbers in ~2 minutes.
2. **The evaluation is honest.** Held-out test set. Bootstrap confidence
   intervals. Sub-human conditions reported, not dropped. See
   [`07_EVALUATION_METHODOLOGY.md`](07_EVALUATION_METHODOLOGY.md).
3. **The offline claim is verifiable.** Not a bullet point — a demo where you
   turn on airplane mode mid-presentation.

---

## Part 3 — Pre-hackathon checklist (do these in the first 2 hours)

```
□  Clone the repo and run `bash setup_env.sh` on every laptop
□  Run `bash run.sh all` — confirm it reproduces the committed numbers
□  Run `bash run.sh test` — the suite must be green
□  Install the app on ONE iPhone and ONE Android phone (see 04_)
□  Run the demo twice end-to-end on both phones
□  Record a 90-second screen capture of the demo as a backup
□  Write the slide deck from Part 5
□  Assign: Demo / ML talk / Slide master / Backup operator
```

**The single most common way to lose:** a demo that only works on the
presenter's phone, on the presenter's wifi, with the presenter's phone
unlocked and the app already open. Rehearse the full cold start.

---

## Part 4 — The build schedule (36 hours)

### Hours 0–6: Make it true
- Everything in the pre-hackathon checklist.
- Run the evaluation on **all** capture profiles. Read `artifacts/eval_results.json`.
- Find the weakest profile. That is your next 4 hours.

### Hours 6–14: Fix the weakest thing
Only one thing. Resist the urge to redesign.
- If restoration is weak on `handheld_heavy` → more docs-per-profile in training.
- If classification is weak on `banking` → add banking templates to `corpus.py`.
- If OCR confidence is low everywhere → check preprocessing order in `evaluate.py`.

Re-run `bash run.sh eval` after every change. Commit the numbers, not just
the code.

### Hours 14–20: Make it demo-proof
- Cold-start test on both platforms, in airplane mode.
- Add the confidence-gated retake prompt — a demo that admits uncertainty
  looks more trustworthy than one that never fails.
- Prepare three physical test documents: a prescription, a bank statement,
  a letter. Laminate them. Have a backup set.

### Hours 20–28: Make it presentable
- Slides (Part 5).
- Rehearse the 60-second pitch 10 times. Time it.
- Prepare for the 4 questions in Part 6.

### Hours 28–34: Freeze
- **No new features after hour 30.** Only bug fixes and rehearsal.
- Final full run on a clean checkout: `git clone && bash setup_env.sh && bash run.sh all`.
- Back up the app folder and the trained models to a USB stick *and* the cloud.

### Hours 34–36: Submit
- README renders correctly on GitHub.
- The demo video is uploaded and the link works.
- Every team member can answer Part 6 questions.

---

## Part 5 — Slide deck (9 slides, 4 minutes)

**1. Problem** — 2.1 billion people with vision impairment; independence vs.
privacy is a forced trade-off in every existing app. One sentence, no more.

**2. Solution** — the pipeline diagram. Camera → restore → read → understand → speak.
Show the actual architecture image from `docs/01_ARCHITECTURE.md`.

**3. Why it's hard** — the failure mode. A shaky hand at 30 cm produces motion
blur that destroys the strokes of a digit. Show the `handheld_heavy` sample
from the contact sheet. This slide earns the ML slide.

**4. The models** — SightLineNet (126K params) + MiniLM (frozen 22.7M encoder,
49,796-param head). Training curves. Params on the slide.

**5. The data** — 174 document templates, 10 physically-motivated capture
profiles, leakage-controlled splits. This is the slide that separates you from
teams that ran `train_test_split` once.

**6. The results** — the table from `docs/03_RESULTS.md`. Include the confidence
intervals. Include the sub-human rows.

**7. It runs on a phone** — the payload table, the ONNX parity proof, the
airplane-mode screenshot.

**8. Privacy** — verifiable from source. Zero network. Why this matters for
medical and legal documents specifically.

**9. What's next** — one line. Then stop talking.

---

## Part 6 — Questions you will be asked

### "How do you know it's 95%+?"
Don't assert. Point at the file:
> "Held-out test set, never used for model selection. Bootstrap 95% CI is
> [x, y]. And here's the command to reproduce it: `bash run.sh eval`. The
> accuracy is field-level — all-or-nothing — because a user who hears the
> wrong dosage is worse off than one who hears nothing."

### "Isn't a synthetic dataset a weakness?"
Concede, then reframe:
> "Yes — our corpus is synthetic, and that's a real limitation. It's why we
> don't quote a number without an interval. What we did about it: the
> degradations are physically motivated (Poisson-Gaussian sensor noise,
> directional motion kernels, real bokeh discs), not Gaussian noise on a
> clean image. The next step, which we'd do with a week and a screen-reader
> user group, is fine-tuning on photographs they actually take."

### "Why MiniLM? Why not fine-tune the whole model?"
> "Because a frozen encoder can't catastrophically forget, it trains in
> seconds on CPU, and it exports to a 700 KB ONNX file. We can show the
> parity check proving the ONNX graph matches PyTorch to 1e-4."

### "What if the model is wrong about a dosage?"
> "That's why every stage carries a confidence score, and why low confidence
> triggers a retake prompt instead of a guess. A blind user who is told
> 'I couldn't read this, please retake' is safe. One who is told '400mg' when
> the label says 500mg is not. Safety is a UX problem, not just an accuracy
> problem."

---

## Part 7 — Submission checklist

```
□  README renders on GitHub (relative links work)
□  `bash setup_env.sh && bash run.sh all` works from a fresh clone
□  `bash run.sh test` is green
□  Every number in README/docs matches artifacts/eval_results.json
□  Demo video uploaded, link works in a private window
□  Live demo or video — whichever is required, both ready
□  Repo is public, has a licence
□  Two screenshots in the README (app UI + results table)
□  Team member names and roles listed
□  Submitted before the deadline — with 2 hours of margin
```

---

## Part 8 — If you are behind schedule

Cut in this order. Never cut in the reverse order.

1. Drop the `general` document class (3 classes instead of 4). −2 hours.
2. Reduce capture profiles from 10 to 5 for the training run. −1 hour.
3. Drop the restorer, ship the classical preprocessing stack + classifier. −4 hours.
4. Ship the desktop pipeline with a web demo instead of the phone app. −6 hours.

**Never cut:**
- The held-out test set and confidence interval. An unmeasured claim loses.
- The airplane-mode demo. It is the single most memorable moment you have.
- One clear, narrow problem. "Reads documents aloud, offline" beats
  "an AI accessibility platform" every time.
