# 18 — The untried accuracy levers for a 942k-parameter CRNN, ranked and costed

> **What this page is.** The companion to `docs/17_OCR_ACCURACY_RESEARCH.md`, and it
> does not repeat it. doc 17 settles *which corpora we may legally use*, the
> *synth-then-finetune recipe*, *decoder families* and *architecture options at this
> data scale*. This page answers five different questions — closed-vocabulary
> decoding, pseudo-labelling, test-time methods that need no retraining, curriculum
> and document-specific augmentation, and the realistic accuracy ceiling with data
> volume — and then ranks what is actually worth doing to **this** repo, with an
> effect size, a cost in CPU-hours on a 4-core box, and a falsifiable acceptance
> test for each.
>
> **How to use it.** Every claim is either `[measured here]` with its file, or cited
> with a URL. Nothing is asserted from memory, and no number is extrapolated without
> saying so.
>
> **Companion tool:** `ml/src/pseudo_label.py` — the pseudo-labelling lever (§2),
> shipped as a measuring instrument that writes nothing by default.

## 0. Where we are, in five numbers

| fact | value | source |
|---|---|---|
| model | 942,166 params, 37-symbol charset, CTC, 3.84 MB ONNX | `docs/17` §1, §5 |
| training data | 52,554 crops from 420 SROIE receipts (294/63/63 by receipt) | `docs/03_RESULTS.md` §3 |
| held-out SROIE word accuracy / CER | **29.30% / 11.95%** (17,945 words) | `docs/03_RESULTS.md` §3 |
| hand-photographed CER / app end-to-end | **56.5% CER**; **0.0%** field recovery, 70.8% CER | `docs/03_RESULTS.md` §3, `docs/14` |
| our own rendered corpus | 22,800 crops / 1,500 pages — train 16,317 / val 3,213 / **test 3,270**, 4 mildest profiles, splits **by page** | `[measured here]` `data/render_corpus/index.json` |

Cost baseline throughout: **~250 s/epoch, 4 cores, no GPU** (`docs/03_RESULTS.md` §3).
So one 40-epoch run is **≈2.8 CPU-hours**, and every training item below is priced
against that.

---

## 1. Closed-vocabulary and constrained decoding

### 1.1 The lexicon-constrained vs lexicon-free gap, with the lexicon stated

The cleanest measurement is the *same* trained network decoded with and without an
output lexicon. CRNN (Shi et al.) reports exactly that, and the lexicon size is in
the column header ([ar5iv](https://ar5iv.labs.arxiv.org/html/1507.05717), Table 2):

| dataset | 50-word | 1k-word | 50k-word | Full | **lexicon-free** | gap, 50-word vs free |
|---|---|---|---|---|---|---|
| IIIT5K | **97.6** | 94.4 | — | — | **78.2** | **+19.4 pp** |
| SVT | **96.4** | — | — | — | **80.8** | **+15.6 pp** |
| IC03 | **98.7** | — | 95.5 | 97.6 | **89.4** | +9.3 pp (+6.1 pp at 50k) |

Three readings, and the third is the one that decides our plan:

1. **The gap is large** — up to 19.4 pp on IIIT5K, 15.6 pp on SVT.
2. **It shrinks as the lexicon grows and as the network gets better.** The same
   network is only 9.3 pp behind at a 50-word lexicon on IC03 but 6.1 pp at 50k —
   and the *lexicon-free* baseline is 89.4% there, not 78%. RARE reports the same
   shape with `50 / 1k / 50k / Full / None` columns
   ([CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf)).
   Roy et al. quantify the decay directly: **accuracy drops more than 10% when the
   lexicon grows from 50 to 1,000 words**, and IIIT5K ships a 0.5M-word lexicon
   ([HAL](https://inria.hal.science/hal-01088739/file/roy14.pdf)).
3. **A large part of a lexicon's gain is vocabulary *memorisation*, not reading.**
   Wan et al. measure CRNN at **91.1% on IIIT5K words that are IN its vocabulary
   against 68.7% on words OUT of it — a 22.5 pp gap — and the gap survives in every
   model family they test** (FAN 17.8 pp, CA-FCN 15.3 pp; ASTAR 74.6 Out-Voc.). Their
   fix is data, not decoding: mixing 25% random-string data lifts IIIT from 77.8% to
   84.4% ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_On_Vocabulary_Reliance_in_Scene_Text_Recognition_CVPR_2020_paper.pdf)).

`docs/17` §6 already rules out *an English dictionary*, and it is right: our
safety-critical tokens are `40218877`, `500mg` and `2024-CV-00456`. The number that
matters is therefore not 19.4 pp — it is how much of that gain survives when the
lexicon contains **our** field vocabulary and **no** English prose.

### 1.2 A restricted alphabet (digits only) and per-field grammars

The published evidence here is about **validity, not accuracy**, and it contains a
warning worth more than the promise.

- Grammar-constrained decoding on structured extraction lifts F1 from **0.062 to
  0.413** (and 0.102 to 0.470) on two low-resource clinical datasets
  ([Frontiers AI 2024](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1406857/full)).
  That is a *format* win: the grammar makes an extractable answer possible at all.
- **Constraining can destroy accuracy when the model's mass is not on valid
  continuations.** Their own note: "as models scale, unconstrained generation
  occasionally outperforms constrained decoding, aligning with recent theoretical
  work on bias introduced by constrained decoding"
  ([TU/e](https://research.tue.nl/en/publications/grammar-constrained-decoding-makes-large-language-models-better-l/)).
  The mechanism is explicit in DCCD: constrained decoding "can distort generation
  when the model assigns low probability mass to valid continuations, pushing
  decoding toward locally valid yet semantically incorrect trajectories" — their fix
  is an unconstrained draft *first*, then constrain, worth up to **+24 pp strict
  accuracy** over standard constrained decoding
  ([arXiv](https://arxiv.org/html/2603.03305v2)).
- The universal caveat, stated by the practitioners: constrained decoding
  "guarantees the output parses; it does not guarantee the output is correct"
  ([packet.ai](https://packet.ai/blog/structured-output-constrained-decoding)).

**Applied to us.** Our model is weak (11.95% CER on the *easy* domain, 56.5% on
photos), so its probability mass is frequently *off* the valid continuation — the
exact regime where naive constraint hurts. The lesson is not "don't constrain"; it
is **constrain last, and only where the constraint is a property of the field rather
than a guess about the token**: a sort code is `NN-NN-NN` whatever it reads, and a
money field is digits whose confusion set is `O↔0, I/S↔1/5, B↔8`. A field grammar is
a *repair* of a nearly-right read, and its downside is bounded by the grammar being
provably true. That is a different bet from trusting a dictionary.

**Expected effect for us.** Not 19.4 pp. The defensible expectation is **+3 to +10 pp
on field exact-match for numeric and structured fields**, because our failure mode is
exactly one wrong glyph per token (`docs/10` §5: `2024-CV-00456` read as
`2024-CV-00458` — one character, `6`→`8`) and a digits-only + grammar constraint can
only remove that class of error *if* it does not also delete the right answer.

**Already built, so this is cheap.** `ml/src/field_lexicon.py` holds the domain
lexicon *and* the field grammar (`PATTERNS`: money, date, dosage, frequency,
account, sort code, case ref, IBAN); `ml/src/ctc_decode.py` holds prefix beam search
with shallow fusion; `ml/src/eval_decode_ab.py` scores decoders **paired** on the same
forward pass with Wilson intervals and exact McNemar; `ml/src/compare_runs.py` turns
two artifacts into IMPROVED / REGRESSED / INCONCLUSIVE. The untried step is
constraining the *field* rather than the *character* — the code exists, the wiring
and the measurement do not.

> ⚠️ **Precondition, found while reading the code for this page.** `field_lexicon.py`
> `_domain_lines()` calls `RC.load()` **with no split argument**, so the lexicon is
> mined from the train **and val and test** labels. Any field-grammar evaluation that
> uses this lexicon is partly scoring the answer key. Fix the split before measuring —
> it is a one-line change and it is not in this lane's file list, so it is flagged here
> rather than done.

---

## 2. Pseudo-labelling / self-training on unlabelled images

### 2.1 It does work, on recognisers in our size class, and the method choice dominates

| method | model | effect | source |
|---|---|---|---|
| Pseudo-Label (PL) | CRNN | **+2.8 pp** | Baek et al. CVPR 2021 |
| Pseudo-Label (PL) | TRBA | +1.7 pp | same |
| PL + RotNet (PR) | CRNN | **+8.6 pp** (beats synthetic-pretrained baseline) | same |
| **Mean Teacher (MT)** | CRNN / TRBA | **−0.2 / −0.4 pp (HURTS)** | same |
| PL, 5%-labelled | CRNN | **+3.1 pp** (82.8 → 85.9 avg) | Zheng et al. CVPR 2022 |
| PL, cross-domain | TRBA | **+3.3 pp** (89.9 → 93.2) | same |
| uncertainty-aware PL selection | CRNN (+SeqCLR init) | **+4.5 / +2.9 pp** acc, −2.5 / −4.5 CER | Seq-UPS, WACV 2023 |

Sources: [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf),
[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Pushing_the_Performance_Limit_of_Scene_Text_Recognizer_Without_Human_CVPR_2022_paper.pdf),
[WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Patel_Seq-UPS_Sequential_Uncertainty-Aware_Pseudo-Label_Selection_for_Semi-Supervised_Text_Recognition_WACV_2023_paper.pdf).

The order of magnitude is **+3 to +9 pp**, and Mean Teacher — the obvious alternative —
is a measured *loss*. Related: the 10M unlabelled images of Union14M are worth
**+5.1 pp** to MAERec via self-supervised pre-training (73.5 → 78.6), and unlabelled
*real* data beats synthetic for that pre-training (78.6 vs 75.0)
([arXiv 2307.08723](https://ar5iv.labs.arxiv.org/html/2307.08723)).

### 2.2 The failure mode, and the standard guard

The failure mode is **confirmation bias**: wrong high-confidence predictions are
re-fed as ground truth and the model reinforces its own errors
(Lee 2013; [Arazo et al. 2020](https://arxiv.org/html/2206.14923v1) — pseudo-labelling
+ confirmation bias, and CTC/OCR models are specifically noted to be
**over-confident with overly sharp outputs**). The consequence is **concept drift**:
performance rises, then falls, because the target the model fits is its own output.

The standard guard is three-part, and it is the state of the art, not folklore:

1. **A confidence threshold on what is kept.** Working points in the literature
   cluster at **0.9 and 0.9995** ([Curriculum Labeling, AAAI 2021](https://cdn.aaai.org/ojs/16852/16852-13-20346-1-2-20210518.pdf)).
2. **A held-out validation set**, and — the guard people forget — **re-initialising
   the model** each self-training round rather than finetuning, which the same paper
   measures as a significant improvement "demonstrating an alternative and perhaps
   simpler solution to alleviate the issue of confirmation bias". Percentile-based
   selection beats a fixed threshold.
3. **A cap on the pseudo-label share.** The measured extreme: **unfiltered**
   pseudo-labels are worse than none (AT-ST at 100% is beaten by a filtering-free
   soft-CTC method that at least *reaches the quality of a finely-tuned 10% filter*)
   ([SoftCTC, arXiv 2212.02135](https://arxiv.org/html/2212.02135v3)). Below the
   threshold you are adding more label noise than signal, and Rizve et al. (2021)
   show **high-confidence-but-wrong labels amplify the bias more than low ones** —
   confidence is not a bug detector.

**What precision is required?** There is no single published "95%" theorem, and I will
not invent one. What the literature does support: filtering is *load-bearing* (100%
unfiltered is a measured loss, SoftCTC §2.1), the operating thresholds cluster at
0.9–0.9995, and precision falls off a cliff with model strength. The honest
engineering rule for our plan is **kept-set precision ≥ ~0.95 before training on it**,
stated as a heuristic operating point with the number printed, not as a result.

### 2.3 What this repo measured today

`ml/src/pseudo_label.py` (this lane) run on the frozen pre-render checkpoint,
threshold 0.90, first 200 train crops:

```
  KEPT                        42  (21.0% of scanned)
  dropped:  low-confidence   158
  PRECISION
    KEPT set       31.0% exact over 42 kept
    mean CER      0.147
    base rate       6.5% exact over 200 scanned (no filtering)
```

`[measured here]`. Two conclusions, and the second is the one that saves an argument:

1. **The threshold is doing real work** — 31.0% precision against a 6.5% base rate is
   a **4.8× lift** from discarding 79% of the crops.
2. **It is nowhere near enough.** 31% precision is roughly a *third* of the ~0.95 the
   literature's operating point implies. **Training on this set today would inject
   ~69% wrong labels and, per §2.2, is expected to make the model worse, not better.**

### 2.4 The trap specific to our situation — and why the renderer is the wrong target

We *can* generate unlimited unlabelled renders. That is a genuine capability, and it
is also a trap, because **for our own renderer the exact label is already free**:
`render_corpus.py` draws each label from the layout, so a rendered crop arrives with
ground truth at zero cost. Pseudo-labelling a render is therefore **strictly worse
than using the label sitting next to it** — you pay inference, you throw away the
truth, and you substitute a 31%-accurate guess.

The lever is real only where labels are **absent or untrustworthy**:

- the 193 hand-photographed pages, whose positional transcript pairing is a recorded
  negative (`HANDOFF.md` §3 — 71.6%→60.7% recall);
- real captures in a distribution we cannot label (the 56.5%-CER domain).

And here is the trap inside the trap: **the model is weakest exactly where we most
need pseudo-labels.** Precision is measured on renders (§2.3 gives the *upper bound*),
and on photographs the same threshold buys strictly less. So the correct sequencing is
**improve the model on labelled data first, then pseudo-label** — the reverse order
filters noise into a model that is already too weak to separate signal from it.

---

## 3. Test-time methods that need no retraining

| method | reported effect | model scale | source |
|---|---|---|---|
| TTA, aggregation study | **up to +2.5% top-1** over a single pass; beats mean-averaging at p = 2e-7 | ImageNet-scale | [Shanmugam, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf) |
| multi-scale {0.5…1.75} + flip, logit average | **+2.3 mIoU** | DeepLabV3+-scale | [SPACEMESHLAB](http://arxiv.org/pdf/2106.04025) |
| TTA on a *small* CRNN | **+0.34 pp** absolute — "variance reduction, not a shift in decision boundaries" | 6.26M-param CRNN, **but a lung-CT classifier, not OCR** | [Frontiers 2026](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1798168/full) |
| **TTA on a CTC OCR recogniser** | **"did not bring any improvement"** — tried both multi-scale prediction-selection and probability averaging before decoding | ultra-light OCR competitor | [Ultra Light OCR, arXiv 2110.12623](https://arxiv.org/html/2110.12623v1) |

**The honest reading for a 942k-parameter CTC model is: expect ~0, and there is a
published null on the closest analogue.** The large numbers in this literature are
from models with 100×–1000× our capacity, where the ensemble has genuinely diverse
members to average; §3's OCR-specific result is the relevant one, and it is a null on
a *CTC* recogniser — the same decoder family we ship.

Two findings that transfer regardless of the outcome:

- **Aggregation should be paired and flip-counted, not just averaged.** Shanmugam:
  "even when test-time augmentation produces a net improvement in accuracy, it can
  change many correct predictions into incorrect predictions." For a product that
  speaks a dosage to a blind user, an equal-and-opposite swap is not neutral.
- **The *training-time* cousin is not null.** Ultra Light OCR's own report shows the
  input scale move (32×320 → 48×480) as the large lever, and `docs/17` §5 already
  documents the width defect for us. TTA is the weak version of a change we can make
  properly.

**Expected effect for us: −0 to +1 pp, most likely indistinguishable from zero.** Cost
is low, and so is the information value — **but it is cheap to falsify**, which is why
it stays on the list at the bottom rather than being dropped.

---

## 4. Difficulty-ordered curriculum and document augmentation

### 4.1 The curriculum evidence is about *order*, and it is real

Curriculum learning (easy examples first) is an established principle
([Bengio et al., ICML 2009](https://icml.cc/Conferences/2009/papers/119.pdf), cited in
`docs/17` §3.3). The version the literature actually validates for vision models is
**synthetic-first, real-last**, and the effect sizes are large:

- **Pre-train on synthetic, finetune on real beats training on all sources jointly**,
  and joint training is sometimes *worse than real data alone* — "structuring the
  learning of the model leads to significant boost"; curriculum over synthetic-only
  training is reported at **+36.6 and +20 mIoU** on two datasets
  ([MTAP 2023](https://link.springer.com/article/10.1007/s11042-023-14662-0)).
- Fog adaptation curriculum: **34.6 → 46.8 mIoU** on Foggy Zurich-test
  ([CMAda](https://ar5iv.arxiv.org/html/1901.01415)).
- Direction matters: **synthetic-to-real beats real-to-synthetic** (61% vs 58–59%,
  71–72% vs 68–69%) in cross-domain action recognition
  ([arXiv 2601.14101](https://arxiv.org/html/2601.14101)).

For OCR specifically, `docs/17` §3.1 already carries the one number that matters and
its caveat: Baek et al. report **+2.2 pp in-distribution, −1.3 pp out-of-distribution**
from finetuning on real data, concluding the real data "is close to the test-time
distribution" or it does harm. **Those are different vision tasks, so treat the large
numbers as an existence proof that order matters, and the +2.2 pp as the
OCR-specific estimate.**

### 4.2 Augmentation policies that help *documents* (not scene text)

`docs/17` §3.2 already lists the families we cover. The genuinely untried ones are the
**ink / paper / scanner** families, and the reference library is Augraphy:

- Augraphy simulates physics-based document degradation — `InkBleed`,
  `DirtyScreen`, `Wrinkles`, `Faxify`, `BadPhotoCopy` — precisely the thermal-fade,
  show-through, ringing and speckle artefacts `realdata.py`'s header argues synthetic
  *scene* rendering cannot produce
  ([arXiv 2208.14558](https://arxiv.org/html/2208.14558v2)).
- **Its effect size is large when used as a test-set generator**: Tesseract word
  accuracy on Augraphy-noised documents was **52% lower on average, up to 84%**
  (same paper). That is a measurement of how much degradation space sits outside our
  nine capture profiles — not of training gain.
- A 2025–26 survey tabulates the training-side numbers and is candid about the size:
  stroke-level augmentation (FgAA) on IAM improves **WER 15.36 → 11.10, CER 5.51 →
  3.97**; VATr++ **CER 13.07 → 6.54**; DIG layout mAP **+9.02** — and concludes
  "Data Augmentation provides **consistent but modest** improvements, serving as a
  low-cost enhancement strategy"
  ([Beyond Human Annotation](https://arxiv.org/html/2601.12318v1)).
- Augraphy's own benchmark is worth reading before adopting it wholesale: the heavy
  transforms are slow — `BadPhotoCopy` at **0.17 img/s**, `ReflectedLight` at
  **0.06 img/s** on a 2-core Xeon ([README](https://github.com/sparkfish/augraphy)).
  On a 4-core CPU box this rules out applying them per-epoch; they must go in at
  **build time and be cached**, which is already this repo's rule
  (`build_dataset.py` header).

**And the strongest project-specific datum in this whole section is already ours:**
the real four-corner homography perspective augmentation was worth
**+3.05 pp accuracy / −3.95 pp CER** (`docs/03_RESULTS.md` §3). A *document* family
that matches the actual capture process is worth ~3 pp. That is the bar the untried
families have to clear — and the reason to expect "consistent but modest", not "large".

---

## 5. The realistic ceiling, and what data volume actually buys

### 5.1 The published curve

The only OCR-specific data-scaling law I could find is from Huawei Noah's Ark Lab,
fitted on a 62M-parameter TrOCR-B across **0.75M → 75M** training crops
([arXiv 2401.00028](http://arxiv.org/pdf/2401.00028v2)). Their fitted power law is
`E(D) = (1.84e5/D)^−0.3271` — the exponent is **−0.33**, i.e. error falls roughly as
the cube root of data. Their measured table (avg word accuracy over 6 open
benchmarks) is the useful part:

| training crops | avg accuracy | error |
|---|---|---|
| 0.75M (5%) | 38.14% | 61.86% |
| 1.50M (10%) | 42.33% | 57.67% |
| 3.75M (25%) | 71.47% | 28.53% |
| 7.50M (50%) | 72.40% | 27.60% |
| 15.00M (100%) | 83.91% | 16.09% |
| 75.00M (500%) | 91.47% | 8.53% |

**Read the top two rows again.** At **0.75M–1.5M crops a 62-million-parameter
transformer averages 38–42%** on open benchmarks. We have 52k crops and are at 29.30%
on a *narrower* test — which is not a failure, it is the low-data regime behaving
exactly as the law predicts. **Do not extrapolate that law down to 50k with
confidence.** I least-squares refitted their published table myself to see what the
extrapolation would claim, and the refit **badly mispredicts their own low-volume
points** (it puts 0.75M at ~82% against their measured 38.14%) — evidence that a single
power law does not span this range cleanly. I therefore report their *table* and their
*stated exponent* rather than my extrapolation, and the qualitative shape is
unambiguous: the curve is brutally steep below ~1M and flattening by ~15M.

### 5.2 What the ceiling looks like, measured by others

- **Hard real data with 4M labelled crops**: the best of 13 models reaches **74.6%
  average** on the Union14M challenge benchmark, and the authors observe the curve
  "increases sharply in the beginning and eventually levels out… adding more real data
  may not lead to significant performance gain" past that point
  ([arXiv 2307.08723](https://ar5iv.labs.arxiv.org/html/2307.08723)).
- **A CRNN's own ceiling on synthetic-heavy data**: 5.5M → 7.0M SynthText crops lifts
  CRNN **78.8% → 83.0%** — from *not discarding non-alphanumeric labels*, not from
  data volume — and real data adds a further **+1.3%**
  ([arXiv 2203.06696](https://arxiv.org/html/2203.06696v2)).

### 5.3 Our own arithmetic: how much CER the last 10 points cost

Word-exact falls off as `(1 − CER)^L` for a token of `L` characters. Computed for the
token lengths our fields actually have:

| token length | 80% exact needs CER | 90% exact needs CER |
|---|---|---|
| 4 (`TOTAL`) | 5.43% | 2.60% |
| 8 (`40218877`) | 2.75% | 1.31% |
| 12 (`2026-10-20`) | 1.84% | 0.87% |
| 18 (`2024-CV-00456`) | 1.23% | 0.58% |

Against our two measured CERs:

- **SROIE scans, 11.95% CER** → implies 36.1% exact at L=8 and 10.1% at L=18. Measured
  word accuracy is **29.30%**, so the model sits slightly *below* the independent-error
  prediction — realistic errors are correlated.
- **Hand-photographed, 56.5% CER** → implies **0.13%** exact at L=8. There is no
  coincidence where the current model reads a photographed dosage correctly, and the
  0.0% field recovery in `docs/14` is that arithmetic, not a bug.

**So the last 10 points to 80–90% word-exact require CER ≈ 1–3%** — a **4–10×
reduction** from the current 11.95% on the easy domain, and far more on photographs.

### 5.4 The answer, stated plainly

**For a from-scratch 942k-parameter CRNN on 50k–500k crops of structured documents:**

- On **our own rendered, closed-domain documents** (37 symbols, limited fonts, known
  layout), the realistic band is **~70–85% word-exact** *once the model is trained on
  the full render corpus and decoding is field-constrained*, because the closed
  alphabet and the field grammar remove error classes that open-benchmark models
  cannot. This is a **projection, not a measurement**, and §6 item 2 exists to measure
  it.
- On **photographed pages**, the published ceiling with a 4M-crop labelled diet is
  **74.6%**, and it takes a 62M-parameter transformer ~15M crops to average 84%. A
  942k CRNN on 500k crops is **more than an order of magnitude short of where the
  power law flattens**. Expecting 80–90% there from data volume alone is not supported
  by any source in this document.
- **500k crops does not buy the last 10 points to 90% on real photographs.** It buys
  perhaps half the error of 50k (exponent ≈ −0.33), which is real but not sufficient.
- The last 10 points are bought, in order, by: **(1) real labelled data volume**
  (no shortcut, and this is `HANDOFF.md` §4's own ranked blocker), **(2) constrained
  decoding + field grammar** on the numeric fields (§1 — buys token exact-match
  *without* touching CER), **(3) pseudo-labels on the label-poor real domain, and
  only after the model is strong enough to filter them (§2)**.

---

## 6. The ranked, costed plan for THIS repo

Each item: expected effect, cost in CPU-hours on 4 cores, and a **falsifiable**
acceptance test. "Cost" is wall-clock CPU on this box, where one 40-epoch run is
≈2.8 h and `docs/17`'s width-320 re-export is already ranked item 0 there and is **not
re-costed here**.

### 1. Field-grammar + digits-only constrained decoding (cheapest, no training)

- **Expected effect: +3 to +10 pp field exact-match** on numeric/structured fields.
  Bounded well below §1.1's 19.4 pp because our lexicon contains no English prose and
  our tokens are numbers. `docs/17` §6 correctly kills a char-LM (~0.3 pp); this is a
  different bet — a *provably-true* field constraint, per §1.2.
- **Cost: 3 CPU-hours** (wiring + two paired evaluations; no training, no new data).
- **Acceptance test:** with the `field_lexicon` split leak fixed, score the held-out
  render **test** split (3,270 crops, page-disjoint) with greedy vs field-constrained,
  then `bash run.sh py ml/src/compare_runs.py OLD.json NEW.json` must print
  **IMPROVED** — a disjoint Wilson interval, or exact McNemar p < 0.05. Anything else
  is recorded as INCONCLUSIVE and the constraint is not shipped. Report the
  correct→incorrect flip count alongside the net gain (§3).

### 2. Measure our own learning curve on rendered crops: 50k → 500k

- **Expected effect: this is the measurement that answers §5 for us**, not a gain in
  itself. The renderer gives unlimited *exactly labelled* crops (labels are drawn — 
  `render_corpus.py`), which is the one capability no other data source here has, so
  the curve is free to measure and directly falsifies or confirms §5.4's 70–85%
  projection.
- **Cost: ~8.5 CPU-hours** = three 40-epoch runs (~2.8 h each) at roughly 100k / 250k /
  500k rendered crops, plus ~0.5 h to build them. Run them sequentially, not
  concurrently — the box has 4 cores.
- **Acceptance test:** render test-split exact-match plotted against crop count; the
  curve must be monotone non-decreasing, and the **flattening point is the reported
  ceiling**. Falsifiable failure: if 500k is within noise of 100k (overlapping Wilson
  intervals) then renderer volume is *not* the binding constraint, capacity or
  regularisation is, and no further render volume should be spent.

### 3. Pseudo-labelling on the label-poor real domain — gated on measured precision

- **Expected effect: +2.8 to +8.6 pp word accuracy** (Baek CVPR 2021 PL +2.8 / PR +8.6;
  Zheng CVPR 2022 +3.1) — **when the kept-set precision clears ~0.95**. Below that the
  published failure mode is a *loss* (§2.2). Measured today: **31.0% precision at
  threshold 0.90** — so this item is **BLOCKED and must not be trained on yet**.
- **Cost: 2 CPU-hours** for the threshold→precision calibration sweep with
  `ml/src/pseudo_label.py` (`--dry-run` until the number is acceptable), **+2.8 h** for
  one 40-epoch retrain = **~5 CPU-hours**, all of it deferrable.
- **Acceptance test:** (a) the calibration sweep must reach **kept-set precision ≥0.95
  with a non-trivial kept count** on the render train split — if it cannot, stop and
  the lever is dead for this checkpoint; (b) then a paired McNemar on the held-out
  render test split, p < 0.05, with the pseudo-label share capped at the size of the
  real labelled set; (c) the guard from §2.2 — re-initialise rather than finetune, and
  record the precision that was actually achieved next to the gain.

### 4. Curriculum order + the untried document degradation families

- **Expected effect: +1 to +4 pp on the degraded profiles.** `docs/17`'s
  +2.2 pp OCR-specific estimate sets the centre; our own homography family is the
  measured comparator at **+3.05 pp**; the survey's "consistent but modest" is the
  ceiling on ambition. Two *distinct* untried moves, and they should be measured
  separately: (a) **order** — clean renders → degraded profiles → real captures, per
  §4.1's synthetic-to-real finding; (b) **families** — Augraphy's ink/paper/scanner
  set, which covers degradation our nine capture profiles do not (52% word-accuracy
  drop as a test generator).
- **Cost: ~7 CPU-hours** — ~4 h to extend `capture`/`build_dataset` with the new
  families *at build time, cached* (Augraphy's heavy transforms run at 0.06–0.17
  img/s, so per-epoch augmentation is not affordable on 4 cores), + 2.8 h for one
  40-epoch run, + the A/B.
- **Acceptance test:** per-profile exact-match on the render test split vs the current
  corpus, **at equal epochs**. The win must be concentrated in the **targeted**
  profiles (a flat shift means something else changed); `worst_case` stays excluded as
  sub-human; and the order experiment is one variable per run, because the epoch-20-vs-39
  comparison is the only clean A/B this project has.

### 5. Test-time augmentation (ranked last: cheap to falsify, published null)

- **Expected effect: −0 to +1 pp, most likely indistinguishable from zero.** §3's only
  OCR-specific TTA result on a CTC recogniser is a null, and the large published
  numbers are on 100×–1000× larger models. It is ranked last **because it is cheap to
  falsify**, not because it is promising.
- **Cost: 1.5 CPU-hours** (a paired harness over the 3,270-crop render test split,
  multi-scale + crop, logit averaging; no retraining).
- **Acceptance test:** paired exact-match with a Wilson 95% interval against greedy on
  the *same forward passes*; if the intervals overlap, record **INCONCLUSIVE and stop**
  — a negative result here is a success, it closes a literature-recommended path. Both
  flip counts (correct→incorrect and incorrect→correct) must be reported, per §3.

**Explicitly excluded from this plan** (see §7) — none of the above re-runs anything
`docs/17` §6 or the recorded negative results already closed.

---

## 7. Already settled — do NOT re-run these

**Settled by `docs/17` (do not repeat):** dataset licensing (MJSynth/SynthText are
non-commercial and unusable; IAM/IIIT5K/SVT/CUTE80 have no licence text; hiertext's
share-alike is a stricter gate than non-commercial); the synth-then-finetune recipe and
Baek's +2.2/−1.3 pp result; decoder *families* (a character-level LM inside beam search
is worth ~0.3–0.5 pp CER — not worth two decoders to maintain); architecture options at
this scale (transformer needs 10⁶–10⁷ crops, attention's gain is largely an implicit
char-LM, input height 48 is not the lever, width 320 is already item 0); a large English
dictionary (nothing to say about `40218877`).

**Settled by measurement in this repo (all recorded, all reverted):**

1. **Six line-detector heuristics** — 0.610 / 0.195 / 0.198 / 0.242 / 0.716 / 0.817
   bands-per-line; the best raised the ceiling (67.9% → 71.1% recall) and **degraded
   CER (56.5% → 61.5%)**, so it was reverted. Variants indistinguishable by McNemar
   on 290 lines (p = 1.000) — a sample-size problem, not a tuning problem
   (`ml/src/eval_ocr.py`).
2. **Finetuning on positional-label crops bootstrapped from photographs** — 71.6% →
   60.7% recall, 2.4% → 1.2% line-exact (`HANDOFF.md` §3). This is the **direct
   precedent for pseudo-labelling**: the labels were wrong, not the idea.
3. **Chunking long lines** — `HSBC UK BANK PLC` 98% squeezed vs 54% chunked; reverted
   (`app/tools/crop_contract.js`).
4. **Word segmentation of a line** — 1,007 words found vs 965 expected, but
   **zero lines rescued and five lines lost**; McNemar χ² = 3.20, p ≈ 0.07
   (`ml/src/word_segment.py`).
5. **Image restoration as an OCR boost** — on the CRNN arms `raw` ties or beats the
   restorer (`docs/03_RESULTS.md` §3).
6. **Tesseract PSM sweeps** — 336 configurations recovered **no field at all**
   (`docs/10_ACCURACY_GAP.md` §3). Tesseract is gone from the app.
7. **Detector micro-tuning in general** — the honest ceiling is data, not heuristics
   (`HANDOFF.md` §4 item 3).
8. **Upsampling to fix the three resolution-bound pages** — 12–17 px lines cannot be
   upsampled back into existence.

**Newly identified in this lane — fix before it invalidates a measurement:**
`field_lexicon.py::_domain_lines()` mines from `RC.load()` **without a split**, so the
field lexicon is built from train **+ val + test** labels. Item 1's evaluation is
partly self-scoring until that is fixed.

---

## 8. Sources

1. Shi, Bai, Yao, *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (CRNN), Table 2 lexicon columns — https://ar5iv.labs.arxiv.org/html/1507.05717
2. Shi et al., *Robust Scene Text Recognition With Automatic Rectification* (RARE), CVPR 2016, `50/1k/50k/Full/None` table — https://openaccess.thecvf.com/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf
3. Roy et al., *Scene Text Recognition and Retrieval for Large Lexicons* (50→1000 lexicon costs >10%; 0.5M lexicon) — https://inria.hal.science/hal-01088739/file/roy14.pdf
4. Wan et al., *On Vocabulary Reliance in Scene Text Recognition*, CVPR 2020 (In-Voc 91.1 vs Out-Voc 68.7 for CRNN; random-string mixing 77.8→84.4) — https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_On_Vocabulary_Reliance_in_Scene_Text_Recognition_CVPR_2020_paper.pdf
5. *Grammar-constrained decoding for structured information extraction*, Frontiers in AI 2024 (F1 0.062→0.413) — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1406857/full
6. *Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers*, TU/e (constraints can hurt larger models) — https://research.tue.nl/en/publications/grammar-constrained-decoding-makes-large-language-models-better-l/
7. *Draft-Conditioned Constrained Decoding* (constrained decoding can distort generation; +24 pp strict accuracy when drafted first) — https://arxiv.org/html/2603.03305v2
8. *Structured Output from LLMs: JSON Mode, Grammars and Constrained Decoding* (validity ≠ correctness) — https://packet.ai/blog/structured-output-constrained-decoding
9. Baek et al., *What If We Only Use Real Datasets for Scene Text Recognition?*, CVPR 2021 (PL +2.8/+1.7; PR +8.6/+2.7; MT −0.2/−0.4) — https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf
10. Zheng et al., *Pushing the Performance Limit of Scene Text Recognizer Without Human Annotation*, CVPR 2022 (CRNN +3.1 pp; cross-domain +3.3 pp) — https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Pushing_the_Performance_Limit_of_Scene_Text_Recognizer_Without_Human_CVPR_2022_paper.pdf
11. Patel et al., *Seq-UPS: Sequential Uncertainty-Aware Pseudo-Label Selection*, WACV 2023 (+4.5/+2.9 pp acc; −2.5/−4.5 CER) — https://openaccess.thecvf.com/content/WACV2023/papers/Patel_Seq-UPS_Sequential_Uncertainty-Aware_Pseudo-Label_Selection_for_Semi-Supervised_Text_Recognition_WACV_2023_paper.pdf
12. *SoftCTC — Semi-Supervised Learning for Text Recognition using Soft Pseudo-Labels* (filtering is load-bearing; unfiltered is worse; over-confidence of CTC/OCR models) — https://arxiv.org/html/2212.02135v3
13. *Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning*, AAAI 2021 (thresholds 0.9/0.9995; re-init beats finetune; percentile selection) — https://cdn.aaai.org/ojs/16852/16852-13-20346-1-2-20210518.pdf
14. Arazo et al., *Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning* (confirmation bias; high-confidence-but-wrong amplifies it, after Rizve et al. 2021) — https://arxiv.org/html/2206.14923v1
15. Shanmugam et al., *Better Aggregation in Test-Time Augmentation*, ICCV 2021 (+2.5% top-1; TTA can flip correct→incorrect) — https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf
16. *SpaceMeshLab* (multi-scale + flip TTA, +2.3 mIoU) — http://arxiv.org/pdf/2106.04025
17. *Learning under extreme data scarcity: an enhanced hybrid CRNN* (TTA on a 6.26M-param CRNN = +0.34 pp — **a lung-CT classifier, not OCR**, cited only for the small-model TTA magnitude) — https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1798168/full
18. *Ultra Light OCR Competition Technical Report* (**TTA brought no improvement** on a CTC recogniser; input scale 32×320→48×480) — https://arxiv.org/html/2110.12623v1
19. *Large OCR Model: An Empirical Study of Scaling Law for OCR* (E(D) exponent −0.327; data-volume table 0.75M→75M) — http://arxiv.org/pdf/2401.00028v2
20. Jiang et al., *Revisiting Scene Text Recognition: A Data Perspective* / Union14M (4M labelled; SOTA 74.6%; curve levels out) — https://ar5iv.labs.arxiv.org/html/2307.08723 · https://arxiv.org/abs/2307.08723
21. *Training Protocol Matters: Towards Accurate Scene Text Recognition via Training Protocol Searching* (CRNN 78.8→83.0) — https://arxiv.org/html/2203.06696v2
22. Groleau et al., *Augraphy: A Data Augmentation Library for Document Images* (52% word-accuracy drop as test generator; img/s benchmarks) — https://arxiv.org/html/2208.14558v2 · https://github.com/sparkfish/augraphy
23. *Beyond Human Annotation: Recent Advances in Data Generation Methods for Document Intelligence* (FgAA WER 15.36→11.10; "consistent but modest") — https://arxiv.org/html/2601.12318v1
24. Gaya et al., *On exploring weakly supervised domain adaptation strategies* (synthetic→real curriculum beats joint training) — https://link.springer.com/article/10.1007/s11042-023-14662-0
25. Sakaridis et al., *Curriculum Model Adaptation with Synthetic and Real Data* (mIoU 34.6→46.8) — https://ar5iv.arxiv.org/html/1901.01415
26. *Curriculum-Based Strategies for Efficient Cross-Domain Action Recognition* (synthetic→real beats real→synthetic) — https://arxiv.org/html/2601.14101
27. Bengio et al., *Curriculum Learning*, ICML 2009 — https://icml.cc/Conferences/2009/papers/119.pdf (also cited in `docs/17`)

In-repo evidence: `ml/src/pseudo_label.py` (**new, this lane**), `ml/src/field_lexicon.py`,
`ml/src/ctc_decode.py`, `ml/src/build_lm.py`, `ml/src/eval_decode_ab.py`,
`ml/src/compare_runs.py`, `ml/src/render_corpus.py`, `ml/src/eval_render_corpus.py`,
`ml/src/word_segment.py`, `ml/src/eval_ocr.py`, `ml/src/build_dataset.py`,
`ml/src/ocr_model.py`, `ml/src/capture.py`, `data/render_corpus/index.json`,
`docs/03_RESULTS.md`, `docs/10_ACCURACY_GAP.md`, `docs/14_HACKATHON_BENCHMARK.md`,
`docs/17_OCR_ACCURACY_RESEARCH.md`, `HANDOFF.md`, `app/tools/crop_contract.js`.
