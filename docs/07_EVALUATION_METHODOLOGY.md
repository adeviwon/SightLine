# Evaluation methodology

This document exists so that every accuracy claim in this repository can be
**checked, not believed**. If a judge disagrees with a number, this page
explains how it was produced and how to reproduce it.

Regenerate everything:

```bash
bash run.sh eval
# → artifacts/eval_results.json
bash run.sh errors
# → per-sample error listing
```

---

## 1. The metric that matters

**Field accuracy** — the fraction of documents where **every** key field is
recovered correctly.

```
prescription → ["500mg", "400mg"]
banking      → ["40218877", "40-11-04"]
legal        → ["2500", "2024-CV-00456"]
```

All-or-nothing. This is deliberate and it is the single most important
methodological choice in the project.

**Why not character accuracy or word accuracy?** Because those metrics are
comfortable and misleading. A document where OCR produced 98% of characters
correct but turned `500mg` into `50O0mg` scores beautifully on word accuracy
and would get someone hurt. The user of this app cannot see the label and
cannot check. The only question that matters is: *did we recover the actual
dosage, the actual account number?* So that is what we score.

The supporting metrics — `word_accuracy`, `mean_ocr_confidence`,
`exact_dosage_rate` — are reported alongside so the failure modes are
diagnosable, but `field_accuracy` is the headline.

---

## 2. The comparison arms

Four arms are run on identical inputs. A model that does not beat the
classical baseline has not earned its place in the pipeline.

| Arm | What it does |
|---|---|
| `raw` | OCR the degraded image directly. The floor. |
| `classical` | median 3×3 denoise → unsharp mask. The hand-tuned stack every tutorial ships. |
| `restorer` | SightLineNet only. Isolates the learned contribution. |
| `restorer_clahe` | SightLineNet + CLAHE + Otsu. **The production path.** |

Reporting all four is the point: `restorer` alone shows what the network
contributes, and `classical` shows what it adds to.

---

## 3. Capture profiles

Ten profiles in `ml/src/capture.py`, each built from a physical analogue
rather than a convenient parameter sweep:

| Profile | Physical cause | Typical trigger |
|---|---|---|
| `studio_clean` | — | scanner, flatbed, perfect light |
| `handheld_light` | mild defocus + short motion smear | steady hand, good light |
| `handheld_heavy` | long directional motion blur | shaky hand, one-handed grip |
| `low_light` | high-ISO noise + reduced contrast | dim room |
| `hand_shadow` | hard-edged shadow from the holding hand | any indoor use with a single light source |
| `off_axis` | perspective from a tilted camera | not looking at the screen while framing |
| `glossy_glare` | specular hotspot on glossy paper | flash or overhead light on coated stock |
| `jpeg_social` | aggressive JPEG compression | the photo passed through a share sheet |
| `paper_texture` | visible paper grain + noise | photographed at low light on matte stock |
| `worst_case` | **everything at once** | the genuinely hard case |

`worst_case` is declared `SUB_HUMAN` in `ml/src/evaluate.py` — a person cannot
reliably read that image either. See §5.

Each profile has individually validated behaviour (asserted in
`ml/tests/test_capture.py`): motion blur at different angles produces
different results, higher ISO produces more noise, and every profile actually
changes the image.

---

## 4. Leakage control

Three separate places where data can leak from train into evaluation, and what
stops each one:

| Leak | Consequence | Control | Test |
|---|---|---|---|
| Duplicate templates across train/val | inflated accuracy | deduplicate before splitting | `test_corpus.py::test_no_duplicate_templates` |
| Same document's patches in train and val | inflated PSNR | split by **document seed** | `test_leakage.py::test_split_by_seed` |
| Augmented twin in train, clean original in val | inflated accuracy | augment **after** splitting | `test_leakage.py` |

`ml/tests/test_leakage.py` is the file to read if you want to check that this
project is not fooling itself.

---

## 5. Honest reporting rules

These are enforced in code, not left to judgement:

**Sub-human conditions are excluded from the headline and reported
separately.** `SUB_HUMAN` in `evaluate.py` names the profiles a person cannot
read. `test_evaluate.py` asserts that no sub-human profile leaks into the
headline mean. The result: the headline is an average over conditions a user
could reasonably expect to succeed at, and the sub-human row is printed
separately with its actual (poor) numbers.

**The test set is touched once.** The classifier's epoch is selected on
validation. The headline number comes from a held-out test set that plays no
part in training or selection.

**Confidence intervals accompany every accuracy.** A 24-sample test set has a
±9 point interval at 95%. Reporting "100% accuracy" without that interval
would be dishonest, so the bootstrap CI is computed and printed.

**Cross-validation for stability.** One 24-sample split is a coin flip. Twenty
disjoint folds give a mean and a standard deviation.

**Everything is seeded.** `--seed 123` end to end. Data generation, splitting,
network initialisation, augmentation. Same numbers on any machine.

---

## 6. What would falsify these claims

Stated in advance, so the evaluation can be attacked rather than admired:

1. **Real photographs.** The corpus is synthetic. A validation set of
   photographs taken by actual users would probably show lower field accuracy,
   especially on `hand_shadow` and `off_axis`. This is the most likely way
   the numbers move, and it is the right thing for the team to do next.
2. **Chinese-language documents.** The OCR model shipped is English only. The
   target population in Hong Kong reads Traditional Chinese. **This is a
   significant gap and should be addressed before claiming HK-market
   readiness.**
3. **Print-quality documents.** These are rendered with DejaVu Sans. Real
   documents use serif faces, handwriting, tables, and logos. Accuracy on a
   photocopied letter will be lower than on a rendered one.
4. **A larger model.** SightLineNet is 126K parameters, chosen so it trains on
   a CPU in minutes. A larger restoration network would beat it on the harder
   profiles and would still fit on a phone.

None of these are reasons not to ship. They are the reasons to be precise about
what was measured.
