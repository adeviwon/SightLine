# Interpreting the accuracy numbers

> Read this before quoting anything from `docs/03_RESULTS.md`.

This page exists because the seed check produced a result worth being precise
about, and burying it in a table would be the dishonest choice.

---

## What the seed check found

`bash seed_check.sh 5` retrains the classifier under five different seeds.
Each seed reshuffles the train/validation/test split, so each one scores a
*different set of held-out documents*.

| seed | held-out test accuracy |
|---|---|
| 123 (the default) | 100% |
| 121 | 95.8% |
| 122 | 91.7% |

The variance across splits is **8 percentage points**. On a 24-sample test set,
one flipped sample is 4.2 points — so the spread is essentially two samples'
worth of noise, and the three numbers are statistically consistent with a
model whose true accuracy is somewhere in the mid-90s.

**The 100% at seed 123 was a lucky split, not a real capability.** Reporting
only that number would have been reporting the best of three draws and calling
it a result.

---

## What to actually say

**Do not say:**

> "Our document classifier achieves 100% accuracy."

That number is real and reproducible — but only for one split, and it does
not describe the model's behaviour on a document it has never seen.

**Do say:**

> "On a held-out test set the classifier scores 92–100% depending on the
> split, with the variation coming from the 24-document test set rather than
> the model. Across five seeds the mean is X% and the worst case is Y%. The
> 20-fold cross-validation on the full corpus is 100% mean, which is the more
> stable estimate."

**On a slide, put the cross-validation number** — it averages over many more
folds and is far less sensitive to any single unlucky split. Keep the
seed-robustness table in the repository as the evidence behind it.

---

## Why this is a strength, not a weakness

A judge who asks "how do you know it's accurate?" is testing whether you
measured honestly. The team that answers:

> "Here is my held-out test score, here is a bootstrap confidence interval on
> it, here is five-fold cross-validation, and here is what happens when you
> change the seed. The honest answer is mid-90s, with an uncertainty of about
> eight points driven by test-set size."

is demonstrating exactly the competence the hackathon is selecting for. The
team that answers "100%" and cannot explain why has not measured anything.

---

## The fix, if there were time

More data. The corpus is 174 templates; each test split is ~24 documents. At
that size the confidence interval on accuracy is ±8 points regardless of how
good the model is. Tripling the corpus to ~500 templates would roughly halve
the interval.

This is the single highest-value change to the classifier, and it is on the
roadmap in [`00_HACKATHON_PLAYBOOK.md`](00_HACKATHON_PLAYBOOK.md) as the
first thing to do with any remaining time.

---

## A note on the end-to-end numbers

The same caution applies to the end-to-end field accuracy in
`docs/03_RESULTS.md`, with one important difference: **that metric is
generated from simulated capture images, not from text classification**, and
it is measured per capture profile rather than over a small resample. It is
also the number that actually matters, because it measures whether the right
dosage comes out of a blurry photo.

Read the per-profile table rather than the headline mean. If the model
succeeds on `handheld_light` and `low_light` and fails on `worst_case`, that
is a truthful and useful characterisation of the system.
