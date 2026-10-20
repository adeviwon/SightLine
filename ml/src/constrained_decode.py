"""
constrained_decode.py — decode the CRNN with the alphabet restricted to the
symbols a field can plausibly contain, and measure what that buys.

    bash run.sh py ml/src/constrained_decode.py --ab --limit 400

THE FAILURE THIS ATTACKS
------------------------
The shipped CRNN emits 37 symbols (0-9A-Z space) and its dominant error mode is
letter/digit confusion, because the two are the same shape in most typefaces:

    PRESCRIPTION  -> PRE5CR1P710N
    DATE01102026  -> 0A7E01102026      (S->5, T->7, O->0, I->1, G->6)

Most of the values this product must read are NUMERIC: account numbers, sort
codes, dates, dosages. A letter emitted inside one of those is not a probabilistic
near-miss to be averaged away, it is a wrong field — "the user who hears the
wrong dosage is worse off than one who hears none" (make_results_doc.py). For
those fields the ambiguity set is not 37 symbols; it is 10, and the confusion
mode can be REMOVED rather than trained away.

EXACT MASKING, NOT POST-HOC FILTERING
-------------------------------------
`mask_logits` sets every disallowed class to -inf BEFORE the decode, so the
search can never reach one however large its logit was. Two tempting shortcuts
are wrong:

  * filtering the output ("drop the letters") needs a delete/insert policy and
    is a heuristic: on "0A7E01102026" dropping the A and E leaves 0-7-0-1-1-0-2-0-2-6,
    which is one character too long and still wrong;
  * mapping letters to digits via `realdata.norm_label`'s confusion table is the
    SCORER's policy, reserved for scoring and explicitly not for training; it
    bakes in a fixed O->0/I->1 table and cannot express "this field may contain a
    separator but not a letter".

The mask is applied to the logits, the collapse is untouched, and the result is
exactly the argmax/beam over the restricted alphabet.

THE DECODE RULE IS DELIBERATELY THE SHIPPED ONE
----------------------------------------------
`decode_constrained(..., beam=1)` uses the SAME rule as `realdata.decode_greedy`
(collapse repeats, drop blanks) — verified by a test that asserts the
unconstrained case reproduces it character for character. That is what makes the
A/B PAIRED: unconstrained vs digits-only differ in the mask and in nothing else.
With beam=1 the score is the log-probability of the greedy path, so it is
comparable across the two masks; with beam=N it is the CTC prefix probability
over the restricted alphabet (`ctc_decode.beam_search`'s semantics — see that
module for why the prefix beam, and for the doubled-character bookkeeping a mask
must not disturb).

THE CONSTRAINT IS PER FIELD, WHICH IS WHY `allowed_for_kind` EXISTS
------------------------------------------------------------------
Same recogniser, same logits, different mask per field. `allowed_for_kind`
returns the symbols a kind can plausibly contain (kinds follow the product's
field grammar, field_lexicon.PATTERNS); symbols the model's charset cannot emit
(a '/' or '.' — this charset has neither) are simply dropped by the mask, and
`decode_constrained` REFUSES an empty intersection instead of returning "".

WHAT IS MEASURED (`--ab`)
-------------------------
Paired on identical logits, one forward pass, two decodes:
unconstrained greedy vs digits-only constrained, scored with the repo's exact
match, Wilson 95% intervals, character-weighted CER and an exact McNemar over
the discordant pairs (ctc_decode.wilson / pair_counts / mcnemar_exact, so the
number is comparable with every other accuracy experiment in this repo).

The holdout is FROZEN (`models/ocr/crnn_before_render.pt`) because
`models/ocr/crnn.pt` is rewritten in place by a retrain and a torn read of it
produced a 0/300, CER 0.976 report once already; the loader below is
eval_render_corpus/eval_decode_ab's guarded one, not a bare torch.load.

THREE HOLDOUT SOURCES, AND WHY
------------------------------
The filter this lane was asked for is "crops whose label is a SINGLE TOKEN and
whose token is entirely digits". On the render corpus that set is EMPTY: its
labels are whole rendered LINES ("DATE 01102026", "PATIENT JANE DOE DOB
04021979"), never bare tokens, so there is not one crop in the held-out render
split whose label is digits only. `--source auto` therefore reports that empty
intersection as a finding and measures the same filter on the two numeric sets
that do exist:

  render   the frozen split, digit-only single-token crops   -> 0 (reported)
  sroie    the frozen SROIE word-crop test split             -> real receipts,
           one token per crop, held out from this checkpoint's training split
  probe    all-digit tokens from capture.DOCS rendered through render_corpus's
           own pipeline (render_document + apply_profile + page_crops): NOT a
           frozen split, and labelled as such wherever it is printed, but the
           only render-domain numeric-token set that exists on disk.

Read the three together: SROIE says whether a model that reads its own domain
well has anything left for the mask to fix, and the probe says what the mask
buys on the documents this product actually serves.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctc_decode as CD   # noqa: E402
import realdata as R      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FROZEN_CKPT = ROOT / "models" / "ocr" / "crnn_before_render.pt"
SROIE_CACHE = ROOT / "artifacts" / "ocr_train.npz"

NEG = float("-inf")          # the mask value: unreachable, exactly
DIGITS = "0123456789"

# Class 0 is the CTC blank (realdata.BLANK); charset[k] is class k+1.
BLANK = R.BLANK

# ── per-field-kind alphabets ──────────────────────────────────────────────────
# The INTENT, not the intersection with the model's charset: a kind may name a
# separator the charset cannot emit (this charset has no '.', '-' or '/'), and
# mask_logits drops those silently rather than making the kind unrepresentable.
# Field kinds and their grammar come from field_lexicon.PATTERNS, so the mask
# and the extractor agree on what a sort code or a dosage looks like.
KINDS = {
    "sort_code": DIGITS,                       # NN-NN-NN: the '-' is not emitted
    "acct":      DIGITS,                       # account / card numbers
    "date":      DIGITS + "-/",                # DD/MM/YYYY, DD-MM-YYYY
    "money":     DIGITS + ".,-+£$€",           # thousands, decimals, sign
    "dosage":    DIGITS + ".",                 # 500mg, 2.5ml
}
ALIASES = {"account": "acct", "account_number": "acct", "sort": "sort_code",
           "sortcode": "sort_code", "sort_code_number": "sort_code",
           "amount": "money", "total": "money", "cost": "money",
           "dose": "dosage", "strength": "dosage"}


def allowed_for_kind(kind):
    """The symbol subset for one field kind. Raises on an unknown kind.

    Raising rather than defaulting to "everything" is the point: a typo'd kind
    that silently decoded unconstrained would produce a report claiming a
    constraint it never applied.
    """
    key = ALIASES.get(str(kind).strip().lower(), str(kind).strip().lower())
    if key not in KINDS:
        raise ValueError("unknown field kind %r; known kinds: %s"
                         % (kind, sorted(KINDS)))
    return KINDS[key]


# ── masking + decoding ────────────────────────────────────────────────────────

def _as_log_scores(scores):
    """[T, C] -> log-scores. A probability row is detected and log-ed.

    Detected rather than declared so a caller can hand over either the model's
    logits or a softmax without a flag: with probabilities the mask (-inf) and
    the beam's log-add sums are still correct after the log.
    """
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a [T, C] score matrix, got shape %r" % (a.shape,))
    if a.size and a.min() >= 0.0 and np.allclose(a.sum(axis=1), 1.0, atol=1e-3):
        a = np.log(np.clip(a, 1e-30, None))
    return a


def _keep_classes(charset, allowed, space):
    """Resolve (charset, allowed, space) -> sorted class indices that survive."""
    want = set(allowed) | ({" "} if space else set())
    hits = {charset.index(s) + 1 for s in want if s in charset}
    if not hits:
        raise ValueError(
            "infeasible mask: none of %r is in charset %r, so the decode could "
            "only ever emit blanks" % ("".join(sorted(want)), charset))
    return np.array(sorted(hits | {BLANK}), dtype=np.int64)


def mask_logits(scores, charset, allowed, space=False):
    """`scores` [T, C] with everything outside (allowed + blank [+ space]) = -inf.

    C must be len(charset) + 1 with class 0 the blank, the model's convention.
    """
    a = _as_log_scores(scores)
    if a.shape[1] != len(charset) + 1:
        raise ValueError("charset %r implies %d classes, scores have %d"
                         % (charset, len(charset) + 1, a.shape[1]))
    keep = _keep_classes(charset, allowed, space)
    out = np.full(a.shape, NEG, dtype=np.float64)
    out[:, keep] = a[:, keep]
    return out


def _i2c(charset):
    return {i + 1: c for i, c in enumerate(charset)}


def _greedy_text(lp, i2c):
    """Collapse repeats then drop blanks — realdata.decode_greedy's rule."""
    idx = lp.argmax(-1)
    out, prev = [], -1
    for i in idx:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(i2c.get(i, ""))
        prev = i
    return "".join(out)


def _restricted_beam(lp, charset, beam):
    """CTC prefix beam over the masked matrix -> (best text, log probability).

    Same bookkeeping as ctc_decode.beam_search (two scores per prefix, so a
    doubled character survives); the only difference is the alphabet, which here
    is whatever the mask left alive.
    """
    i2c = _i2c(charset)
    T, C = lp.shape
    beams = {"": [0.0, NEG]}
    for t in range(T):
        row = lp[t]
        l_blank = row[BLANK]
        nxt = {}
        for prefix, (p_b, p_nb) in beams.items():
            p_tot = CD._lae(p_b, p_nb)
            e = nxt.setdefault(prefix, [NEG, NEG])
            e[0] = CD._lae(e[0], p_tot + l_blank)
            for c in range(1, C):
                lpc = row[c]
                if lpc == NEG:              # masked: unreachable, not merely low
                    continue
                ch = i2c.get(c)
                if ch is None:
                    continue
                if prefix and ch == prefix[-1]:
                    # repeat: from a blank it EXTENDS, from the char it COLLAPSES
                    ext = nxt.setdefault(prefix + ch, [NEG, NEG])
                    ext[1] = CD._lae(ext[1], p_b + lpc)
                    e[1] = CD._lae(e[1], p_nb + lpc)
                else:
                    ext = nxt.setdefault(prefix + ch, [NEG, NEG])
                    ext[1] = CD._lae(ext[1], p_tot + lpc)
        if len(nxt) > beam:
            nxt = dict(sorted(nxt.items(),
                              key=lambda kv: -CD._lae(kv[1][0], kv[1][1]))[:beam])
        beams = nxt if nxt else beams
    ranked = sorted(beams.items(), key=lambda kv: -CD._lae(kv[1][0], kv[1][1]))
    if not ranked:
        return "", NEG
    text, (pb, pnb) = ranked[0]
    return text, CD._lae(pb, pnb)


def decode_constrained(scores, charset, allowed, beam=1, space=False):
    """Restricted CTC decode -> (text, score).

    `scores`   [T, C] logits or probabilities, class 0 = blank.
    `charset`  the emittable symbols, index k -> class k+1 (realdata.CHARSET).
    `allowed`  symbols that may be emitted; anything else is masked to -inf.
    `beam`     1 = greedy (the shipped rule, score = log-prob of that path);
               N > 1 = restricted prefix beam, return the best hypothesis.
    `space`    also permit the space class (off by default: a numeric field with
               a space in it is a different field).

    Raises ValueError on an empty/infeasible mask rather than returning "".
    """
    lp = mask_logits(scores, charset, allowed, space=space)
    if int(beam) <= 1:
        idx = lp.argmax(-1)
        score = float(sum(lp[t, int(idx[t])] for t in range(lp.shape[0])))
        return _greedy_text(lp, R.I2C if charset == R.CHARSET else _i2c(charset)), score
    return _restricted_beam(lp, charset, int(beam))


# ── the paired measurement ────────────────────────────────────────────────────

def _render_digit_crops(split, limit):
    """Held-out render crops whose label is a single, entirely-digit token."""
    import build_dataset as B
    import render_corpus as RC
    from PIL import Image

    data = RC.load(split=split)
    n_all = len(data["labels"])
    idx = [i for i, l in enumerate(data["labels"])
           if str(l).isdigit() and len(str(l).split()) == 1]
    n_match = len(idx)
    idx = idx[:limit]
    arrs = [B.crop_to_array(Image.fromarray(np.asarray(data["crops"][i], np.uint8)))
            for i in idx]
    refs = [str(data["labels"][i]) for i in idx]
    return arrs, refs, ("render_corpus %s split: %d crops, %d with a single-token "
                        "all-digit label" % (split, n_all, n_match))


def _sroie_digit_crops(split, limit):
    """Held-out SROIE word crops whose transcript is entirely digits.

    `artifacts/ocr_train.npz` is packed by build_dataset.py with a
    receipt-level split; the model was trained on its `train` rows, so `test`
    rows are unseen documents. Crops are pre-cut tokens — this is the one
    holdout on disk where "single token, all digits" is actually populated.

    The members are bound ONCE, outside the loop: numpy's NpzFile is lazy and
    `z["images"]` re-reads and re-decompresses the whole 500 MB member on every
    __getitem__ — measured, 282 accesses pulled 8 GB off disk and had not
    finished after twelve minutes.
    """
    with np.load(SROIE_CACHE, allow_pickle=True) as z:
        labels, splits = z["labels"], z["splits"]
        widths, images = z["widths"], z["images"]
        idx = [i for i in range(len(labels))
               if str(splits[i]) == split and str(labels[i]).isdigit()]
        n_match = len(idx)
        arrs = [np.ascontiguousarray(images[i, :, :int(widths[i])])
                for i in idx[:limit]]
        refs = [str(labels[i]) for i in idx[:limit]]
    return arrs, refs, ("SROIE held-out word-crop %s split: %d crops with an "
                        "all-digit label" % (split, n_match))


def _render_token_crops(split, limit):
    """Digit-only tokens rendered exactly like the product's documents.

    NOT the frozen split, and labelled as such in the report. The reason it
    exists: data/render_corpus/crops.npz labels are whole LINES ("DATE
    01102026"), so the frozen split contains ZERO digit-only single-token crops
    and cannot answer the question this lane asks. This probe renders the
    all-digit tokens that actually appear in capture.DOCS (DOB, dates, sort
    codes, account numbers, counts) through the SAME pipeline render_corpus
    uses — capture.render_document + capture.apply_profile + the tight ink box
    of render_corpus.page_crops — so the crop geometry, typeface, scale and the
    nine capture profiles are the product's, and the label is drawn rather than
    detected. The frozen checkpoint has never seen this domain at all, which is
    exactly the regime its confusion errors were measured in.
    """
    import zlib

    import build_dataset as B
    import capture as C
    import render_corpus as RC
    from PIL import Image

    toks = []
    for lines, _dtype in C.DOCS:
        for line in lines:
            for t in RC.train_label(line).split():
                if t.isdigit() and t not in toks:
                    toks.append(t)
    if not toks:
        return [], [], "constructed probe: no all-digit tokens in capture.DOCS"

    arrs, refs = [], []
    for i in range(limit):
        tok = toks[i % len(toks)]
        profile = RC.PROFILES[i % len(RC.PROFILES)]
        scale = (0.85, 1.0, 1.15)[i % 3]
        font = C.FONTS[zlib.crc32(("%s|%d" % (tok, i)).encode()) % len(C.FONTS)]
        page_seed = zlib.crc32(("%s|%d" % (tok, i)).encode()) & 0xFFFFFF
        # A blank line 0 so the token is drawn in the BODY face at line 1: line 0
        # uses the heading face (render_document), and a one-line page would type
        # every token bold.
        clean = C.render_document(["", tok], width=900, seed=page_seed,
                                 scale=scale, font=font)
        deg = C.apply_profile(clean, seed=page_seed ^ 0x5A5A,
                              **C.CAPTURE_PROFILES[profile])
        gray = np.asarray(deg.convert("L"), dtype=np.uint8)
        for crop, label in RC.page_crops(gray, ["", tok], scale):
            if label == tok:
                # the same 32px geometry the corpus and the app use
                arrs.append(B.crop_to_array(
                    Image.fromarray(np.asarray(crop, np.uint8))))
                refs.append(tok)
    return arrs, refs, ("constructed probe (NOT the frozen split): %d all-digit "
                        "token crops rendered by the render-corpus pipeline "
                        "(capture.DOCS' %d distinct all-digit values x %d profiles "
                        "x 3 scales)"
                        % (len(arrs), len(toks), len(RC.PROFILES)))


def _paired(refs, lps, allowed, beam):
    """Unconstrained vs constrained over the SAME logits -> the rows to report."""
    preds = {
        "unconstrained": [R.decode_greedy(lp.argmax(-1)) for lp in lps],
        "digits-only": [decode_constrained(lp, R.CHARSET, allowed, beam=beam)[0]
                        for lp in lps],
    }
    rows = {}
    for name, p in preds.items():
        bits = CD.exact_match_bits(p, refs)
        k = sum(bits)
        lo, hi = CD.wilson(k, len(bits))
        strict = [int(str(a).replace(" ", "") == str(b).replace(" ", ""))
                  for a, b in zip(p, refs)]
        rows[name] = {"n": len(bits), "exact": k,
                      "exact_rate": k / max(len(bits), 1), "ci": [lo, hi],
                      "cer": CD.cer_aggregate(p, refs), "bits": bits,
                      "strict": sum(strict), "preds": p}
    return rows


def _print_table(title, note, ckpt, rows, refs, secs_fwd, secs_dec, show=6):
    n = rows["unconstrained"]["n"]
    print("=" * 78)
    print("CONSTRAINED DECODE A/B — %s" % title)
    print("checkpoint %s (%s, %.1f MB)  [frozen; crnn.pt is mid-retrain]"
          % (ckpt, time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(Path(ckpt).stat().st_mtime)),
             Path(ckpt).stat().st_size / 1e6))
    print(note)
    print("ONE forward pass, %d crops, two decodes of the SAME logits "
          "(forward %.1fs, decode %.1fs)" % (n, secs_fwd, secs_dec))
    print("=" * 78)
    print("%-14s %6s %9s %-22s %8s %7s"
          % ("variant", "right", "exact", "Wilson 95% CI", "CER", "strict"))
    for name in ("unconstrained", "digits-only"):
        r = rows[name]
        print("%-14s %6d %8.2f%% [%6.2f%%, %6.2f%%]   %8.4f %7d"
              % (name, r["exact"], 100 * r["exact_rate"], 100 * r["ci"][0],
                 100 * r["ci"][1], r["cer"], r["strict"]))
    print("  (strict = raw string equality; `exact` follows realdata.norm_label, "
          "whose\n   O->0/I->1 table already forgives some of the confusions this "
          "mask removes)")

    g, c = rows["unconstrained"], rows["digits-only"]
    pc = CD.pair_counts(g["bits"], c["bits"])
    p = CD.mcnemar_exact(pc["variant_only"], pc["greedy_only"])
    print("-" * 78)
    print("PAIRED comparison (exact McNemar), unconstrained vs digits-only, "
          "same %d crops:" % n)
    print("  both right            %6d" % pc["both_right"])
    print("  digits-only only      %6d   <- constraint fixed it" % pc["variant_only"])
    print("  unconstrained only    %6d   <- constraint broke it" % pc["greedy_only"])
    print("  both wrong            %6d" % pc["both_wrong"])
    delta = 100 * (c["exact_rate"] - g["exact_rate"])
    changed = sum(1 for a, b in zip(g["preds"], c["preds"])
                  if R.norm_label(a) != R.norm_label(b))
    print("  exact match delta     %+6.2f pts   (%d -> %d crops)"
          % (delta, g["exact"], c["exact"]))
    print("  crops whose decode CHANGED %6d   (a mask that changes nothing would "
          "make\n                                    every number below meaningless)"
          % changed)
    letters = [i for i, p in enumerate(g["preds"]) if any(c.isalpha() for c in p)]
    flips = sum(1 for i in letters if c["bits"][i] and not g["bits"][i])
    print("  unconstrained decodes with a LETTER %5d/%d  <- the headroom the mask "
          "acts on;\n     %d of them became an exact match"
          % (len(letters), n, flips))
    print("  McNemar exact two-sided p = %.4g   (discordant pairs n=%d)"
          % (p, pc["variant_only"] + pc["greedy_only"]))
    if pc["variant_only"] + pc["greedy_only"] == 0:
        print("  -> no discordant pairs at all: the two decodes agree on every "
              "crop; this sample cannot separate them.")
    elif p >= 0.05:
        print("  -> NOT significant at 0.05. On this sample the constraint is "
              "indistinguishable from the baseline; report the interval.")
    elif delta > 0:
        print("  -> digits-only is better on this sample (p < 0.05).")
    else:
        print("  -> digits-only is WORSE on this sample (p < 0.05).")

    print("\nfirst %d crops the constraint fixed (truth | unconstrained -> "
          "digits-only):" % show)
    shown = 0
    for ref, gp, cp in zip(refs, g["preds"], c["preds"]):
        if not _match(gp, ref) and _match(cp, ref):
            print("  %-24s | %-24s -> %s" % (ref, gp, cp))
            shown += 1
            if shown >= show:
                break
    if not shown:
        print("  (none)")
    print("\nfirst %d the constraint got wrong that the baseline got right:"
          % show)
    shown = 0
    for ref, gp, cp in zip(refs, g["preds"], c["preds"]):
        if _match(gp, ref) and not _match(cp, ref):
            print("  %-24s | %-24s -> %s" % (ref, gp, cp))
            shown += 1
            if shown >= show:
                break
    if not shown:
        print("  (none)")


def _match(pred, ref):
    return R.norm_label(pred) == R.norm_label(ref)


def _ab(a):
    import torch
    torch.set_num_threads(max(1, a.threads))

    import eval_decode_ab as AB     # the guarded loader + the batched forward

    ckpt = Path(a.ckpt or FROZEN_CKPT)
    if not ckpt.exists():
        print("no checkpoint at %s" % ckpt, file=sys.stderr)
        return 1
    allowed = allowed_for_kind(a.kind)
    emit = "".join(sorted(set(allowed) & set(R.CHARSET)))
    print("field kind %r -> allowed %r; the charset can emit %r%s"
          % (a.kind, allowed, emit,
             "  (space also allowed)" if a.space else ""))

    sources = []
    if a.source in ("auto", "render"):
        sources.append(("render-frozen-split", _render_digit_crops))
    if a.source in ("auto", "sroie"):
        sources.append(("sroie", _sroie_digit_crops))
    if a.source in ("auto", "probe"):
        sources.append(("render-token-probe", _render_token_crops))

    model = None
    ran = 0
    out = {"kind": a.kind, "allowed": allowed, "beam": a.beam,
           "ckpt": str(ckpt), "sources": {}}
    for name, loader in sources:
        arrs, refs, note = loader(a.split, a.limit)
        print("%s: %s" % (name, note), flush=True)
        if len(arrs) < a.min_n:
            if a.source == "auto":
                print("  -> too few matching crops here to measure anything; "
                      "moving on to the next holdout.", flush=True)
            continue
        if model is None:
            model, ck = AB.load_model(str(ckpt))
            print("loaded %s (guarded: a torn read raises)" % ck)
        t0 = time.time()
        lps = AB.log_probs_for(model, arrs)
        fwd = time.time() - t0
        t0 = time.time()
        rows = _paired(refs, lps, allowed, a.beam)
        dec = time.time() - t0
        _print_table("%s, label = single all-digit token, kind=%s"
                     % (name, a.kind), note, ckpt, rows, refs, fwd, dec)
        g, c = rows["unconstrained"], rows["digits-only"]
        pc = CD.pair_counts(g["bits"], c["bits"])
        out["sources"][name] = {
            "note": note, "n": g["n"],
            "rows": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("bits", "preds")} for k, v in rows.items()},
            "paired": {**pc, "changed": sum(
                1 for a, b in zip(g["preds"], c["preds"])
                if R.norm_label(a) != R.norm_label(b)),
                "p_exact_mcnemar": CD.mcnemar_exact(pc["variant_only"],
                                                    pc["greedy_only"]),
                "delta_pts": 100 * (c["exact_rate"] - g["exact_rate"])},
        }
        ran += 1

    if not ran:
        print("no holdout on disk has digit-only single-token crops; nothing to "
              "measure.", file=sys.stderr)
        return 1
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
        print("\nwrote " + a.out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true",
                    help="the paired measurement (unconstrained vs digits-only)")
    ap.add_argument("--kind", default="sort_code",
                    help="field kind whose allowed subset is the constraint")
    ap.add_argument("--beam", type=int, default=1,
                    help="1 = greedy (the shipped rule); N = restricted prefix beam")
    ap.add_argument("--space", action="store_true", help="also allow the space class")
    ap.add_argument("--source", default="auto",
                    choices=("auto", "render", "sroie", "probe"),
                    help="holdout to measure on: render = the frozen render split; "
                         "sroie = the frozen SROIE word-crop split; probe = all-digit "
                         "tokens rendered through the render pipeline (NOT frozen); "
                         "auto = all three, skipping any with too few matching crops")
    ap.add_argument("--split", default="test", help="holdout split (never train)")
    ap.add_argument("--ckpt", default="",
                    help="default: the FROZEN models/ocr/crnn_before_render.pt")
    ap.add_argument("--limit", type=int, default=400,
                    help="max matching crops per holdout (keep the run short)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="below this many matching crops, do not report a row")
    ap.add_argument("--threads", type=int, default=2,
                    help="keep low: a training job owns the other cores")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.split == "train":
        print("refusing to score on the TRAIN split: that number is memorisation",
              file=sys.stderr)
        return 2
    if not a.ab:
        ap.print_help()
        return 2
    return _ab(a)


if __name__ == "__main__":
    sys.exit(main())
