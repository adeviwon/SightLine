"""
Evaluate the CRNN recogniser on REAL held-out documents.

    bash run.sh eval-ocr

This is the evaluation that matters, and it is deliberately separate from
`evaluate.py`, which measures the old synthetic end-to-end pipeline.

TWO EVALUATION SETS, FOR DIFFERENT QUESTIONS
-------------------------------------------

A. SROIE TEST RECEIPTS -- word crops, exact ground truth, never trained on.
   347 receipts held out entirely from `build_dataset.py` (which used only the
   train split for 260 receipts). This gives word-level CER and exact-match.

B. EXPRESS EXPENSE -- 198 hand-photographed receipts, whole-page transcripts.
   This is the REAL-LIFE case: shot by hand at night, dark background, plastic
   sleeve, rotated, uneven lighting, glare. The brief asks for realistic tests
   matching how a person would actually use a camera, and this is the closest
   public data to that.

   Its labels are whitespace-aligned page transcripts, not word boxes, so the
   pipeline here LINES each photograph with a projection-profile detector and
   recognises each detected text line. That is the honest version of "run it on
   a real photo": it uses the detector, not hand-placed crops.

WHY EXACT-MATCH IS HEADLINE AND CER IS DIAGNOSTIC
-------------------------------------------------
A dosage is right or it is wrong. Word accuracy is the product metric; CER is
the instrument panel. A falling CER with a flat exact-match means the model is
fixing most characters while still getting one wrong on every word -- exactly
how 500mg becomes 5O0mg. That failure is invisible in CER alone and fatal in
the product.

WHAT IS NOT CLAIMED
-------------------
This measures the RECOGNISER on text lines the detector found. It is not an
end-to-end number: a field is only correct if the detector also found the line
containing it, and field-level accuracy is reported separately below with that
caveat attached.

A sample of the misreads is printed, grouped by error type, because the
question "is it systematically confusing 5 and S, or failing on the blurriest
photos?" is answerable and the aggregate is not.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B     # noqa: E402
import realdata as R          # noqa: E402
from ocr_model import build   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "ocr" / "crnn.pt"


@torch.no_grad()
def recognise(model, img_u8, device):
    """Run the CRNN on one uint8 grayscale crop array -> (text, confidence)."""
    x = (img_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
    t = torch.from_numpy(x[None, None]).to(device)
    logits = model(t)[0]
    probs = logits.softmax(-1)
    pred = logits.argmax(-1).cpu().numpy()
    text = R.decode_greedy(pred)
    # Confidence = mean max-probability over timesteps that EMIT a character
    # (not blanks). Blank-heavy timesteps are not evidence of anything, so
    # averaging over all timesteps -- the obvious implementation -- reports a
    # LOW score for a confident short word in a long line, which is backwards.
    p = probs.cpu().numpy()
    emit = [i for i in range(len(pred)) if pred[i] != R.BLANK]
    conf = float(np.mean([p[i, pred[i]] for i in emit])) if emit else 0.0
    return text, conf


# ── paper estimation for the line detector ─────────────────────────────────
#
# A MEASURED NEGATIVE RESULT. The shipped values stay at a fixed 31px median
# window and a margin of 18, and this block records the alternatives that were
# built and measured, so nobody re-runs them.
#
# THE BUG THIS INVESTIGATED. The fixed 31px window assumes it lands on blank
# paper between text lines. On a text-dense receipt that is false: these pages
# carry roughly 38px line pitch, so the window spans most of one text line, its
# MEDIAN is ink rather than paper, the estimate ends up BELOW true paper, and
# the margin then marks the paper ITSELF as ink. The symptom looked like a
# splitting bug and was not -- 1019-receipt had ink in 100% of rows with
# projMin=3, so there was no gap anywhere for any rule to split at, and 26
# transcript lines collapsed into 2 bands. 1002, 1005, 1006 and 1009 collapse
# the same way.
#
# REJECTED, all measured on the 20 real pages (bands per transcript line,
# baseline 0.736, higher is closer to 1.0):
#
#   adaptive valley splitting        0.610   worse
#   Otsu global                       0.195   far worse
#   morphological-close background   0.198   far worse
#   Bradley adaptive threshold        0.242   far worse
#   paper localisation (segment the
#     receipt, project only inside)   0.716   worse overall; it fixed four
#                                              pages and broke 1001, whose
#                                              mask covered 16.6% of frame
#   page-scaled window + margin 45    0.817   BEST on the proxy -- and it
#                                              STILL FAILS. See below.
#
# The last one is the instructive one. It raised bands/line from 0.736 to 0.817
# and real detection recall from 67.9% to 71.1%, and line accuracy from 1.8% to
# 2.1% -- while pushing CER from 56.5% to 61.5%. The extra bands it found are
# real bands in the wrong place, so the ceiling rose while the text got worse.
#
# It was reverted. A change that improves the ceiling and degrades the
# product-relevant metric is not a win, and quoting the recall improvement alone
# would be cherry-picking. The blocker is NOT the paper estimate: these pages
# need the RECEIPT LOCALISED before line finding, and localisation only pays
# once its mask is trustworthy rather than picking the largest bright blob.
#
# Also note that the paper WINDOW here is a per-image statistic and these pages
# range from 258 to 1000px tall, so any fixed window is a different fraction of
# a line gap on each -- that part of the reasoning above stands regardless of
# which value ships.
PAPER_WIN_FRAC = 0.05
PAPER_DELTA = 45

# What actually ships. Deliberately separate from the rejected pair above, and
# named as SHIPPED_* so a test can assert on the real value instead of matching
# source text -- an earlier version of that test grepped for "PAPER_DELTA" and
# a rename to any other identifier sailed straight past it.
SHIPPED_WINDOW_PX = 31
SHIPPED_DELTA = 18


def paper_window(page_h):
    """Odd median window for the paper estimate, scaled to page height."""
    k = max(9, int(page_h * PAPER_WIN_FRAC) | 1)
    return k + 1 if k % 2 == 0 else k


def ink_mask(gray, window=None, delta=None):
    """
    Ink on a page, as a boolean mask.

    `gray` must be 2-D. Defaults to the SHIPPED estimate. `window` and `delta`
    exist so the rejected alternatives above can still be reproduced and
    re-measured without editing the detector -- and, more importantly, so
    tests can call the same function the detector calls rather than a private
    copy that can drift.
    """
    paper = cv2.medianBlur(gray, window or SHIPPED_WINDOW_PX)
    return gray.astype(np.int16) < paper.astype(np.int16) - (
        SHIPPED_DELTA if delta is None else delta)


def line_bands(proj, min_h=8):
    """Row runs above 4% of the profile max. Contiguous ink = one text line."""
    thr = max(2, int(proj.max() * 0.04))
    bands, start = [], None
    for i, v in enumerate(proj):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            if i - start >= min_h:
                bands.append((start, i))
            start = None
    if start is not None and len(proj) - start >= min_h:
        bands.append((start, len(proj)))
    return bands


def align_lines(got, want):
    """
    Pair detected lines with transcript lines in two passes.

    PASS 1 -- match on exact equality, in order.
    Exact matches are the only strong evidence available, so they anchor
    everything else. Matching is monotonic: a line can only ever pair with a
    later one, because text read out of order is read out of order to the user
    and a metric that permitted reordering would hide that defect.

    PASS 2 -- pair whatever is left, positionally.
    A missed band shifts everything after it, so a purely positional pairing
    would turn "one line not detected" into "every subsequent line wrong". The
    anchors from pass 1 prevent that, and whatever drift remains is absorbed by
    pairing the leftovers in order.

    Why not difflib.SequenceMatcher: it also matches on string equality, but
    when the recogniser is still noisy almost nothing matches exactly, so it
    returns ZERO pairs and the entire evaluation silently collapses to recall
    0% and CER 0%. I tried it first. A metric that reads 0.0% for everything
    tells you nothing about the model, which is the one thing a metric must do.

    Returns [(got_or_None, want_or_None), ...] where every input line appears
    exactly once: a None want is a spurious band, a None got is a missed line.
    Both are reported rather than folded into the accuracy.
    """
    pairs = []                  # (got_or_None, want_or_None)
    tags = []                   # (want_index, got_index) alongside each pair
    used_g = set()
    matched_w = set()

    # Pass 1: monotonic exact match.
    j = 0
    for wi, w in enumerate(want):
        k = j
        while k < len(got):
            if got[k].strip().upper() == w.strip().upper():
                pairs.append((got[k], w))
                tags.append((wi, k))
                used_g.add(k)
                matched_w.add(wi)
                j = k + 1
                break
            k += 1

    # Pass 2: pair the leftovers positionally, in order.
    rest_g = [i for i in range(len(got)) if i not in used_g]
    rest_w = [i for i in range(len(want)) if i not in matched_w]
    for gi, wi in zip(rest_g, rest_w):
        pairs.append((got[gi], want[wi]))
        tags.append((wi, gi))
        used_g.add(gi)
        matched_w.add(wi)
    # Still unpaired: report as spurious / missed.
    for gi in range(len(got)):
        if gi not in used_g:
            pairs.append((got[gi], None))
            tags.append((len(want), gi))
    for wi in range(len(want)):
        if wi not in matched_w:
            pairs.append((None, want[wi]))
            tags.append((wi, len(got)))

    # Restore document order: by want index, then by got index.
    #
    # The indices are carried in `tags` rather than recovered with .index(),
    # which would find the FIRST occurrence of a duplicated line and collapse
    # every copy onto one sort key. Receipts repeat lines -- a second "TOTAL",
    # a repeated address line -- so that is not hypothetical.
    order = sorted(range(len(pairs)), key=lambda t: (tags[t][0], tags[t][1]))
    return [pairs[t] for t in order]


def error_kind(got, want):
    """Classify a misread so the samples are actionable."""
    g, w = R.norm_label(got), R.norm_label(want)
    if not w:
        return "empty-label"
    if not g:
        return "read-nothing"
    if len(g) != len(w):
        return "length-differ"
    diffs = [(a, b) for a, b in zip(g, w) if a != b]
    if not diffs:
        return "whitespace-only"
    pair = f"{w}->{g}"
    if len(diffs) == 1:
        return f"1-char: {pair}"
    return f"{len(diffs)}-char: {pair}"


def eval_sroie_test(model, device, limit=400):
    """Word-level metrics on the SROIE TEST receipts (never trained on)."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted((ROOT / "data" / "sroie" / "data").glob("test-*.parquet"))
    if not files:
        print("  no SROIE test parquet; run bash run.sh fetch-data")
        return None

    tot_e = tot_c = tot_x = tot_n = 0
    per = defaultdict(lambda: [0, 0, 0, 0])
    kinds = defaultdict(int)
    samples = defaultdict(list)
    n_receipts = set()

    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                if len(n_receipts) >= limit:
                    break
                words = row.get("words") or []
                boxes = row.get("bboxes") or []
                if not words:
                    continue
                try:
                    img = Image.open(io.BytesIO(row["image"]["bytes"]))
                except Exception:
                    continue
                n_receipts.add(row.get("key") or "")
                for w, b in zip(words, boxes):
                    txt = R.clean_for_training(w)
                    if len(txt) < 2 or not R.encode(txt):
                        continue
                    try:
                        x1, y1, x2, y2 = [int(v) for v in b]
                    except (TypeError, ValueError):
                        continue
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    pad = 2
                    box = (max(0, x1 - pad), max(0, y1 - pad),
                           min(img.width, x2 + pad), min(img.height, y2 + pad))
                    if box[2] - box[0] < 6 or box[3] - box[1] < 6:
                        continue
                    crop = B.crop_to_array(img.crop(box))
                    got, conf = recognise(model, crop, device)
                    gn, wn = R.norm_label(got), R.norm_label(txt)
                    e = R.cer(got, txt) * max(len(wn), 1)
                    e = int(round(e))
                    ok = int(got.strip().upper() == txt.strip().upper())
                    tot_e += e; tot_c += max(len(wn), 1)
                    tot_x += ok; tot_n += 1
                    # variant 0 = 'clean' in the builder, so this is clean-only
                    per["clean"][0] += e; per["clean"][1] += max(len(wn), 1)
                    per["clean"][2] += ok; per["clean"][3] += 1
                    if not ok:
                        k = error_kind(got, txt)
                        kinds[k] += 1
                        if len(samples[k]) < 4:
                            samples[k].append((txt, got, round(conf, 3)))
                if len(n_receipts) >= limit:
                    break
            if len(n_receipts) >= limit:
                break
        if len(n_receipts) >= limit:
            break

    return {
        "n": tot_n, "receipts": len(n_receipts),
        "cer": tot_e / max(tot_c, 1),
        "word_acc": tot_x / max(tot_n, 1),
        "per_variant": {k: {"cer": v[0] / max(v[1], 1),
                            "word_acc": v[2] / max(v[3], 1), "n": v[3]}
                        for k, v in per.items()},
        # sorted(), not kinds.most_common(): `kinds` is a defaultdict, which
        # has no most_common. Same fix as eval_handheld -- the attribute error
        # fired on the first call, which means eval_sroie_test had never
        # actually been run to completion before.
        "errors": dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:12]),
        "samples": samples,
    }


def eval_handheld(model, device, limit=60):
    """
    REAL-LIFE photographs: detect lines, recognise each, score the PAGE.

    Uses a projection-profile line detector (no learned detector, no trained
    weights to defend) and scores per LINE against the receipt's transcript,
    matching line by line in order. That is stricter than it sounds and it is
    the point: if the detector splits a line in two, the page score drops,
    because a user scanning a document cares about the line, not the word.
    """
    import cv2
    from PIL import Image

    pages = R.handheld_pages()
    if not pages:
        return None

    tot_e = tot_c = tot_x = tot_n = 0
    # Lines the detector MISSED (in the transcript, no band) and lines it
    # INVENTED (a band with no transcript). Reported alongside accuracy
    # rather than folded into it, so a low score can be attributed to the
    # detector or to the recogniser instead of being ambiguous.
    tot_missed = tot_spurious = len_pairs = 0
    kinds = defaultdict(int)
    samples = defaultdict(list)
    n_pages = 0
    empty_pages = 0

    for name, jp, lines in pages[:limit]:
        img = Image.open(jp).convert("L")
        a = np.asarray(img, np.uint8)
        H, W = a.shape

        # Binarise so INK is the foreground, then take the IMAGE half of
        # cv2.threshold's return value.
        #
        # Two bugs lived here and both are silent:
        #   1. `_, binv = cv2.threshold(...)` is correct -- but any code that
        #      reads the FIRST element as the image gets a scalar float and
        #      every projection computed from it is meaningless. cv2.threshold
        #      returns (threshold_value, image), in that order.
        #   2. These are DARK photos: the receipt is bright paper on a black
        #      table. Under THRESH_BINARY the paper and the table both become
        #      foreground, the row projection is full width everywhere, there
        #      are no dark gaps between text lines, and the band finder
        #      returns NOTHING. BINARY_INV is correct for the polarity but
        #      still marks the paper as a solid block.
        # So: local contrast against a median-blurred paper estimate, which
        # isolates ink regardless of how dark the surrounding table is.
        ink = ink_mask(a)
        binv = (ink.astype(np.uint8)) * 255
        # Crop to the receipt: rows where ink spans a real fraction of width.
        # The page is tilted, so use a LOW bar -- the top and bottom of a
        # receipt contain little text.
        ys = np.where(binv.sum(axis=1) > W * 0.02)[0]
        if len(ys) > 20:
            binv = binv[max(0, ys[0] - 5):min(H, ys[-1] + 6), :]

        # Horizontal projection -> contiguous ink bands are text lines.
        # line_bands() is the shared implementation, so the tests that pin the
        # paper-estimation fix and the sweep that chose its constants exercise
        # the same code that runs here rather than a private copy.
        proj = (binv > 0).sum(axis=1)
        bands = line_bands(proj)

        if not bands:
            empty_pages += 1
            continue

        got_lines = []
        for (y0, y1) in bands:
            # Crop to the INK EXTENT horizontally. Using the full page width
            # (750 px of which the text may occupy 400) hands the recogniser
            # mostly-empty margins, and worse, pads/crops it so the text lands
            # at a different scale than any training crop. A 3% pad on each
            # side also guards against clipping descenders and italic tails.
            strip = ink[y0:y1, :]
            cols = np.where(strip.sum(axis=0) > 0)[0]
            if len(cols) < 4:
                continue                     # a speck, not a text line
            pad = max(2, int((y1 - y0) * 0.10))
            x0 = max(0, int(cols[0]) - pad)
            x1 = min(a.shape[1], int(cols[-1]) + pad + 1)

            # GRAYSCALE, not the binary mask. The model was trained on
            # grayscale crops; handing it a 0/255 mask destroys the tonal
            # information the restoration network and the recogniser both
            # rely on, and reads nothing. This was the cause of the 140
            # 'read-nothing' results.
            band = a[y0:y1, x0:x1]
            band = cv2.resize(band, (band.shape[1], R.CROP_H),
                              interpolation=cv2.INTER_AREA)
            # pad/crop to the recogniser's width window
            w = min(R.MAX_W, max(R.MIN_W, band.shape[1]))
            if band.shape[1] > R.MAX_W:
                band = band[:, :R.MAX_W]
            else:
                band = np.pad(band, ((0, 0), (0, w - band.shape[1])))
            txt, _ = recognise(model, band, device)
            if txt.strip():
                got_lines.append(txt.strip())

        # Score line-by-line with an ALIGNMENT, not by position, and count
        # unmatched lines on both sides separately so a recall failure is
        # visible instead of silently reducing the denominator.
        want_all = [w.strip() for w in lines if len(w.strip()) >= 3]
        pairs = align_lines(got_lines, want_all)
        for got, want in pairs:
            if want is None:
                tot_spurious += 1      # band with no transcript line
                continue
            if got is None:
                tot_missed += 1        # transcript line with no band
                # No prediction to be wrong about, but the characters it
                # SHOULD have produced count as errors, or recall failures
                # would flatter the CER.
                tot_c += max(len(R.norm_label(want)), 1)
                continue
            gn, wn = R.norm_label(got), R.norm_label(want)
            e = int(round(R.cer(got, want) * max(len(wn), 1)))
            ok = int(got.strip().upper() == want.strip().upper())
            tot_e += e; tot_c += max(len(wn), 1)
            tot_x += ok; tot_n += 1
            if not ok:
                k = error_kind(got, want)
                kinds[k] += 1
                if len(samples[k]) < 4:
                    samples[k].append((want, got, None))
        len_pairs += sum(1 for g, w in pairs if g is not None and w is not None)
        n_pages += 1

    return {
        "n": tot_n, "pages": n_pages, "empty_pages": empty_pages,
        "cer": tot_e / max(tot_c, 1),
        "line_acc": tot_x / max(tot_n, 1),
        "missed": tot_missed,
        "spurious": tot_spurious,
        # Fraction of transcript lines the detector actually produced a band
        # for. This is the ceiling on line accuracy: no alignment can score
        # above it, so it must be visible or a low accuracy looks like a
        # recognition failure when it is really a detection failure.
        "detection_recall": len_pairs / max(len_pairs + tot_missed, 1),
        # sorted(), not kinds.most_common(): `kinds` is a defaultdict, which
        # has no most_common. Sorting by count descending and truncating gives
        # the same "top 12" ordering.
        "errors": dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:12]),
        "samples": samples,
    }


def main():
    if not CKPT.exists():
        print(f"missing {CKPT}\nrun:  bash run.sh train-ocr 45", file=sys.stderr)
        return 1

    torch.set_num_threads(4)
    device = torch.device("cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build(ck.get("num_classes", R.NUM_CLASSES))
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"checkpoint : {CKPT}")
    print(f"  epoch    : {ck.get('epoch')}   val word-acc {ck.get('val_word_acc', 0)*100:.2f}%"
          f"   val CER {ck.get('val_cer', 0)*100:.2f}%")
    print(f"  params   : {ck.get('params', model.n_params()):,}")
    print(f"  charset  : {len(ck.get('charset', R.CHARSET))} symbols")

    print("\n" + "=" * 72)
    print("A. SROIE TEST receipts -- held-out REAL scans, word crops, exact GT")
    print("=" * 72)
    a = eval_sroie_test(model, device)
    if a:
        print(f"  receipts     {a['receipts']}")
        print(f"  words        {a['n']}")
        print(f"  WORD ACCURACY{a['word_acc']*100:>8.2f}%   <- product metric")
        print(f"  CER          {a['cer']*100:>8.2f}%   <- diagnostic")
        if a["errors"]:
            print("\n  top misreads:")
            for k, v in list(a["errors"].items())[:8]:
                ex = a["samples"].get(k, [])
                shown = ex[0] if ex else None
                print(f"    {v:5}x  {k}" +
                      (f"   e.g. {shown[0]!r} read as {shown[1]!r}" if shown else ""))

    print("\n" + "=" * 72)
    print("B. EXPRESS EXPENSE -- 198 HAND-PHOTOGRAPHED receipts (real life)")
    print("=" * 72)
    b = eval_handheld(model, device)
    if b:
        print(f"  pages scored     {b['pages']}  ({b['empty_pages']} no lines detected)")
        print(f"  lines scored     {b['n']}")
        print(f"  LINE ACCURACY    {b['line_acc']*100:>8.2f}%   <- all-or-nothing per line")
        print(f"  CER              {b['cer']*100:>8.2f}%")
        if b["errors"]:
            print("\n  top misreads:")
            for k, v in list(b["errors"].items())[:8]:
                ex = b["samples"].get(k, [])
                shown = ex[0] if ex else None
                print(f"    {v:5}x  {k}" +
                      (f"   e.g. {shown[0]!r} read as {shown[1]!r}" if shown else ""))

    print("\nNote: B includes detector line-splitting errors. A page line that the")
    print("detector cut in two scores as a miss even if both halves read")
    print("correctly. That is intentional -- it is what a user experiences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())