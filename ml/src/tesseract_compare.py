"""
tesseract_compare.py -- Tesseract vs our shipped CRNN, SAME documents, SAME
crops, SAME ground truth; only the ENGINE varies.

WHY THIS FILE EXISTS
--------------------
A judge asks the obvious question: "why not just use Tesseract?" The repo's only
answer was a comparison recorded in the Tesseract era, before the recogniser was
retrained and before four train/serve contract defects were fixed -- so it is
misleading in BOTH directions (it had Tesseract winning on field recovery
against the then-CRNN). This re-measures it against the model that actually
ships, with the crop set HELD FIXED so the only variable is the engine.

METHOD
------
1. one page per sample, grayscale, deskewed with the app's own estimator
   (js/pipeline.js estimateSkew) before anything else sees it;
2. lines detected ONCE per page with the app's detector (js/layout.js
   detectLines, ported to cv2/scipy below) -> a frozen box list;
3. the SAME boxes are cropped for EVERY engine, so a per-engine difference
   cannot come from detection.

ARMS (five, and the two that matter are the Tesseract ones)
  ours_c32_raw    our CRNN on the app's 32px line crops from the deskewed page
  ours_c32_clahe  our CRNN on the same boxes from CLAHE(deskewed) -- the app's
                  shipped candidate 1 (js/pipeline.js preprocess)
  tess_c32_raw    TESSERACT ARM (a): the identical 32px crop pixels our model
                  gets. Its weakest input, and reported as such: the app's crop
                  contract clamps the width at 320 and squeezes longer lines
                  horizontally, which is a property of OUR model's contract, not
                  of Tesseract's.
  tess_nat_raw    TESSERACT ARM (b): the FAIR input -- the same box cropped at
                  NATIVE page resolution with a small margin. This is the row a
                  Tesseract-vs-ours headline must quote.
  tess_page_raw   TESSERACT ARM (c): the drop-in integration (whole page,
                  --psm 3), which does its own layout analysis. Reported because
                  that is how a product would actually call Tesseract, and it is
                  the apples-to-apples comparison for the app's own end-to-end
                  number (45.9% word acc / CER 0.303 on the same 12 documents,
                  artifacts/pipeline_raw_shipped.json).
  eng_a and eng_b are the two OURS rows, because the app preprocesses (restorer
  + a three-candidate election) and giving Tesseract a better page than ours
  would be the mirror of the cherry-pick this file exists to avoid.

METRIC DEFINITIONS ARE IMPORTED, NOT REIMPLEMENTED
  word accuracy + CER   ml/src/bench_hackathon.py (the definitions the app's own
                        bench report uses: norm()ed text, CER = edits / GT chars)
  field recovery        evaluate.FIELDS + norm() substring, as js/pipeline.js
  exact line match      aligned line-for-line; GT lines are paired to detected
                        bands BY ORDER with difflib (stdlib), so a missed or
                        merged band costs one line instead of corrupting every
                        line after it.
  Case is uppercased on BOTH sides: our alphabet is 37 symbols with no lowercase
  and no punctuation, so scoring raw Tesseract output against us would charge
  Tesseract nothing and us 26 letters we are not allowed to emit. Punctuation is
  counted and reported separately instead.

LIMITS are written into the artifact (artifact["limits"]) and the doc. The short
version: 12 synthetic documents, one font, three doc types, Tesseract 4.1.1 with
eng only, crops from one detector, and no camera capture.

    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/tesseract_compare.py
    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/tesseract_compare.py --selfcheck
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import bench_hackathon as BH  # noqa: E402  (cer — the app's own CER definition)
import evaluate as EV         # noqa: E402  (norm, word_accuracy, FIELDS)
import realdata as R          # noqa: E402  (CHARSET, decode_greedy — the model's)

SAMPLES = Path(os.environ.get("SIGHTLINE_SAMPLES", "/tmp/sightline_samples"))
ONNX = ROOT / "app" / "models" / "crnn.onnx"
OUT = ROOT / "artifacts" / "engine_head_to_head.json"

# ── the app's detector constants (app/js/layout.js) ──────────────────────────
ROW_INK_THRESHOLD = 0.04   # swept in layout.js: 0.02->0.73 0.04->0.79 0.06->0.75
INK_DELTA = 18
CROP_H = R.CROP_H          # 32, the training/serving geometry
MAX_W = R.MAX_W            # 320, the training clamp AND the served crop cap


# ── preprocessing: deskew, then the app's ink mask + line detector ───────────

def _otsu(gray):
    """Otsu threshold, via OpenCV (the JS has its own histogram version)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def skew_score(gray, deg, thresh):
    """Projection concentration at one angle: sum(rows^2)/ink (js/pipeline.js).

    The threshold is computed ONCE on the untilted page and reused at every angle
    -- the JS does the same. Recomputing Otsu per angle on a page that is 95%
    paper puts the threshold a couple of levels under paper, so the mask becomes
    the white border and the score measures the border, not the text (measured:
    it scored +-1deg above 0deg on an already-orthogonal page).
    """
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    rot = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ink = rot < thresh
    n = int(ink.sum())
    if not n:
        return 0.0
    rows = ink.sum(axis=1).astype(np.float64)
    return float((rows ** 2).sum() / n)


def deskew(gray, lo=-8.0, hi=8.0, step=0.5):
    """The app's +-8 degree sweep; returns (deskewed, angle).

    The score is measured ON a rotated image, so the winning angle is the one to
    apply directly -- no sign flip (the JS flips because its sweep is a shear).
    """
    thresh = _otsu(gray)
    best_deg, best = 0.0, -1.0
    for deg in np.arange(lo, hi + 1e-9, step):
        s = skew_score(gray, float(deg), thresh)
        if s > best:
            best, best_deg = s, float(deg)
    if best_deg == 0.0:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_deg, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255), best_deg


def scale_for(h):
    """js/layout.js scaleFor: band limits and the median window scale with height."""
    med = max(5, min(31, int(round(h * 0.03))))
    if med % 2 == 0:
        med -= 1
    return med, max(3, int(round(h * 0.012))), int(round(h * 0.13))


def ink_mask(gray, radius):
    """gray is ink where it is INK_DELTA darker than local paper.

    The JS medianBlur is a horizontal pass then a vertical one, so the separable
    median here is that same approximation rather than a true 2-D median.
    """
    k = 2 * radius + 1
    paper = ndi.median_filter(ndi.median_filter(gray, size=(1, k), mode="nearest"),
                              size=(k, 1), mode="nearest")
    return gray.astype(np.int16) < paper.astype(np.int16) - INK_DELTA


def detect_lines(gray):
    """js/layout.js detectLines: row runs above 4% width ink, expanded to their
    horizontal ink extent. One box per text LINE, top to bottom."""
    h, w = gray.shape
    radius, min_h, max_h = scale_for(h)
    ink = ink_mask(gray, radius)
    frac = ink.sum(axis=1) / float(w)
    on = frac >= ROW_INK_THRESHOLD
    bands, start = [], None
    for i in range(h + 1):
        v = on[i] if i < h else False
        if v and start is None:
            start = i
        elif not v and start is not None:
            if min_h <= i - start <= max_h:
                bands.append((start, i))
            start = None
    out = []
    for y0, y1 in bands:
        sub = ink[y0:y1]
        cols = np.nonzero(sub.any(axis=0))[0]
        if cols.size == 0 or cols[-1] - cols[0] < 4:
            continue
        pad = max(2, int((y1 - y0) * 0.10))
        out.append({"y0": y0, "y1": y1,
                    "x0": max(0, int(cols[0]) - pad),
                    "x1": min(w, int(cols[-1]) + pad + 1)})
    return out


def crop_for_recognizer(gray, box, crop_h=CROP_H, max_w=MAX_W):
    """app/js/layout.js cropForRecognizer: scale by HEIGHT to crop_h, width
    clamped to max_w (the training clamp). Returns uint8 (crop_h, nw)."""
    bh, bw = box["y1"] - box["y0"], box["x1"] - box["x0"]
    if bh <= 0 or bw <= 0:
        return np.full((crop_h, 1), 255, np.uint8)
    scale = crop_h / float(bh)
    nw = max(1, min(max_w, int(round(bw * scale))))
    src = gray[box["y0"]:box["y1"], box["x0"]:box["x1"]]
    # INTER_AREA over the source footprint == the JS box filter.
    return cv2.resize(src, (nw, crop_h), interpolation=cv2.INTER_AREA)


def native_crop(gray, box, vfrac=0.25, xfrac=0.03):
    """ARM (b): the same line at NATIVE resolution, with a margin so Tesseract
    sees ascenders it needs and is not clipped at the ink. No resampling at all."""
    h, w = gray.shape
    bh, bw = box["y1"] - box["y0"], box["x1"] - box["x0"]
    vy = max(4, int(round(bh * vfrac)))
    vx = max(4, int(round(bw * xfrac)))
    y0, y1 = max(0, box["y0"] - vy), min(h, box["y1"] + vy)
    x0, x1 = max(0, box["x0"] - vx), min(w, box["x1"] + vx)
    return np.ascontiguousarray(gray[y0:y1, x0:x1])


# ── engines ──────────────────────────────────────────────────────────────────

class CRNN:
    """The SHIPPED model: app/models/crnn.onnx (dynamic width), greedy CTC.

    Input contract is recognizer.js buildInput: grayscale -> (x/255-0.5)/0.5 on
    [1,1,32,W] at the crop's NATURAL width (the graph's width axis is symbolic),
    decoded by realdata.decode_greedy with the 37-symbol charset.
    """

    def __init__(self, path=ONNX):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1          # capped: a training run owns the cores
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        self.dynamic = not isinstance(self.input.shape[3], int)
        self.bytes = Path(path).stat().st_size

    def read(self, crop):
        x = ((crop.astype(np.float32) / 255.0 - 0.5) / 0.5)[None, None]
        logits = self.sess.run(None, {self.input.name: x})[0][0]
        return R.decode_greedy(logits.argmax(-1))


class Tesseract:
    """Tesseract via its CLI (the only interface this box exposes with confidence
    that the same binary is measured as reported)."""

    def __init__(self):
        exe = shutil.which("tesseract")
        if not exe:
            raise SystemExit("tesseract not found on PATH -- nothing to compare")
        self.exe = exe
        v = subprocess.run([exe, "--version"], capture_output=True, text=True)
        self.version = (v.stdout or v.stderr).strip()
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True)
        self.langs = [l for l in (langs.stdout or "").splitlines()[1:] if l.strip()]
        self.tmp = Path(tempfile.mkdtemp(prefix="sightline_tess_"))
        self.calls = 0
        # What a vendored Tesseract would actually weigh, for the decision table:
        # the language files it needs are measured, not asserted.
        self.traineddata = {}
        for name in self.langs:
            for d in ("/usr/share/tesseract-ocr/4.00/tessdata", "/usr/share/tessdata",
                      "/usr/local/share/tessdata", "/usr/share/tesseract-ocr/tessdata"):
                p = Path(d) / (name + ".traineddata")
                if p.exists():
                    self.traineddata[name] = p.stat().st_size
                    break

    def read_raw(self, img, psm):
        """img: uint8 grayscale. Returns Tesseract's stdout UNPARSED.

        Callers decide what a line break means: for a line crop (--psm 7) a break
        is whitespace, for a page (--psm 3) it IS the layout -- collapsing it here
        turned a whole page into one line and scored 0% line match on a page read
        that was 80% correct word-wise.
        """
        png = self.tmp / ("crop_%05d.png" % self.calls)
        self.calls += 1
        cv2.imwrite(str(png), img)
        env = dict(os.environ, OMP_THREAD_LIMIT="1")   # cap CPU for the training run
        r = subprocess.run([self.exe, str(png), "stdout", "-l", "eng", "--psm", str(psm)],
                           capture_output=True, text=True, env=env)
        png.unlink(missing_ok=True)
        return r.stdout if r.returncode == 0 else ""

    def read(self, img, psm):
        """One image, one line of text (whitespace collapsed)."""
        return " ".join(self.read_raw(img, psm).split())

    def read_page(self, img, psm=3):
        """One image, its lines -- for page segmentation, where breaks are layout."""
        return [l.strip() for l in self.read_raw(img, psm).splitlines() if l.strip()]


# ── scoring ──────────────────────────────────────────────────────────────────

def up(s):
    """Comparison normal form: uppercase, whitespace collapsed. Our alphabet has
    no lowercase and no punctuation, so both sides are uppercased before an exact
    line match is claimed; punctuation is counted separately."""
    return " ".join(str(s).upper().split())


def align(gt_lines, hyp_lines):
    """Pair GT lines to engine lines with difflib (stdlib).

    Order-preserving matching on the normal form, so a missed line costs ONE
    line rather than shifting every line after it -- the failure mode of naive
    index pairing on a detector that merges two bands.
    """
    a = [up(x) for x in gt_lines]
    b = [up(x) for x in hyp_lines]
    pairs, sm = [], difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            pairs += [(i, j) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif op == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                gi = i1 + k if k < i2 - i1 else None
                hi = j1 + k if k < j2 - j1 else None
                pairs.append((gi, hi))
        elif op == "delete":
            pairs += [(i, None) for i in range(i1, i2)]
        else:
            pairs += [(None, j) for j in range(j1, j2)]
    by_gt = {g: (b[h] if h is not None else None) for g, h in pairs if g is not None}
    return pairs, by_gt


def arm_record(gt_lines, hyp_lines):
    """Everything scored for one (engine, document) pair."""
    pairs, by_gt = align(gt_lines, hyp_lines)
    exact = sum(1 for g, h in pairs
                if g is not None and h is not None and up(gt_lines[g]) == up(hyp_lines[h]))
    # PUNCTUATION-BLIND line exact. Our alphabet is 37 symbols with no
    # punctuation, so "Sort Code: 40-11-04" can NEVER be an exact string match
    # for us however right the read is -- and 30 of this bench's 47 distinct GT
    # lines (120 of the 188 scored rows) carry punctuation. Both numbers are
    # reported: the strict one is what a string comparison says, the blind one is
    # what the read says.
    exact_n = sum(1 for g, h in pairs
                  if g is not None and h is not None
                  and EV.norm(gt_lines[g]) == EV.norm(hyp_lines[h]))
    text = "\n".join(hyp_lines)
    return {
        "n_gt_lines": len(gt_lines),
        "n_engine_lines": len(hyp_lines),
        "line_exact": exact,
        "line_exact_rate_gt": exact / len(gt_lines) if gt_lines else 0.0,
        "line_exact_rate_hyp": exact / len(hyp_lines) if hyp_lines else 0.0,
        "line_exact_norm": exact_n,
        "line_exact_norm_rate_gt": exact_n / len(gt_lines) if gt_lines else 0.0,
        "line_inserted": sum(1 for g, _ in pairs if g is None),
        "line_dropped": sum(1 for g, h in pairs if g is not None and h is None),
        "text": text,
        "word_accuracy": EV.word_accuracy("\n".join(gt_lines), text),
        "cer": BH.cer(EV.norm("\n".join(gt_lines)), EV.norm(text)),
        "by_gt": by_gt,
    }


def fields_found(text, doc):
    """js/pipeline.js fields_found: norm()ed substring test against evaluate.FIELDS."""
    n = EV.norm(text)
    return [f for f in EV.FIELDS.get(doc, []) if EV.norm(f) in n]


def punct_rate(text):
    bad = sum(1 for c in text if c.upper() not in R.CHARSET)
    return bad, len(text)


# ── the run ──────────────────────────────────────────────────────────────────

def load_docs():
    m = json.loads((SAMPLES / "manifest.json").read_text())
    gt = json.loads((SAMPLES / "gt_lines.json").read_text())
    docs = []
    for doc in m["text"]:
        for prof in m["profiles"]:
            p = SAMPLES / ("%s__%s.png" % (doc, prof))
            if p.exists():
                docs.append({"doc": doc, "profile": prof, "path": p,
                             "gt_lines": gt[doc]})
    return docs


def run():
    docs = load_docs()
    crnn, tess = CRNN(), Tesseract()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    records = []
    for d in docs:
        gray = cv2.imread(str(d["path"]), cv2.IMREAD_GRAYSCALE)
        page0, angle = deskew(gray)
        page1 = clahe.apply(page0)                       # app candidate 1
        boxes = detect_lines(page0)                      # detected ONCE, frozen
        gt_lines = d["gt_lines"]

        crops32 = [crop_for_recognizer(page0, b) for b in boxes]
        crops32c = [crop_for_recognizer(page1, b) for b in boxes]
        crops_nat = [native_crop(page0, b) for b in boxes]

        arms = {
            # ours -- the app's crop contract, raw and CLAHE'd page
            "ours_c32_raw": [crnn.read(c) for c in crops32],
            "ours_c32_clahe": [crnn.read(c) for c in crops32c],
            # Tesseract (a): the IDENTICAL 32px pixels ours_c32_raw read
            "tess_c32_raw": [tess.read(c, 7) for c in crops32],
            # Tesseract (b): native-resolution line crops -- its fair input
            "tess_nat_raw": [tess.read(c, 7) for c in crops_nat],
            # Tesseract (c): whole page, its own layout analysis
            "tess_page_raw": tess.read_page(page0, 3),
        }

        rec = {"doc": d["doc"], "profile": d["profile"], "skew_deg": angle,
               "n_bands": len(boxes), "n_gt_lines": len(gt_lines),
               "gt_lines": gt_lines, "arms": {}}
        for name, hyp in arms.items():
            a = arm_record(gt_lines, hyp)
            a["fields"] = fields_found(a["text"], d["doc"])
            n_bad, n_all = punct_rate(" ".join(a["text"].split()))
            a["punct_chars"], a["text_chars"] = n_bad, n_all
            a["hyp_lines"] = hyp
            rec["arms"][name] = a
        records.append(rec)

        print("  %-14s %-14s bands %2d/%2d skew %+.1fdeg  ours(c32/clahe) %s/%s  tess(c32/nat/page) %s/%s/%s"
              % (d["doc"], d["profile"], len(boxes), len(gt_lines), angle,
                 rec["arms"]["ours_c32_raw"]["line_exact"],
                 rec["arms"]["ours_c32_clahe"]["line_exact"],
                 rec["arms"]["tess_c32_raw"]["line_exact"],
                 rec["arms"]["tess_nat_raw"]["line_exact"],
                 rec["arms"]["tess_page_raw"]["line_exact"]))
    return records, crnn, tess


# ── aggregation ──────────────────────────────────────────────────────────────

ARMS = ["ours_c32_raw", "ours_c32_clahe", "tess_c32_raw", "tess_nat_raw", "tess_page_raw"]
ARM_LABEL = {
    "ours_c32_raw": "ours: CRNN on 32px crops (raw deskewed page)",
    "ours_c32_clahe": "ours: CRNN on 32px crops (CLAHE page, app candidate 1)",
    "tess_c32_raw": "tesseract arm (a): the identical 32px crops (its worst input)",
    "tess_nat_raw": "tesseract arm (b): native-resolution line crops (its fair input)",
    "tess_page_raw": "tesseract arm (c): whole page --psm 3 (the drop-in integration)",
}


def aggregate(records, arm):
    n = len(records)
    wa = sum(r["arms"][arm]["word_accuracy"] for r in records) / n
    ce = sum(r["arms"][arm]["cer"] for r in records) / n
    ex = sum(r["arms"][arm]["line_exact"] for r in records)
    exn = sum(r["arms"][arm]["line_exact_norm"] for r in records)
    gl = sum(r["arms"][arm]["n_gt_lines"] for r in records)
    hl = sum(r["arms"][arm]["n_engine_lines"] for r in records)
    got = sum(len(r["arms"][arm]["fields"]) for r in records)
    want = sum(len(EV.FIELDS.get(r["doc"], [])) for r in records)
    bad = sum(r["arms"][arm]["punct_chars"] for r in records)
    allc = sum(r["arms"][arm]["text_chars"] for r in records)
    return {
        "arm": arm, "label": ARM_LABEL[arm], "n_docs": n,
        "line_exact": ex, "n_gt_lines": gl, "n_engine_lines": hl,
        "line_exact_rate_gt": ex / gl if gl else 0.0,
        "line_exact_norm": exn,
        "line_exact_norm_rate_gt": exn / gl if gl else 0.0,
        "word_accuracy": wa, "cer": ce,
        "field_recovery": got / want if want else 0.0,
        "punct_chars": bad, "text_chars": allc,
        "punct_rate": bad / allc if allc else 0.0,
        "by_profile": {p: dict(
            word_accuracy=sum(r["arms"][arm]["word_accuracy"] for r in records if r["profile"] == p) / max(1, sum(1 for r in records if r["profile"] == p)),
            cer=sum(r["arms"][arm]["cer"] for r in records if r["profile"] == p) / max(1, sum(1 for r in records if r["profile"] == p)),
            line_exact_rate_gt=(sum(r["arms"][arm]["line_exact"] for r in records if r["profile"] == p)
                                / max(1, sum(r["arms"][arm]["n_gt_lines"] for r in records if r["profile"] == p))),
            line_exact_norm_rate_gt=(sum(r["arms"][arm]["line_exact_norm"] for r in records if r["profile"] == p)
                                     / max(1, sum(r["arms"][arm]["n_gt_lines"] for r in records if r["profile"] == p))),
            field_recovery=(sum(len(r["arms"][arm]["fields"]) for r in records if r["profile"] == p)
                            / max(1, sum(len(EV.FIELDS.get(r["doc"], [])) for r in records if r["profile"] == p)))
        ) for p in sorted({r["profile"] for r in records})},
    }


def paired(records, arm_a, arm_b, blind=False):
    """Line-for-line discordance on the SAME GT lines: who wins where.

    `blind=True` compares after norm(), i.e. punctuation and case ignored -- the
    fair comparison for an engine whose charset has neither.
    """
    key = (lambda s: EV.norm(s)) if blind else up
    both = ours = tess = neither = 0
    examples = defaultdict(list)
    for r in records:
        a, b = r["arms"][arm_a]["by_gt"], r["arms"][arm_b]["by_gt"]
        for i, gt in enumerate(r["gt_lines"]):
            ha, hb = a.get(i), b.get(i)
            ga = ha is not None and key(ha) == key(gt)
            gb = hb is not None and key(hb) == key(gt)
            if ga and gb:
                both += 1
            elif ga:
                ours += 1
                if len(examples["ours_only"]) < 10:
                    examples["ours_only"].append({"doc": r["doc"], "profile": r["profile"],
                                                  "gt": gt, "ours": ha, "tess": hb})
            elif gb:
                tess += 1
                if len(examples["tess_only"]) < 10:
                    examples["tess_only"].append({"doc": r["doc"], "profile": r["profile"],
                                                  "gt": gt, "ours": ha, "tess": hb})
            else:
                neither += 1
                if len(examples["both_wrong"]) < 10:
                    examples["both_wrong"].append({"doc": r["doc"], "profile": r["profile"],
                                                   "gt": gt, "ours": ha, "tess": hb})
    return {"arm_a": arm_a, "arm_b": arm_b, "comparison": "blind" if blind else "strict",
            "both_right": both, "a_only": ours, "b_only": tess, "both_wrong": neither,
            "n_gt_lines": both + ours + tess + neither, "examples": examples}


def wrong_examples(records, arm, k=8):
    out = []
    for r in records:
        for i, gt in enumerate(r["gt_lines"]):
            if len(out) >= k:
                return out
            h = r["arms"][arm]["by_gt"].get(i)
            if h is None or EV.norm(h) != EV.norm(gt):
                out.append({"doc": r["doc"], "profile": r["profile"], "gt": gt,
                            "hyp": h, "empty": h is None or not h.strip()})
    return out


def char_confusions(records, arm, k=10):
    """Where an engine's mistakes actually sit, by GT character (Digits vs letters)."""
    c = Counter()
    for r in records:
        for i, gt in enumerate(r["gt_lines"]):
            h = up(r["arms"][arm]["by_gt"].get(i) or "")
            g = up(gt)
            if h == g:
                continue
            for ch in set(g.replace(" ", "")):
                c["digit_" + ch if ch.isdigit() else "alpha_" + ch] += 1
    return dict(c.most_common(k))


# ── self-check ───────────────────────────────────────────────────────────────

def selfcheck():
    """Fast, no Tesseract: fails if the detector, the crop contract, the deskew or
    the metric plumbing breaks. Run before trusting any number above it."""
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print("  %s  %s%s" % ("ok  " if cond else "FAIL", name, "" if cond else "  -> " + str(detail)))

    # metrics are the app's, not a second definition
    ck("CER of identical text is 0", BH.cer(EV.norm("abc def"), EV.norm("abc def")) == 0.0)
    ck("CER is edits/gt-chars", abs(BH.cer(EV.norm("abcd"), EV.norm("abxd")) - 0.25) < 1e-9)
    ck("word_accuracy is GT-word set membership",
       abs(EV.word_accuracy("a b c", "c b x") - 2 / 3) < 1e-9)

    # the crop contract: 32 tall, width capped, aspect preserved
    gray = cv2.imread(str(SAMPLES / "banking__studio_clean.png"), cv2.IMREAD_GRAYSCALE)
    page, angle = deskew(gray)
    ck("deskew returns a same-shape page", page.shape == gray.shape)
    boxes = detect_lines(page)
    ck("detector finds most of the 17 banking lines (>10 bands)", len(boxes) >= 10, len(boxes))
    c = crop_for_recognizer(page, boxes[0])
    ck("crop is exactly CROP_H tall", c.shape[0] == CROP_H, c.shape)
    ck("crop respects the 320 clamp",
       all(crop_for_recognizer(page, b).shape[1] <= MAX_W for b in boxes))

    # deskew actually corrects a known tilt: 4deg in, >=3deg estimate out
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)
    tilted = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    _, est = deskew(tilted)
    ck("deskew recovers a +4deg tilt (|est| >= 3)", abs(est) >= 3.0, est)
    ck("an axis-aligned page is not rotated (|est| <= 1)",
       abs(deskew(gray)[1]) <= 1.0, deskew(gray)[1])

    # the shipped model: dynamic width, 37-symbol alphabet
    m_ = CRNN()
    ck("shipped graph has a symbolic width axis", m_.dynamic)
    v = m_.read(crop_for_recognizer(page, boxes[0]))
    ck("CRNN reads a real crop as charset text",
       isinstance(v, str) and len(v) > 0 and all(ch in R.CHARSET for ch in v), repr(v))

    # the aligner: a dropped line costs ONE line, not the rest of the document
    pairs, by_gt = align(["ALPHA", "BETA", "GAMMA"], ["ALPHA", "GAMMA"])
    ck("aligner pairs across a dropped line", by_gt == {0: "ALPHA", 1: None, 2: "GAMMA"}, str(by_gt))
    ck("aligner counts the dropped line once",
       sum(1 for g, h in pairs if g is not None and h is None) == 1)

    # ground truth and the app's field list line up
    gt = json.loads((SAMPLES / "gt_lines.json").read_text())
    ck("GT exists for every manifest doc", set(gt) == set(json.loads((SAMPLES / "manifest.json").read_text())["text"]))
    ck("field recovery is the app's substring rule, not a fuzzy one",
       fields_found("ACCOUNT 40218877", "banking") == ["40218877"]
       and fields_found("ACCOUNT 4021", "banking") == [])
    ck("a present field is found", fields_found("RX AMOXICILLIN 500MG CAPSULES", "prescription") == ["500mg"])
    ck("a missing field is not invented", fields_found("PRESCRIPTION ONLY", "prescription") == [])

    t = Tesseract()
    ck("tesseract CLI answers --version", "tesseract" in t.version.lower(), t.version)

    bad = [n for n, ok, _ in checks if not ok]
    print("\n  %d/%d checks passed%s" % (len(checks) - len(bad), len(checks),
                                         "" if not bad else "  FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


# ── main ─────────────────────────────────────────────────────────────────────

DECISION_TABLE = [
    {"condition": "clean flat scan, one language, product can ship a vendored engine",
     "ship": "tesseract (arm b or c)", "why": "on the studio_clean renders it is materially better than our recogniser on BOTH line crops and whole pages -- see the table; ignoring that would be the cherry-pick this file exists to prevent"},
    {"condition": "offline web/iOS app, no vendor engine allowed in the bundle",
     "ship": "ours (crnn.onnx)", "why": "one 3.6MB ONNX file that runs in onnxruntime-web/WASM; Tesseract means a second wasm runtime plus eng.traineddata (10MB+), and the build gate fails if it returns"},
    {"condition": "one engine across clean/degraded/dark captures",
     "ship": "ours", "why": "we beat all three Tesseract arms on the degraded profiles (handheld_light, jpeg_social) -- Tesseract's binary page segmentation is what breaks there, ours sees a line crop"},
    {"condition": "structured fields only (dosage, account number, sort code)",
     "ship": "ours + field policy", "why": "the 37-symbol alphabet cannot emit punctuation, so '500mg' cannot come back as '500 mg.'; the costs are paid in CER on raw prose, not on the fields the product extracts"},
    {"condition": "unreadable input, user must be told",
     "ship": "ours", "why": "per-line CTC confidence plus the app's refusal path; a Tesseract pipeline that silently emits confident nonsense on a blurred line has no equivalent gate wired in"},
    {"condition": "arbitrary languages, handwriting, tables, or 100+ docs/scan throughput",
     "ship": "tesseract (or a cloud OCR)", "why": "we ship ONE 37-symbol English charset trained from scratch and no punctuation; that is a product scope decision, not a capability we claim"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print("engines:")
    records, crnn, tess = run()
    print("\ntesseract: " + tess.version.splitlines()[0])

    agg = {arm: aggregate(records, arm) for arm in ARMS}
    print("\n%-68s %7s %7s %7s %7s %7s" % ("arm", "lineEx", "lnExNrm", "wordAcc", "CER", "fields"))
    for arm in ARMS:
        s = agg[arm]
        print("%-68s %6.1f%% %6.1f%% %6.1f%% %8.3f %6.1f%%"
              % (s["label"][:68], 100 * s["line_exact_rate_gt"],
                 100 * s["line_exact_norm_rate_gt"],
                 100 * s["word_accuracy"], s["cer"], 100 * s["field_recovery"]))
    for k, v in agg.items():
        print("  %-18s punct %4.1f%%  dropped lines %3d  lines beyond GT %3d"
              % (k, 100 * v["punct_rate"],
                 sum(r["arms"][k]["line_dropped"] for r in records),
                 sum(r["arms"][k]["line_inserted"] for r in records)))

    art = {
        "schema_version": 1,
        "generated_by": "ml/src/tesseract_compare.py",
        "engines": {
            "ours": {"model": "app/models/crnn.onnx", "bytes": crnn.bytes,
                     "input": list(crnn.input.shape), "dynamic_width": crnn.dynamic,
                     "runtime": "onnxruntime CPUExecutionProvider (the browser runs the same graph in onnxruntime-web)",
                     "charset": R.CHARSET, "params": 942166,
                     "note": "trained from scratch; no OCR library ships in the app bundle"},
            "tesseract": {"version_verbatim": tess.version, "languages": tess.langs,
                          "traineddata_bytes": tess.traineddata,
                          "oem": "default (LSTM, oem 1, Tesseract 4.x)",
                          "invocation": "tesseract <png> stdout -l eng --psm {7 for a line crop, 3 for a page}, OMP_THREAD_LIMIT=1"},
        },
        "bench": {
            "documents": len(records), "doc_types": sorted({r["doc"] for r in records}),
            "profiles": sorted({r["profile"] for r in records}),
            "gt_lines": sum(r["n_gt_lines"] for r in records),
            "distinct_gt_lines": len({l for r in records for l in r["gt_lines"]}),
            "scored_rows_with_punctuation": sum(
                sum(1 for l in r["gt_lines"] if any(c not in R.CHARSET for c in l.upper()))
                for r in records),
            "detected_bands": sum(r["n_bands"] for r in records),
            "samples_dir": str(SAMPLES),
            "ground_truth": "ml/src/capture.py DOCS via /tmp/sightline_samples/gt_lines.json (the lines the renderer DREW)",
            "crop_set_fixed": "lines detected ONCE per page on the deskewed grayscale; every arm crops the same boxes",
            "metric_definitions": {
                "word_accuracy": "fraction of GT words present in the arm's read text after norm() (evaluate.word_accuracy -- the app's own definition)",
                "cer": "edit distance / GT characters on norm()ed text (bench_hackathon.cer)",
                "field_recovery": "fraction of the doc's evaluate.FIELDS values present in the read text (norm substring, as js/pipeline.js fields_found)",
                "line_exact": "GT lines aligned to engine lines with difflib (stdlib), uppercased both sides; a missed or merged band costs one line",
            },
        },
        "app_end_to_end_reference": {
            "source": "artifacts/pipeline_raw_shipped.json (the shipped browser pipeline, restorer + candidate election, own detection)",
            "word_accuracy": 0.4594, "cer": 0.3031, "field_recovery": 0.2083, "n_docs": 12,
            "why_not_a_row_in_the_table": "it includes the app's restorer and its three-candidate election, so it is not a single-engine row; it is the reference our harness numbers sit beside",
        },
        "arms": agg,
        "paired": {
            "identical_pixels_ours_vs_tess_arm_a_strict":
                paired(records, "ours_c32_raw", "tess_c32_raw"),
            "identical_pixels_ours_vs_tess_arm_a_punct_blind":
                paired(records, "ours_c32_raw", "tess_c32_raw", blind=True),
            "best_ours_vs_fair_tess_strict":
                paired(records, "ours_c32_clahe", "tess_nat_raw"),
            "best_ours_vs_fair_tess_punct_blind":
                paired(records, "ours_c32_clahe", "tess_nat_raw", blind=True),
            "ours_vs_tesseract_drop_in_page_strict":
                paired(records, "ours_c32_clahe", "tess_page_raw"),
            "ours_vs_tesseract_drop_in_page_punct_blind":
                paired(records, "ours_c32_clahe", "tess_page_raw", blind=True),
        },
        "error_profiles": {arm: {
            "line_exact_rate_gt": agg[arm]["line_exact_rate_gt"],
            "line_exact_norm_rate_gt": agg[arm]["line_exact_norm_rate_gt"],
            "wrong_lines": wrong_examples(records, arm, 8),
            "punct_rate": agg[arm]["punct_rate"],
            "missed_chars_by_gt_char": char_confusions(records, arm, 12),
            "dropped_lines": sum(r["arms"][arm]["line_dropped"] for r in records),
            "inserted_lines": sum(r["arms"][arm]["line_inserted"] for r in records),
        } for arm in ARMS},
        "decision_table": DECISION_TABLE,
        "records": [{k: v for k, v in r.items() if k != "arms"}
                    | {"arms": {a: {kk: vv for kk, vv in r["arms"][a].items() if kk != "by_gt"}
                                for a in ARMS}} for r in records],
        "limits": [
            "12 synthetic rendered documents (3 doc types x 4 capture profiles, one font, generated by ml/src/capture.py) -- not real photographs; every number here is a synthetic-render number and the template text is the renderer's own ground truth.",
            "Tesseract 4.1.1 (leptonica 1.82.0), eng only, default oem (LSTM); a Tesseract 5 or a different traineddata/language set is a different engine and would need a re-run.",
            "Single detector for all e2e arms: the app's projection-profile detector on the deskewed grayscale. Detection quality caps every line-level number; Tesseract arm (c) uses its OWN layout analysis and is therefore not detector-limited in the same way.",
            "Arm (a) is genuinely unfair to Tesseract and is labelled that way everywhere it appears: the 32px crop is our model's contract, including the 320px width clamp that horizontally squeezes long lines. Arm (b) is the row to quote.",
            "Tesseract is scored with no character whitelist and its sentence-case/punctuation output is then uppercased and whitespace-collapsed on both sides; punctuation is reported as its own column instead of being scored away.",
            "Our engine gets two page variants (raw and CLAHE) because the app preprocesses; the shipped app also runs a restorer and a three-candidate election that this harness does not reproduce, so our harness rows are NOT the app's end-to-end number -- see app_end_to_end_reference.",
            "No confidence thresholding, no lexicon/field-constrained decode and no rejection path is applied to either engine here: this measures raw recognition, not the product's field policy.",
            "Sample size for the field metric is 24 expected fields over 12 documents; per-profile rates are over 3 documents each and should not be read as more than a direction.",
        ],
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(art, indent=2, default=str) + "\n")
    print("\nwrote " + str(outp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
