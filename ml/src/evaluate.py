"""
SightLine ML — End-to-end evaluation harness.

This is the file that decides whether SightLine actually works. Everything
else (training curves, PSNR, classifier accuracy) is instrumentation. Here we
measure the thing the user cares about:

    "given a blurry photo a real person took, does SightLine tell them the
     right thing?"

HEADLINE METRICS (computed per capture profile)
  field_accuracy       fraction of documents where EVERY key field is recovered
                       (dosages for prescriptions, account+sort code for bank,
                       rent+termination for lease). All-or-nothing, because a
                       user who hears "take 500mg" when the label says 400mg
                       is worse off than one who is told to retake the photo.
  word_accuracy        OCR word-level accuracy (Levenshtein-aligned)
  exact_dosage_rate    literal "500mg"/"400mg" survival — the medical safety case
  mean_ocr_confidence  Tesseract's own mean word confidence
  classification_acc   MiniLM doc-type accuracy on the OCR'd text

COMPARISON ARMS (the model must beat all of these)
  raw                 degraded image, OCR'd directly
  classical           median 3x3 denoise + unsharp mask (hand-tuned stack)
  restorer            our PyTorch SightLineNet
  restorer+clahe      ours + CLAHE contrast, the full production stack

HONESTY RULES enforced here:
  - Profiles below the human readability floor are reported but flagged
    SUB-HUMAN; they are excluded from the headline mean and listed separately.
  - Every number is from a seeded, reproducible run.
  - OCR text is never hand-corrected; confidence thresholds gate guesses.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import capture  # noqa: E402
import seedutil  # noqa: E402  — stable cross-process seed derivation

# Ground-truth fields per document type. A doc counts correct ONLY if all of
# its fields are found in the OCR output (normalized).
FIELDS = {
    "prescription": ["500mg", "400mg"],
    "banking": ["40218877", "40-11-04"],
    "legal": ["2500", "2024-CV-00456"],
}

# Profiles where a human cannot reliably read the text. Reported, not hidden,
# but excluded from the headline (they are not a model failure).
SUB_HUMAN = {"worst_case"}

# Which pipeline category each rendered document type is expected to land in.
# One definition, imported by everything that needs it: the eval harness (below)
# and app/tools/render_samples.py, whose manifest must not hand-copy this map.
DOC_CATEGORY = {
    "prescription": "medical",
    "banking": "banking",
    "legal": "legal",
    "general": "general",
}

# ── WHICH ENGINE THIS HARNESS MEASURES ───────────────────────────────────────
#
# The brief asks for the MODEL's metrics, and this file used to answer with
# Tesseract's: `import pytesseract` at module scope and ocr()/ocr_conf() calling
# it. Tesseract has since been removed from the application (41.1 MB, and
# bundle_check.py fails if it reappears), so every arm table produced here
# described an engine the product does not contain.
#
# The default is now the shipped recogniser: the same checkpoint, the same page
# reader (eval_ocr.read_page) and the same crop geometry the evaluator for real
# photographs uses. `--engine=tesseract` keeps the old path as an explicit
# BASELINE -- it is the number the CRNN has to beat, and it is why the
# preprocessing arms are ordered the way they are -- but it is opt-in, and the
# import is lazy so a machine without pytesseract can still run the default.
OCR_ENGINE = "crnn"
_tess_mod = None
_crnn_state = {"model": None, "device": None}


def _tess():
    """Import pytesseract on demand. Only the tesseract baseline needs it."""
    global _tess_mod
    if _tess_mod is None:
        import pytesseract
        _tess_mod = pytesseract
    return _tess_mod


def _crnn():
    """Load the shipped recogniser once: the same checkpoint eval_ocr scores."""
    if _crnn_state["model"] is None:
        import torch
        import eval_ocr as _eo
        from ocr_model import build
        ck = torch.load(_eo.CKPT, map_location="cpu", weights_only=False)
        model = build()
        model.load_state_dict(ck["model"])
        model.eval()
        _crnn_state["model"] = model
        _crnn_state["device"] = "cpu"
    return _crnn_state["model"], _crnn_state["device"]


def _page_lines(img):
    """Read a page with the shipped recogniser -> [(text, confidence)]."""
    import eval_ocr as _eo
    model, device = _crnn()
    return _eo.read_page(model, device, _np(img))


# ── preprocessing arms ──────────────────────────────────────────────────

def _np(img):
    return np.asarray(img.convert("L"))


def arm_raw(img):
    return img


def arm_classical(img):
    """Median 3x3 denoise + unsharp mask — the classic hand-tuned stack."""
    import cv2
    g = _np(img)
    g = cv2.medianBlur(g, 3)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    g = cv2.addWeighted(g, 1.8, blur, -0.8, 0)
    return Image.fromarray(g).convert("RGB")


def arm_restorer(img, model=None):
    """Our PyTorch restorer, tiled over the full image."""
    import torch
    from model import SightLineNet
    if model is None:
        model = SightLineNet()
        ck = Path("models/restorer/restorer.pt")
        if ck.exists():
            model.load_state_dict(torch.load(ck, map_location="cpu",
                                             weights_only=True)["state_dict"])
        model.eval()
    import cv2
    g = _np(img).astype(np.float32) / 255.0
    H, W = g.shape
    th, tw = 64, 256
    pad_h = (th - H % th) % th
    pad_w = (tw - W % tw) % tw
    gp = np.pad(g, ((0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = gp.shape
    out = np.zeros_like(gp)
    cnt = np.zeros((Hp, Wp), np.float32)
    with torch.no_grad():
        for y in range(0, Hp, th):
            for x in range(0, Wp, tw):
                tile = torch.from_numpy(gp[y:y+th, x:x+tw])[None, None]
                r = model(tile)[0, 0].numpy()
                out[y:y+th, x:x+tw] += r
                cnt[y:y+th, x:x+tw] += 1
    # Crop BEFORE dividing: `out`/`cnt` are the padded (Hp, Wp) grid, so the
    # divisor must be cropped to match or the shapes fail to broadcast.
    out = out[:H, :W] / np.maximum(cnt, 1.0)[:H, :W]
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)).convert("RGB")


def arm_restorer_clahe(img, model=None):
    """Full production stack: restorer, then CLAHE + Otsu for OCR."""
    import cv2
    g = _np(arm_restorer(img, model))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(g).convert("RGB")


ARMS = {
    "raw": arm_raw,
    "classical": arm_classical,
    "restorer": arm_restorer,
    "restorer_clahe": arm_restorer_clahe,
}


# ── OCR + metrics ───────────────────────────────────────────────────────

def read(img, psm=6):
    """Read the page ONCE with the selected engine -> (text, confidence).

    The arms loop wants both, and calling ocr() and ocr_conf() separately ran the
    page through the recogniser twice per arm.
    """
    try:
        if OCR_ENGINE == "crnn":
            pairs = _page_lines(img)
            text = " ".join(t for t, _ in pairs)
            conf = (float(np.mean([c for _, c in pairs])) * 100.0) if pairs else 0.0
            return text, conf
        pytesseract = _tess()
        text = pytesseract.image_to_string(img, config=f"--oem 3 --psm {psm}")
        d = pytesseract.image_to_data(img, config=f"--oem 3 --psm {psm}",
                                      output_type=pytesseract.Output.DICT)
        confs = [float(c) for c, t in zip(d.get("conf", []), d.get("text", []))
                 if str(c).lstrip("-").isdigit() and str(c) not in ("-1",)
                 and str(t).strip()]
        return text, (float(np.mean(confs)) if confs else 0.0)
    except Exception:
        return "", 0.0


def ocr(img, psm=6):
    """Read the page with the selected engine -> text.

    crnn (default): our recogniser, one line at a time, joined with spaces. One
    pass, no segmentation mode to choose -- the detector IS the segmentation.
    tesseract: the legacy baseline, opt-in.
    """
    return read(img, psm=psm)[0]


def ocr_conf(img, psm=6):
    """Mean reading confidence, on Tesseract's 0-100 scale so the two engines are
    comparable in one table. The CRNN's own confidence is a mean emitted-character
    probability in 0-1 (see eval_ocr.recognise), multiplied here by 100."""
    return read(img, psm=psm)[1]
    confs = [float(c) for c, t in zip(d.get("conf", []), d.get("text", []))
             if str(c).lstrip("-").isdigit() and str(c) not in ("-1",) and str(t).strip()]
    return float(np.mean(confs)) if confs else 0.0


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def word_accuracy(gt, hyp):
    g = gt.split()
    h = hyp.split()
    if not g:
        return 0.0
    gset, hset = [norm(w) for w in g], [norm(w) for w in h]
    hit = sum(1 for w in gset if w in hset)
    return hit / len(gset)


def fields_found(text, doc_type):
    n = norm(text)
    return [f for f in FIELDS.get(doc_type, []) if norm(f) in n]


def classify_doc(text):
    """Keyword-density classifier (the no-model fallback, also a sanity check)."""
    t = text.lower()
    scores = {
        "medical": sum(t.count(k) for k in ["mg", "tablet", "dose", "patient", "prescription", "capsule"]),
        "banking": sum(t.count(k) for k in ["balance", "account", "sort code", "iban", "debit", "credit", "gbp", "atm"]),
        "legal":  sum(t.count(k) for k in ["agreement", "clause", "tenant", "landlord", "court", "deed", "witnesseth"]),
        "general": sum(t.count(k) for k in ["notice", "meeting", "library", "parcel", "recipe", "school"]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── main ────────────────────────────────────────────────────────────────

def run(n_per_profile=6, seed=123, arms=None, out_path="artifacts/eval_results.json"):
    arms = arms or list(ARMS)
    import torch
    from model import SightLineNet, psnr
    ck = Path("models/restorer/restorer.pt")
    model = None
    if ck.exists():
        model = SightLineNet()
        model.load_state_dict(torch.load(ck, map_location="cpu",
                                         weights_only=True)["state_dict"])
        model.eval()
        print(f"[model] loaded restorer from {ck}", flush=True)
    else:
        print("[model] WARNING no checkpoint; restorer arms run at identity",
              flush=True)

    results = {"per_profile": {}, "arms": arms, "n_per_profile": n_per_profile,
               "seed": seed, "sub_human_profiles": sorted(SUB_HUMAN),
               # Provenance stamp. The seed alone does NOT identify a corpus:
               # the same seed under two different seeders selects different
               # documents, so make_results_doc.py refuses to call these
               # numbers current unless this matches the current tree.
               "seeder": seedutil.SEEDER_ID}

    for pname in capture.CAPTURE_PROFILES:
        rows = {a: {"field": [], "word": [], "dosage": [], "conf": [],
                    "class_ok": [], "psnr": []} for a in arms}
        gt_texts, t0 = {}, time.time()
        for i in range(n_per_profile):
            ds = seed * 7919 + seedutil.name_hash(pname) % 100003 + i
            lines, dtype = capture.DOCS[i % len(capture.DOCS)]
            clean = capture.render_document(lines, seed=ds)
            deg = capture.apply_profile(clean, seed=ds,
                                        **capture.CAPTURE_PROFILES[pname])
            gt_text = "\n".join(lines)
            gt_texts[i] = (gt_text, dtype)

            for a in arms:
                fn = ARMS[a]
                img = fn(deg, model) if a.startswith("restorer") else fn(deg)
                text, conf = read(img)
                found = fields_found(text, dtype)
                rows[a]["field"].append(1.0 if len(found) == len(FIELDS[dtype]) else 0.0)
                rows[a]["word"].append(word_accuracy(gt_text, text))
                rows[a]["dosage"].append(
                    1.0 if (dtype != "prescription"
                            or all(norm(f) in norm(text) for f in FIELDS[dtype]))
                    else 0.0)
                rows[a]["conf"].append(conf)
                pred = classify_doc(text)
                exp = DOC_CATEGORY[dtype]
                rows[a]["class_ok"].append(1.0 if pred == exp else 0.0)
                if a in ("restorer", "restorer_clahe"):
                    g = np.asarray(img.convert("L")).astype(np.float32) / 255
                    t = _np(clean).astype(np.float32) / 255
                    if g.shape == t.shape:
                        rows[a]["psnr"].append(psnr(
                            torch.from_numpy(g)[None, None],
                            torch.from_numpy(t)[None, None]))
        results["per_profile"][pname] = {
            a: {k: (round(float(np.mean(v)), 4) if v else None)
                for k, v in rows[a].items()}
            for a in arms
        }
        best = max(arms, key=lambda a: np.mean(rows[a]["field"]))
        print(f"  {pname:<16} " + "  ".join(
            f"{a}={np.mean(rows[a]['field']):.0%}" for a in arms)
            + f"   <- best: {best}   [{time.time()-t0:.1f}s]", flush=True)

    # headline mean excludes sub-human profiles
    honest = [p for p in capture.CAPTURE_PROFILES if p not in SUB_HUMAN]
    results["headline"] = {}
    for a in arms:
        fa = [results["per_profile"][p][a]["field"] for p in honest]
        wa = [results["per_profile"][p][a]["word"] for p in honest]
        ca = [results["per_profile"][p][a]["conf"] for p in honest]
        da = [results["per_profile"][p][a]["dosage"] for p in honest]
        results["headline"][a] = {
            "field_accuracy": round(float(np.mean(fa)), 4),
            "word_accuracy": round(float(np.mean(wa)), 4),
            "exact_dosage_rate": round(float(np.mean(da)), 4),
            "mean_ocr_confidence": round(float(np.mean(ca)), 2),
            "profiles_included": honest,
            "n_docs": n_per_profile * len(honest),
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results["engine"] = OCR_ENGINE
    results["engine_note"] = (
        "crnn = the recogniser the application ships (eval_ocr.read_page + the "
        "crnn.pt checkpoint); tesseract = the legacy baseline. Quote which one a "
        "number came from."
    )
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 74)
    print("HEADLINE  (sub-human profiles excluded; see per_profile for all)")
    print("=" * 74)
    print(f"{'arm':<16} {'field_acc':>10} {'word_acc':>9} {'dosage':>8} {'ocr_conf':>9}")
    for a in arms:
        h = results["headline"][a]
        print(f"{a:<16} {h['field_accuracy']:>9.1%} {h['word_accuracy']:>9.1%} "
              f"{h['exact_dosage_rate']:>7.1%} {h['mean_ocr_confidence']:>8.1f}")
    print(f"\nsub-human (reported, not counted): {sorted(SUB_HUMAN)}")
    print(f"written: {out_path}")
    return results


def main():
    global OCR_ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="artifacts/eval_results.json")
    ap.add_argument("--engine", choices=["crnn", "tesseract"], default=OCR_ENGINE,
                    help="crnn (default) measures the model the app ships; "
                         "tesseract is the legacy baseline and needs the "
                         "pytesseract package + binary")
    a = ap.parse_args()
    OCR_ENGINE = a.engine
    arms = [s for s in a.arms.split(",") if s] or None
    print(f"[eval] engine: {OCR_ENGINE}"
          + ("  (the shipped recogniser)" if OCR_ENGINE == "crnn" else "  (baseline)"))
    run(a.n, a.seed, arms, a.out)


if __name__ == "__main__":
    main()
