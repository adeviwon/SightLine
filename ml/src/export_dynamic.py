"""
export_dynamic.py — re-export the CRNN with a DYNAMIC width axis.

WHY THIS FILE EXISTS
--------------------
`app/models/crnn.onnx` (the shipped artifact) declares a FIXED input width of
256, so `recognizer.js` right-pads every crop up to 256 and decodes that padded
tensor. Right-padding is NOT inert here: the recurrent stage is a 2-layer
BIDIRECTIONAL LSTM, so the backward pass starts at the RIGHT EDGE of the padded
tensor — inside the zero padding — and moves the hidden states over the CONTENT
too. Measured max |Δ| over the valid frames: 9.10 at content_w=64, 15.28 at
200. Truncating the decode to the content's frames (which the app already does)
removes trailing characters but cannot undo that perturbation.

The root cause of the fixed width is `SightLineCRNN.set_export_mode(True)`: the
explicit per-timestep Python LSTM loop (`_lstm_loop`) unrolls the recurrence at
trace time, which BAKES T into the graph. Tracing the FUSED `nn.LSTM` instead
(export mode OFF) lets torch emit a real ONNX `LSTM` op, whose sequence axis is
dynamic natively — so the graph accepts [B, 1, 32, W] for ANY W and computes
what eager PyTorch computes.

So the fix is a re-export with `dynamic_axes` on width (dim 3) and batch (dim 0),
writing to `artifacts/crnn_dynamic.onnx`. The shipped `app/models/crnn.onnx` is
DELIBERATELY NOT overwritten by this module.

WHAT IT PROVIDES
----------------
  build_dynamic(ckpt)          load crnn.pt, eval(), fused-LSTM (export) mode
  export_dynamic(...)          the torch.onnx.export call + size report
  verify_parity(...)           Python-ORT vs PyTorch at several widths -> table
  choose_bucket(w, buckets)    fallback: smallest bucket >= w, never smaller
  BUCKETS                      the fallback bucket set (widest = legacy 256)
  write_fixture(path)          crop bytes + Python reference decodes for the
                               browser check (app/tools/dynamic_width_check.js)
  report_impact(...)           crop-level exact/CER: natural vs shipped vs fix

CLI
    bash run.sh py ml/src/export_dynamic.py                 # export + parity
    bash run.sh py ml/src/export_dynamic.py --fixture       # browser fixture
    bash run.sh py ml/src/export_dynamic.py --impact 200    # product impact
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

CKPT = ROOT / "models" / "ocr" / "crnn.pt"
DYNAMIC_ONNX = ROOT / "artifacts" / "crnn_dynamic.onnx"
SHIPPED_ONNX = ROOT / "app" / "models" / "crnn.onnx"
FIXTURE = ROOT / "artifacts" / "dynamic_width_fixture.json"
CROPS_NPZ = ROOT / "data" / "render_corpus" / "crops.npz"

# The fallback rung: a fixed-width graph padded only to the nearest bucket >=
# content width. The ladder must span the WHOLE reachable content range:
# build_dataset.crop_to_array clamps every crop to [MIN_W, MAX_W] = [16, 320],
# so the top rung is MAX_W. Without that rung any crop wider than the previous
# top (256) would be assigned a bucket BELOW its content and silently lose its
# tail -- the same class of loss as truncating at a cap, which measured WORSE
# (3.2% vs 3.6% word exact, CER 85.1% vs 70.8%).
BUCKETS = (64, 96, 128, 192, 256, 320)

PARITY_WIDTHS = (32, 47, 63, 64, 128, 200, 256)


def choose_bucket(content_w: int, buckets=BUCKETS) -> int:
    """
    The smallest bucket >= content_w. NEVER smaller than the content.

    A bucket below the content width would truncate real ink, so the return
    value is always >= content_w; content wider than the largest bucket is
    clamped up to it (the caller is expected to have clamped content to the
    training MAX_W first, so this only fires on a genuinely oversized crop).
    """
    cw = int(content_w)
    for b in sorted(buckets):
        if b >= cw:
            return int(b)
    return int(max(buckets))


def build_dynamic(ckpt=CKPT):
    """The checkpoint in FUSED-LSTM (export) mode: the dynamic-width trace."""
    import torch
    from ocr_model import build
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    m = build(ck.get("num_classes", 38))
    m.load_state_dict(ck["model"])
    m.eval()
    # export_safe_lstm=False -> forward uses the FUSED nn.LSTM, which the ONNX
    # exporter lowers to a real LSTM op with a dynamic sequence axis. Setting
    # this True is what made the old graph fixed-width; see ocr_model._lstm_loop.
    m.set_export_mode(False)
    return m


def export_dynamic(ckpt=CKPT, out=DYNAMIC_ONNX, opset=17, width=128) -> Path:
    """
    Trace the fused-LSTM model with a dynamic width + batch axis.

    `width` is only the DUMMY trace width; it is not baked into the graph, so
    the value does not matter beyond giving the tracer a concrete shape to
    start from. Any torch warning about a variable-length LSTM is printed.
    """
    import torch
    torch.set_num_threads(1)
    m = build_dynamic(ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, 32, int(width))
    # dynamo=False is LOAD-BEARING. This torch defaults torch.onnx.export to the
    # dynamo exporter, and its decomposition of aten.lstm rebuilds the SAME
    # per-timestep Python loop that set_export_mode(True) used -> the trace
    # fails with "stack expects each tensor to be equal size, but got
    # torch.Size([Min(((((((s0 - 1)//4)) - 1)//2)) + 1, ...)". Only the legacy
    # TorchScript exporter lowers the FUSED nn.LSTM to a real ONNX LSTM op,
    # whose sequence axis is dynamic natively -- which is the whole fix.
    torch.onnx.export(
        m, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 3: "width"},
                      "logits": {0: "batch", 1: "timesteps"}},
        opset_version=opset, do_constant_folding=True, dynamo=False)
    # onnxruntime-web has no MountedFiles API, so a Python-style external .data
    # sidecar loads on a laptop and fails on a phone. Reuse the exporter's
    # inliner rather than writing a second one.
    from export_onnx import _inline_external_data
    _inline_external_data(out)
    size = out.stat().st_size
    print(f"[export] crnn_dynamic.onnx  {size/1024/1024:.2f} MB  "
          f"(dynamic batch + width, dummy width {width})", flush=True)
    return out


def _ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def verify_parity(model=None, onnx=DYNAMIC_ONNX, widths=PARITY_WIDTHS):
    """
    Compare the dynamic ONNX graph against PyTorch at each width.

    Returns (rows, ok). Each row: (W, T_torch, T_onnx, max|Δ|, torch_str,
    onnx_str, match). `ok` is False if any argmax string differs or any delta
    exceeds tolerance. Uses a FIXED random seed so the rows are reproducible.
    """
    import torch
    torch.set_num_threads(1)
    from realdata import decode_greedy
    if model is None:
        model = build_dynamic()
    sess = _ort_session(onnx)
    rng = np.random.default_rng(1234)
    rows, ok = [], True
    for W in widths:
        x = rng.uniform(-1, 1, (1, 1, 32, int(W))).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        a, b = ref[0].argmax(-1), got[0].argmax(-1)
        d = float(np.abs(ref - got).max())
        sa, sb = decode_greedy(a), decode_greedy(b)
        match = (sa == sb) and ref.shape == got.shape
        ok = ok and match and d <= 1e-3
        rows.append((int(W), int(ref.shape[1]), int(got.shape[1]), d, sa, sb, match))
    return rows, ok


def _print_parity(rows, ok):
    print(f"  {'W':>4} {'T_torch':>7} {'T_onnx':>6} {'max|Δ|':>10}  "
          f"{'torch':<20} {'onnx':<20} match")
    for W, tt, to, d, sa, sb, match in rows:
        print(f"  {W:>4} {tt:>7} {to:>6} {d:>10.2e}  "
              f"{sa!r:<20} {sb!r:<20} {'yes' if match else 'NO'}")
    print(f"  parity: {'OK' if ok else 'FAIL'}")


def write_fixture(path=FIXTURE, widths=(32, 47, 63, 64, 128, 200, 256),
                  ckpt=CKPT):
    """
    Crop bytes + Python reference decodes for the BROWSER check.

    The crops are real rendered lines, resized to exactly `W` px wide at
    CROP_H=32 (bilinear) so the browser and Python feed byte-identical inputs;
    the browser must get the SAME decode the Python model gets at that width.
    `shipped` is what `recognizer.js` produces today (pad to 256, decode the
    content's frames) so the browser can show the bug too.
    """
    import torch
    from PIL import Image
    from realdata import decode_greedy
    torch.set_num_threads(1)
    z = np.load(CROPS_NPZ, allow_pickle=True)
    crops = z["crops"]
    m = build_dynamic(ckpt)
    out = {"crop_h": 32, "widths": [], "crops": {}, "py_natural": {},
           "py_natural_t": {}, "py_shipped": {}}
    for i, W in enumerate(widths):
        src = np.asarray(crops[3 + i], np.uint8)
        arr = np.asarray(Image.fromarray(src).resize((int(W), 32),
                                                     Image.BILINEAR), np.uint8)
        arr = np.ascontiguousarray(arr)
        x = ((arr.astype(np.float32) / 255.0) - 0.5) / 0.5
        with torch.no_grad():
            nat = m(torch.from_numpy(x)[None, None])[0].argmax(-1).numpy()
            padded = np.zeros((1, 1, 32, 256), np.float32)
            padded[0, 0, :, :int(W)] = x
            sh = m(torch.from_numpy(padded))[0].argmax(-1).numpy()
        limit = max(1, int(W) // 8)          # what recognizer.timestepsFor does
        k = str(int(W))
        out["widths"].append(int(W))
        out["crops"][k] = base64.b64encode(arr.tobytes()).decode()
        out["py_natural"][k] = decode_greedy(nat)
        out["py_natural_t"][k] = int(nat.shape[0])
        out["py_shipped"][k] = decode_greedy(sh, limit=min(limit, sh.shape[0]))
    out["charset"] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    Path(path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[fixture] {path}  widths {out['widths']}")
    return out


def _content_w(crop_u8) -> int:
    """The served width build_dataset.crop_to_array would produce for this crop."""
    h, w = np.asarray(crop_u8).shape[:2]
    scale = 32.0 / float(h)
    return max(16, min(320, int(round(w * scale))))


def report_training_padding(batch_size=64, width_tol=24, split="train"):
    """
    The padding distribution TRAINING actually used.

    `train_ocr.collate` pads a width-bucketed batch only to the batch's own
    max (not to 256), and passes per-sample `input_lengths` so the loss never
    supervises the padding. This reproduces that bucketing on the real widths
    and reports the pad fraction (wmax - w)/w a crop saw in training -- the
    distribution the serve path should match.
    """
    z = np.load(ROOT / "artifacts" / "ocr_train.npz", allow_pickle=True)
    widths, splits = z["widths"], z["splits"]
    idx = np.where(np.array([str(s) for s in splits]) == split)[0]
    order = idx[np.argsort(widths[idx], kind="stable")]
    fracs = []
    batch = []
    cur_w = None
    buckets = []
    for k in order:
        w = int(widths[k])
        if batch and (w - cur_w) > width_tol:
            buckets.append(batch)
            batch = []
        batch.append(w)
        cur_w = w
    if batch:
        buckets.append(batch)
    for b in buckets:
        for s in range(0, len(b), batch_size):
            run = b[s:s + batch_size]
            wmax = max(run)
            for w in run:
                fracs.append(0.0 if w == 0 else (wmax - w) / float(w))
    f = np.asarray(fracs)
    print(f"training padding (split={split}, n={f.size}, batch={batch_size}, "
          f"width_tol={width_tol}):")
    print(f"  pad fraction (wmax-w)/w: mean {100*f.mean():.2f}%  "
          f"median {100*np.median(f):.2f}%  p90 {100*np.percentile(f,90):.2f}%  "
          f"max {100*f.max():.2f}%")
    print(f"  batches padded to the batch max (never to 256); padding value 0; "
          f"input_lengths masks it out of the CTC loss")
    return f


def report_impact(n=200, ckpt=CKPT):
    """
    Crop-level exact-match + CER on a held-out slice of OUR rendered documents.

      (i)   natural width   — PyTorch at the crop's own width (the training cond.)
      (ii)  shipped path    — app/models/crnn.onnx padded to 256, decoded to
                              min(floor(cw/8), T) frames — recognizer.js as it is
      (iii) fix             — the DYNAMIC graph at the crop's own width

    Also reports the population of crops whose content width exceeds the
    shipped graph's 256 (their tail is cut at serve time, independent of
    padding) and the same three arms on that slice.
    """
    import torch
    torch.set_num_threads(1)
    import build_dataset as B
    import realdata as R
    from PIL import Image
    from realdata import decode_greedy

    model = build_dynamic(ckpt)
    shipped = _ort_session(SHIPPED_ONNX)
    dyn = _ort_session(DYNAMIC_ONNX) if DYNAMIC_ONNX.exists() else None

    z = np.load(CROPS_NPZ, allow_pickle=True)
    crops, labels, split = z["crops"], z["labels"], z["split"]
    test = [i for i in range(len(crops)) if str(split[i]) == "test"]
    widths = np.array([_content_w(crops[i]) for i in test])

    def arms_for(idxs):
        acc = {k: [0, 0.0, 0] for k in ("natural", "shipped", "fix")}
        rows = []
        for i in idxs:
            arr = B.crop_to_array(Image.fromarray(np.asarray(crops[i], np.uint8)))
            cw = int(arr.shape[1])
            x = ((arr.astype(np.float32) / 255.0) - 0.5) / 0.5
            ref = R.norm_label(labels[i])
            with torch.no_grad():
                got_nat = decode_greedy(
                    model(torch.from_numpy(x)[None, None])[0].argmax(-1).numpy())
            # Shipped: the 256px graph. Content wider than 256 is tail-cut.
            cut = x[:, :256]
            padded = np.zeros((1, 1, 32, 256), np.float32)
            padded[0, 0, :, :cut.shape[1]] = cut
            sh = shipped.run(None, {"input": padded})[0][0].argmax(-1)
            limit = min(max(1, cw // 8), sh.shape[0])   # recognizer.js Ttrue
            got_sh = decode_greedy(sh, limit=limit)
            got_fix = got_nat
            if dyn is not None:
                got_fix = decode_greedy(
                    dyn.run(None, {"input": x[None, None]})[0][0].argmax(-1))
            for name, got in (("natural", got_nat), ("shipped", got_sh),
                              ("fix", got_fix)):
                acc[name][0] += 1
                acc[name][1] += 1.0 if R.norm_label(got) == ref else 0.0
                acc[name][2] += R.cer(got, ref)
            rows.append({"cw": cw,
                         "agree": int(R.norm_label(got_sh) == R.norm_label(got_nat)),
                         "exact_sh": int(R.norm_label(got_sh) == ref),
                         "cer_sh": R.cer(got_sh, ref),
                         "exact_nat": int(R.norm_label(got_nat) == ref)})
        return acc, rows

    PAD_BANDS = [(0.0, 0.25), (0.25, 1.0), (1.0, 3.0), (3.0, 1e9)]

    def band_report(rows):
        print(f"    {'pad band (256-cw)/cw':<22} {'n':>4} {'shipped==natural':>17} "
              f"{'shipped exact':>13} {'shipped CER':>11}")
        for lo, hi in PAD_BANDS:
            sel = [r for r in rows if lo <= (256 - r["cw"]) / r["cw"] < hi]
            if not sel:
                continue
            n = len(sel)
            ag = sum(r["agree"] for r in sel) / n
            ex = sum(r["exact_sh"] for r in sel) / n
            ce = sum(r["cer_sh"] for r in sel) / n
            print(f"    [{lo:.2f}, {hi if hi < 1e8 else float('inf'):>4})"
                  f"{'':<11} {n:>4} {100*ag:>16.1f}% {100*ex:>12.1f}% {ce:>11.3f}")

    def show(title, acc):
        print(f"  {title}  ({acc['natural'][0]} crops)")
        print(f"    {'arm':<10} {'exact':>8} {'CER':>7}")
        res = {}
        for name, (k, ex, ce) in acc.items():
            k = max(1, k)
            res[name] = {"n": acc[name][0], "exact": ex / k, "cer": ce / k}
            print(f"    {name:<10} {100*ex/k:>7.1f}% {ce/k:>7.3f}")
        return res

    rng = np.random.default_rng(0)
    test_w = np.array(widths)
    keep = test_w >= 16
    test = [i for i, k in zip(test, keep) if k]
    widths = test_w[keep]
    narrow_all = [i for i, w in zip(test, widths) if w <= 256]
    wide_pool = [i for i, w in zip(test, widths) if w > 256]
    overall = list(rng.choice(np.array(test), size=min(n, len(test)),
                              replace=False))
    wide = list(rng.choice(np.array(wide_pool),
                           size=min(n, len(wide_pool)), replace=False)) if wide_pool else []

    print(f"population: {len(test)} held-out crops; max content_w "
          f"{int(widths.max())}px (training clamp MAX_W=320); "
          f"{len(wide_pool)} ({100.0*len(wide_pool)/max(1,len(test)):.1f}%) wider "
          f"than 256 — their tail is cut at serve time by the 256px graph; only "
          f"{len(narrow_all)} crops are narrow enough for the 256 graph to hold "
          f"their full ink")
    acc_overall, _ = arms_for(overall)
    res_overall = show(f"random {len(overall)}-crop slice (all widths)", acc_overall)
    acc_near, rows_near = arms_for(narrow_all)
    res_near = show("all crops <=256px (padding bug only)", acc_near)
    band_report(rows_near)
    # Population: how many held-out crops sit in each pad band at all.
    print("    held-out population by pad band (256-cw)/cw:")
    for lo, hi in PAD_BANDS:
        cnt = int(((widths >= 16) & (widths <= 256)
                   & ((256 - widths) / np.maximum(widths, 1) >= lo)
                   & ((256 - widths) / np.maximum(widths, 1) < hi)).sum())
        print(f"      [{lo:.2f}, {hi if hi < 1e8 else float('inf'):>4}): {cnt} crops")
    res_wide = None
    if wide:
        acc_wide, _ = arms_for(wide)
        res_wide = show(f">256px (padding bug + serve-time tail cut; "
                        f"random {len(wide)} of {len(wide_pool)})", acc_wide)
    return {"overall": res_overall, "near": res_near, "wide": res_wide,
            "n_wide": len(wide_pool), "n_narrow": len(narrow_all),
            "n_test": len(test)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--impact", type=int, default=0, metavar="N")
    ap.add_argument("--train-pad", action="store_true")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)

    if a.train_pad:
        report_training_padding()
        return 0
    if a.fixture:
        write_fixture()
        return 0
    if a.impact:
        report_impact(a.impact)
        return 0

    out = export_dynamic(width=a.width, opset=a.opset)
    rows, ok = verify_parity(onnx=out)
    _print_parity(rows, ok)
    print(f"  artifact {out.relative_to(ROOT)}  "
          f"{out.stat().st_size/1024/1024:.2f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
