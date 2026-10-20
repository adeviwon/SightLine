"""
fix_padding.py — serve-time width: NO right-padding, ever.

THE BUG
-------
`models/onnx/crnn.onnx` was exported at a FIXED width of 256px, so every crop
narrower than 256 was right-padded with zeros before inference and the whole
256-wide logit sequence was decoded. That is not the condition the model was
trained under, and it is not inert:

  * training fed each crop at its NATURAL width and passed CTC `input_lengths`
    (see train_ocr.collate: `input_lengths = model.timesteps(width)`), so a crop
    of width W was scored against T(W) = W/8 timesteps;
  * the CRNN's recurrent stage is a 2-layer BIDIRECTIONAL LSTM, so its backward
    pass STARTS AT THE RIGHT EDGE — which, in a padded tensor, is inside the
    zero padding. The padding therefore moves the hidden states in the valid
    region, not just the frames beyond it.

Measured on the shipped checkpoint (`models/ocr/crnn.pt`), comparing a real
64px crop padded to 256 against the same crop at 64px:

    content   T_nat  decode(natural)  decode(padded, limit=T_nat)
    64        8      '1071'           '1073'
    128       16     '01143008'       '0114300'
    200       25     'U C2 0'         'RAE  CB 0'

and the max |Δ| between the two runs' logits over the VALID frames (first
T_nat = 8) is 9.1 — so this is not a trailing-frame problem that a decoder
limit can fix. Truncating the decode to the valid frames removes spurious
TRAILING characters; it cannot undo the perturbation of the valid ones.

WHY THE FIX IS A DYNAMIC-WIDTH EXPORT
-------------------------------------
The fixed width existed because the recurrent stage refused to trace with a
symbolic sequence length. The old workaround was an explicit per-timestep
Python LSTM loop (`SightLineCRNN.set_export_mode(True)`), which unrolls the
recurrence at trace time and therefore BAKES T into the graph — that is what
made the width fixed, not the LSTM itself.

Tracing the FUSED `nn.LSTM` instead (export mode OFF) lets torch emit a real
ONNX `LSTM` op, whose sequence axis is dynamic natively, so the graph accepts
[B, 1, 32, W] for ANY W and computes exactly what eager PyTorch computes:

    W= 64  torch T= 8  onnx T= 8  max|Δ| 8.6e-06  argmax same
    W=128  torch T=16  onnx T=16  max|Δ| 1.3e-05  argmax same
    W=200  torch T=25  onnx T=25  max|Δ| 5.7e-05  argmax same
    W=256  torch T=32  onnx T=32  max|Δ| 1.7e-05  argmax same

So no padding is needed at all: serve the crop at its own width, decode all T
frames, and the serve path reproduces the trained condition exactly.

WHAT THIS MODULE PROVIDES
-------------------------
  timesteps(W)        exact T_out(W) of the conv stack, matching
                      ocr_model.SightLineCRNN.timesteps()
  frame_limit(cw, gw) the truncate rule for a graph that IS padded, so a
                      caller on the old fixed-width graph masks the padding
  pad_for_export(...) the serve helper: graph_w=None -> natural width, NO
                      padding (the fix); graph_w=256 -> the legacy padded
                      tensor plus the frame limit, kept so the regression test
                      can measure what the fix removed

Run `bash run.sh py ml/src/fix_padding.py` for the frame map.
"""

from __future__ import annotations

import numpy as np

CROP_H = 32              # keep in sync with realdata.CROP_H
MAX_W = 320              # the training clamp, realdata.MAX_W
LEGACY_EXPORT_W = 256    # the fixed width the old graph + recognizer.js used


def timesteps(width: int) -> int:
    """
    Exact number of CTC timesteps the conv stack emits for a crop of `width` px.

    Derived from the geometry in ocr_model.SightLineCRNN (two convs with
    kernel 3, stride (1,2), padding 1 reduce width to floor((W+1)/2) each, then
    a final MaxPool2d((1,2)) halves it once more), and CROSS-CHECKED against
    `SightLineCRNN.timesteps()` — which runs the conv stack — over every width
    16..320 in test_fix_padding.py. A closed form is used here rather than a
    forward pass so the JS mirror and the tests can share one cheap function.

    T(W) == W // 8 exactly when W is a multiple of 8 (every crop width the
    render corpus produces); for other widths the floor divisions make it
    differ, e.g. W=47 -> 6 (47//8 == 5) and W=63 -> 8 (63//8 == 7).
    """
    w = int(width)
    if w < 1:
        return 1
    a = (w + 1) // 2          # conv stride (1,2), k=3, pad=1
    b = (a + 1) // 2          # second such conv
    return max(1, b // 2)     # final MaxPool2d((1, 2))


def frame_limit(content_w: int, graph_w: int | None) -> int:
    """
    Frames of a graph's output that correspond to CONTENT, not padding.

    With a dynamic-width graph there is no padding, so the limit is simply
    T(content_w). Against a fixed `graph_w` the equivalent rule is

        round(content_w * T(graph_w) / graph_w)

    which for graph_w=256 reduces to the content_w//8 the app used.
    """
    cw = int(content_w)
    if graph_w is None:
        return timesteps(cw)
    gw = int(graph_w)
    return max(1, int(round(cw * timesteps(gw) / gw)))


def pad_for_export(crop, content_w: int, graph_w: int | None = None):
    """
    Build the tensor to feed the recogniser, plus the frame limit to decode to.

    graph_w=None (THE FIX): returns the crop at its own width — no padding, so
    the padded-equivalent path IS the natural-width path and decodes
    identically by construction. `crop` may be [h, W] or [1, h, W]; values are
    expected already normalised to [-1,1] (realdata's transform), as the
    export_onnx._sample_real_crop crops are.

    graph_w=256 (LEGACY): returns a zero-padded [h, 256] tensor and the frame
    limit, i.e. exactly what recognizer.js did before this fix. Kept because
    the regression test has to be able to measure the defect it guards.
    """
    c = np.asarray(crop, np.float32)
    if c.ndim == 3:
        c = c[0]
    if c.ndim != 2:
        raise ValueError(f"expected a 2-D crop, got shape {c.shape}")
    cw = int(content_w)
    if c.shape[1] < cw:
        raise ValueError(f"crop is {c.shape[1]}px wide but content_w={cw}")
    c = np.ascontiguousarray(c[:, :cw], np.float32)
    if graph_w is None:
        return c, timesteps(cw)
    gw = int(graph_w)
    if cw > gw:
        raise ValueError(f"content_w={cw} exceeds the graph width {gw}")
    out = np.zeros((c.shape[0], gw), np.float32)
    out[:, :cw] = c
    return out, frame_limit(cw, gw)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from ocr_model import build

    m = build()
    print("frame map  T_out(W)   (model.timesteps vs fix_padding.timesteps)")
    bad = 0
    for w in range(16, MAX_W + 1):
        t_here, t_model = timesteps(w), m.timesteps(w)
        if t_here != t_model:
            bad += 1
            print(f"  MISMATCH W={w}: {t_here} vs {t_model}")
    step = [32, 64, 96, 128, 160, 192, 200, 224, 256, 288, 320]
    print("  " + "  ".join(f"W={w}->T={timesteps(w)}" for w in step))
    print(f"  all widths 16..{MAX_W}: {'AGREE' if bad == 0 else f'{bad} MISMATCHES'}")
    print(f"  multiples of 8: T(W) == W//8 -> "
          f"{all(timesteps(w) == w // 8 for w in range(8, MAX_W + 1, 8))}")
    print(f"  frame_limit(64, 256) = {frame_limit(64, LEGACY_EXPORT_W)}"
          f"   frame_limit(64, None) = {frame_limit(64, None)}")
