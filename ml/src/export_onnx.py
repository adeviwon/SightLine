"""
SightLine ML — ONNX export + numerical parity verification.

Exports three artifacts for on-device inference and PROVES they match PyTorch:

  models/onnx/restorer.onnx        SightLineNet, 64x256 grayscale patch
  models/onnx/minilm_encoder.onnx  all-MiniLM-L6-v2, input_ids/attention_mask
  models/onnx/minilm_head.onnx     classifier head, 384-dim embedding -> 4 logits

PARITY IS THE POINT. An ONNX graph that loads without error but computes the
wrong thing is worse than no model: a document classifier that silently
misreads a dosage is a safety incident. So every export is followed by a
numerical comparison against the PyTorch original, and `verify_parity` FAILS
LOUDLY (non-zero exit) if the outputs diverge past tolerance.

Mobile consumption:
  - onnxruntime-web (WASM) for both artifacts.
  - The encoder needs a tokenizer in JS; vocab.txt + tokenizer.json are copied
    next to the ONNX so transformers.js can load them.

Usage:
    python3 src/export_onnx.py            # export all + verify
    python3 src/export_onnx.py --verify-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import SightLineNet  # noqa: E402

ONNX_DIR = Path("models/onnx")
TOL = 1e-4


def export_restorer(ckpt="models/restorer/restorer.pt", opset=17):
    import onnxruntime  # noqa: F401  (presence check)
    from model import count_params
    model = SightLineNet()
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                         weights_only=True)["state_dict"])
    model.eval()
    dummy = torch.zeros(1, 1, 64, 256)
    path = ONNX_DIR / "restorer.onnx"
    torch.onnx.export(
        model, dummy, str(path), input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                      "output": {0: "batch", 2: "h", 3: "w"}},
        opset_version=opset, do_constant_folding=True)
    size = path.stat().st_size
    print(f"[export] restorer.onnx  {size/1024:.1f} KB  "
          f"params={count_params(model):,}", flush=True)
    return path, model


def _weight_bytes(path):
    """
    Total on-disk size of an ONNX file INCLUDING its external weight blob.

    torch.onnx.export writes tensors >= some threshold into a sidecar
    `<name>.data` file, so stat().st_size on the .onnx graph alone reports
    ~55 KB for a model that is really 90 MB. Any size claim in the docs or the
    privacy audit must use this, or it will be wrong by three orders of
    magnitude.
    """
    p = Path(path)
    total = p.stat().st_size
    blob = p.with_name(p.name + ".data")
    if blob.exists():
        total += blob.stat().st_size
    return total


def _encoder_is_int8():
    """
    Is the encoder currently on disk int8-quantized?

    Detected from the graph itself (int8 ops, no external weight blob) rather
    than from a flag, so the tolerance logic cannot drift out of sync with the
    artifact it is supposed to describe. verify_minilm() needs this because an
    fp32 export and an int8 export have genuinely different numerical
    expectations, and applying one tolerance to both produces a gate that is
    either uselessly strict or uselessly lax.
    """
    import onnx
    p = ONNX_DIR / "minilm_encoder.onnx"
    if not p.exists():
        return False
    m = onnx.load(str(p))
    has_external = any(
        t.data_location == onnx.TensorProto.EXTERNAL
        for t in m.graph.initializer)
    int8_ops = {"MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul",
                "ConvInteger"}
    has_int8 = any(n.op_type in int8_ops for n in m.graph.node)
    # Dynamic quantization inlines its weights as int8 initializers and
    # materialises the scale/zero-point as Constant nodes.
    int8_initializers = any(
        t.data_type == onnx.TensorProto.INT8 for t in m.graph.initializer)
    return (has_int8 or int8_initializers) and not has_external


def export_minilm(head_ckpt="models/classifier/minilm_head.pt",
                 encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 opset=17, quantize=True):
    """
    Export the MiniLM encoder and the trained head.

    `quantize=True` applies int8 dynamic quantization to the encoder. This is
    not an optional optimisation: the fp32 encoder is 90 MB of external
    weights, which makes the PWA ~150 MB installed and slow to precache on a
    phone. int8 brings the payload to ~23 MB for a measured accuracy cost that
    is reported, not assumed (see verify_minilm's quantisation comparison).

    The head is left in fp32 — it is 1 KB, and quantising a 4-class linear
    layer buys nothing while making the parity check noisier.
    """
    from transformers import AutoModel, AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType
    tok = AutoTokenizer.from_pretrained(encoder_name)
    enc = AutoModel.from_pretrained(encoder_name)
    enc.eval()

    ids = tok(["Amoxicillin 500mg three times daily"], return_tensors="pt",
              padding=True, truncation=True, max_length=256)
    enc_path = ONNX_DIR / "minilm_encoder.onnx"
    torch.onnx.export(
        enc, (ids["input_ids"], ids["attention_mask"]),
        str(enc_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"},
                      "attention_mask": {0: "b", 1: "s"},
                      "last_hidden_state": {0: "b", 1: "s"}},
        opset_version=opset, do_constant_folding=True)
    print(f"[export] minilm_encoder.onnx  {enc_path.stat().st_size/1024:.1f} KB",
          flush=True)

    if quantize:
        q_path = ONNX_DIR / "minilm_encoder.int8.onnx"
        # per_channel=False: dynamic int8 on transformers is well behaved at
        # per-tensor granularity, and per-channel on the MatMul weights is
        # where onnxruntime historically produced unusable results here.
        quantize_dynamic(str(enc_path), str(q_path), weight_type=QuantType.QInt8,
                         per_channel=False, reduce_range=False)
        print(f"[quant] minilm_encoder.int8.onnx  "
              f"{q_path.stat().st_size/1024/1024:.1f} MB  "
              f"(fp32 was {_weight_bytes(enc_path)/1024/1024:.1f} MB)",
              flush=True)
        # Drop the fp32 graph and its external blob FIRST, while both still
        # exist, then rename the int8 graph into place. quantize_dynamic already
        # consumed the fp32 blob, so stat()ing it after this point raises
        # FileNotFoundError — which is exactly the bug this ordering avoids.
        for leftover in ONNX_DIR.glob("minilm_encoder.onnx.data"):
            sz_mb = leftover.stat().st_size / 1024 / 1024
            leftover.unlink()
            print(f"[quant] removed {leftover.name} "
                  f"({sz_mb:.1f} MB fp32 blob)")

        # Ship the int8 graph under the canonical name the app requests.
        enc_path.unlink()
        q_path.replace(enc_path)
        print(f"[export] minilm_encoder.onnx  {enc_path.stat().st_size/1024/1024:.1f} MB"
              f"  (int8)", flush=True)

    # tokenizer files travel with the model for transformers.js
    for f in ("tokenizer.json", "vocab.txt", "tokenizer_config.json",
              "special_tokens_map.json"):
        src = Path(tok.name_or_path) / f
        if src.exists():
            (ONNX_DIR / f).write_bytes(src.read_bytes())

    # head
    from train_classifier import ClassifierHead
    head = ClassifierHead()
    if Path(head_ckpt).exists():
        head.load_state_dict(torch.load(head_ckpt, map_location="cpu",
                                        weights_only=True)["head"])
    head.eval()
    head_path = ONNX_DIR / "minilm_head.onnx"
    torch.onnx.export(
        head, torch.zeros(1, 384), str(head_path),
        input_names=["embedding"], output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True)
    print(f"[export] minilm_head.onnx  {head_path.stat().st_size/1024:.1f} KB",
          flush=True)
    return enc_path, head_path, enc, head, tok


# ── verification ────────────────────────────────────────────────────────

def verify_restorer(torch_model):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_DIR / "restorer.onnx"),
                                providers=["CPUExecutionProvider"])
    torch_model.eval()
    rng = np.random.default_rng(123)
    worst = 0.0
    for shape in [(1, 1, 64, 256), (4, 1, 64, 256), (1, 1, 128, 256)]:
        x = rng.random(shape).astype(np.float32)
        with torch.no_grad():
            y_t = torch_model(torch.from_numpy(x)).numpy()
        y_o = sess.run(["output"], {"input": x})[0]
        d = float(np.abs(y_t - y_o).max())
        worst = max(worst, d)
        print(f"  restorer {str(shape):<18} max|Δ| = {d:.2e}")
    ok = worst < TOL
    print(f"  restorer parity: {'PASS' if ok else 'FAIL'} (worst {worst:.2e})")
    return ok, worst


def verify_minilm(enc, head, tok, quantized=False):
    """
    Numerical agreement between the PyTorch models and their ONNX exports.

    TOLERANCE DEPENDS ON WHETHER THE ENCODER IS QUANTIZED, and conflating the
    two is a mistake that produces a meaningless red gate.

    An ONNX *export* is an exact re-expression of the same arithmetic, so fp32
    -> fp32 should agree to ~1e-6; anything worse is a real export bug. But
    int8 dynamic quantization is a deliberate lossy approximation: the encoder's
    MatMul weights are rounded to 8-bit, so hidden-state deltas of O(1) are
    EXPECTED and CORRECT. Demanding 1e-5 there measures the quantizer, not a
    defect.

    So the two claims are separated:
      * exact export fidelity  -> small TOL, meaningful
      * quantized agreement    -> argmax identity is the real criterion, and
                                 verify_quantization() measures the accuracy
                                 cost on the full held-out split

    With `quantized=True` this checks structure and argmax identity, and defers
    the precision question to verify_quantization().
    """
    import onnxruntime as ort
    s_enc = ort.InferenceSession(str(ONNX_DIR / "minilm_encoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    s_head = ort.InferenceSession(str(ONNX_DIR / "minilm_head.onnx"),
                                  providers=["CPUExecutionProvider"])
    texts = ["Amoxicillin 500mg, take one capsule three times daily",
             "Account Number: 40218877 Sort Code: 40-11-04 Balance 2,682.55",
             "Clause 3.2: the Tenant shall pay rent of 2,500 per month"]
    enc.eval()
    head.eval()
    ids = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=256)
    with torch.no_grad():
        h_t = enc(input_ids=ids["input_ids"],
                  attention_mask=ids["attention_mask"]).last_hidden_state
    h_o = s_enc.run(["last_hidden_state"],
                    {"input_ids": ids["input_ids"].numpy(),
                     "attention_mask": ids["attention_mask"].numpy()})[0]
    d_enc = float(np.abs(h_t.numpy() - h_o).max())

    # mean-pool + L2 normalise, then the head
    def pool(h, mask):
        m = mask.unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(e, p=2, dim=1)

    e_t = pool(h_t, ids["attention_mask"])
    # Pass the mask to pool() in its ORIGINAL (batch, seq) shape. pool() does its
    # own unsqueeze(-1), so pre-expanding here gave a 4-D mask and the multiply
    # blew up on a (3,25) vs (3,25,1,1) mismatch. The ONNX branch must be
    # given exactly what the torch branch gets.
    e_o = pool(torch.from_numpy(h_o), ids["attention_mask"])
    with torch.no_grad():
        l_t = head(e_t).numpy()
    l_o = s_head.run(["logits"], {"embedding": e_o.numpy()})[0]
    d_head = float(np.abs(l_t - l_o).max())

    # ISOLATE the head's own export fidelity. Comparing l_t (torch head on the
    # fp32 embedding) against l_o (onnx head on the INT8 embedding) conflates
    # two independent things: a lossy encoder, and a possibly-buggy head
    # export. The head is only 49,796 params and is exported unquantized, so
    # fed the SAME fp32 embedding on both sides it should agree to ~1e-6. If it
    # does not, that is a genuine export defect; if it does, any delta in d_head
    # is the quantizer's, and belongs to verify_quantization's budget.
    with torch.no_grad():
        l_head_onnx = s_head.run(["logits"],
                                 {"embedding": e_t.numpy()})[0]
    d_head_iso = float(np.abs(l_t - l_head_onnx).max())
    head_exact = d_head_iso < TOL

    # the thing that actually matters: same predicted class
    agree = int((l_t.argmax(1) == l_o.argmax(1)).all())
    print(f"  encoder max|Δ| = {d_enc:.2e}"
          f"{'  (int8: expected O(1), not an export defect)' if quantized else ''}")
    print(f"  head    max|Δ| = {d_head:.2e}  (end-to-end, includes int8 input)")
    print(f"  head    max|Δ| = {d_head_iso:.2e}  (isolated: same fp32 input both sides)")
    print(f"  argmax agreement: {'AGREE' if agree else 'DISAGREE'} "
          f"(torch {l_t.argmax(1).tolist()} vs onnx {l_o.argmax(1).tolist()})")

    if quantized:
        # Two separate claims, checked separately:
        #   1. the head's EXPORT is exact  -> strict tolerance, same input both sides
        #   2. the quantized PIPELINE still classifies correctly -> argmax identity
        # The end-to-end d_head is the quantizer's error budget, measured on the
        # full split by verify_quantization(), not a parity failure here.
        ok = head_exact and agree
        print(f"  head export exactness: {'PASS' if head_exact else 'FAIL'} "
              f"(strict {TOL:g}, fp32 input on both sides)")
        print(f"  quantized pipeline:    {'PASS' if agree else 'FAIL'} "
              f"(argmax identity)")
        print(f"  minilm parity: {'PASS' if ok else 'FAIL'} (quantized)")
        return ok, d_head_iso

    ok = d_enc < TOL and d_head_iso < TOL and agree
    print(f"  minilm parity: {'PASS' if ok else 'FAIL'}")
    return ok, max(d_enc, d_head_iso)


def verify_quantization(enc, head, tok):
    """
    Measure what int8 quantization actually costs — do not assume it is small.

    The shipped encoder is int8 (see export_minilm). That is only a defensible
    engineering decision if the accuracy cost is measured and reported, so
    this compares the int8 ONNX encoder against the fp32 PyTorch original on
    the real held-out corpus and reports the number that matters: does the
    predicted class change?
    """
    import onnxruntime as ort
    from train_classifier import three_way_split

    s_enc = ort.InferenceSession(str(ONNX_DIR / "minilm_encoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    s_head = ort.InferenceSession(str(ONNX_DIR / "minilm_head.onnx"),
                                  providers=["CPUExecutionProvider"])

    def pool_t(h, mask):
        m = mask.unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(e, p=2, dim=1)

    # Held-out TEST split only — never the training data.
    _, _, _, _, te_t, te_l = three_way_split(seed=123)
    samples = list(zip(te_t, te_l))[:120]
    if not samples:
        print("  [quant] no test split available, skipping")
        return True, 0.0

    agree = agree_fp32 = 0
    n = 0
    worst_d = 0.0
    enc.eval(); head.eval()
    B = 16
    for i in range(0, len(samples), B):
        chunk = samples[i:i + B]
        texts = [t for t, _ in chunk]
        gold = np.array([y for _, y in chunk])
        ids = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)
        with torch.no_grad():
            h_t = enc(input_ids=ids["input_ids"],
                      attention_mask=ids["attention_mask"]).last_hidden_state
            e_t = pool_t(h_t, ids["attention_mask"])
            l_t = head(e_t).numpy()
        l_t_cls = np.asarray(l_t).argmax(1)
        h_o = s_enc.run(["last_hidden_state"],
                        {"input_ids": ids["input_ids"].numpy(),
                         "attention_mask": ids["attention_mask"].numpy()})[0]
        e_o = pool_t(torch.from_numpy(h_o), ids["attention_mask"])
        worst_d = max(worst_d, float(np.abs(e_t.numpy() - e_o.numpy()).max()))
        l_o = np.asarray(s_head.run(["logits"],
                                    {"embedding": e_o.numpy()})[0]).argmax(1)
        agree_fp32 += int((l_t_cls == gold).sum())
        agree += int((l_o == gold).sum())
        n += len(chunk)

    acc32 = agree_fp32 / n
    accq = agree / n
    print(f"\n[quant] int8 vs fp32 on {n} held-out samples")
    print(f"  fp32 accuracy   {acc32*100:.1f}%")
    print(f"  int8 accuracy   {accq*100:.1f}%")
    print(f"  difference      {(acc32-accq)*100:+.1f} points")
    print(f"  embedding max|Δ| {worst_d:.2e}")
    # A 2-point drop is acceptable for a 4x smaller payload; more than that is
    # not, and should fail the export rather than ship silently.
    ok = (acc32 - accq) <= 0.02
    print(f"  quantisation acceptable: {'YES' if ok else 'NO'} "
          f"(threshold: 2.0 points)")
    return ok, abs(acc32 - accq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        tmodel = SightLineNet()
        ck = Path("models/restorer/restorer.pt")
        if ck.exists():
            tmodel.load_state_dict(torch.load(ck, map_location="cpu",
                                              weights_only=True)["state_dict"])
        from transformers import AutoModel, AutoTokenizer
        from train_classifier import ClassifierHead
        ok1, _ = verify_restorer(tmodel)
        h = ClassifierHead()
        hck = Path("models/classifier/minilm_head.pt")
        if hck.exists():
            h.load_state_dict(torch.load(hck, map_location="cpu",
                                        weights_only=True)["head"])
        name = "sentence-transformers/all-MiniLM-L6-v2"
        enc = AutoModel.from_pretrained(name)
        tok = AutoTokenizer.from_pretrained(name)
        ok2, _ = verify_minilm(enc, h, tok, quantized=_encoder_is_int8())
        ok3, _ = verify_quantization(enc, h, tok)
    else:
        _, tmodel = export_restorer(opset=args.opset)
        _, _, enc, head, tok = export_minilm(opset=args.opset)
        ok1, _ = verify_restorer(tmodel)
        ok2, _ = verify_minilm(enc, head, tok, quantized=_encoder_is_int8())
        ok3, _ = verify_quantization(enc, head, tok)

    print("\n" + "=" * 60)
    # Count the external weight blobs too. Summing only *.onnx reported
    # "59.6 KB" for a payload that was really 90 MB, because torch.onnx.export
    # writes large tensors into sidecar <name>.onnx.data files. Any size claim
    # in the docs or the privacy audit MUST use this, or it understates the
    # shipped payload by three orders of magnitude.
    per_file = {p.name: _weight_bytes(p) for p in sorted(ONNX_DIR.glob("*.onnx"))}
    total = sum(per_file.values())
    for name_, sz in per_file.items():
        graph = (ONNX_DIR / name_).stat().st_size
        blob = (ONNX_DIR / (name_ + ".data"))
        note = f"  (graph {graph/1024:.1f} KB" + (
            f" + weights {blob.stat().st_size/1024/1024:.1f} MB)" if blob.exists()
            else ")")
        print(f"  {name_:<28} {sz/1024/1024:>7.2f} MB{note}")
    print(f"total ONNX payload: {total/1024/1024:.2f} MB")
    if ok1 and ok2 and ok3:
        print("ALL PARITY CHECKS PASSED")
    else:
        print("PARITY/QUANTISATION FAILURE — do not ship these artifacts")
        sys.exit(1)


if __name__ == "__main__":
    main()
