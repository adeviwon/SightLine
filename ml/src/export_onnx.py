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


def export_minilm(head_ckpt="models/classifier/minilm_head.pt",
                 encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 opset=17):
    from transformers import AutoModel, AutoTokenizer
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


def verify_minilm(enc, head, tok):
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
    ids_p = ids["attention_mask"].numpy().astype(np.float32)[:, :, None]
    e_o = pool(torch.from_numpy(h_o), torch.from_numpy(ids_p))
    with torch.no_grad():
        l_t = head(e_t).numpy()
    l_o = s_head.run(["logits"], {"embedding": e_o.numpy()})[0]
    d_head = float(np.abs(l_t - l_o).max())

    # the thing that actually matters: same predicted class
    agree = int((l_t.argmax(1) == l_o.argmax(1)).all())
    print(f"  encoder max|Δ| = {d_enc:.2e}")
    print(f"  head    max|Δ| = {d_head:.2e}")
    print(f"  argmax agreement: {'AGREE' if agree else 'DISAGREE'} "
          f"(torch {l_t.argmax(1).tolist()} vs onnx {l_o.argmax(1).tolist()})")
    ok = d_enc < TOL and d_head < TOL and agree
    print(f"  minilm parity: {'PASS' if ok else 'FAIL'}")
    return ok, max(d_enc, d_head)


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
        from transformers import AutoModel
        from train_classifier import ClassifierHead
        ok1, _ = verify_restorer(tmodel)
        h = ClassifierHead()
        hck = Path("models/classifier/minilm_head.pt")
        if hck.exists():
            h.load_state_dict(torch.load(hck, map_location="cpu",
                                        weights_only=True)["head"])
        ok2, _ = verify_minilm(AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"), h,
            AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").config and
            __import__("transformers").AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"))
    else:
        _, tmodel = export_restorer(opset=args.opset)
        _, _, enc, head, tok = export_minilm(opset=args.opset)
        ok1, _ = verify_restorer(tmodel)
        ok2, _ = verify_minilm(enc, head, tok)

    print("\n" + "=" * 60)
    total = sum(p.stat().st_size for p in ONNX_DIR.glob("*.onnx"))
    print(f"total ONNX payload: {total/1024:.1f} KB")
    if ok1 and ok2:
        print("ALL PARITY CHECKS PASSED")
    else:
        print("PARITY FAILURE — do not ship these artifacts")
        sys.exit(1)


if __name__ == "__main__":
    main()
