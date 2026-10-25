"""Which checkpoint does app/models/crnn.onnx actually contain? §28 of the
validation-gates skill: identify the shipped artifact by OUTPUT, never by the
document that names it."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import build_dataset as B  # noqa: E402
import realdata as R  # noqa: E402
import render_corpus as RC  # noqa: E402
from ocr_model import build  # noqa: E402
from PIL import Image  # noqa: E402

data = RC.load(split="test")
n = 16
arrays = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
          for c in data["crops"][:n]]
refs = [str(x) for x in data["labels"][:n]]

import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(str(ROOT / "app" / "models" / "crnn.onnx"),
                            providers=["CPUExecutionProvider"])
onnx_out = []
for a in arrays:
    x = (np.asarray(a, np.float32)[None, None] / 255.0 - 0.5) / 0.5
    lp = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    onnx_out.append(R.decode_greedy(lp[0].argmax(-1)))

import torch  # noqa: E402
for ck in ["crnn.pt", "crnn_retrain3_best.pt", "crnn_retrain3_shipped_archive.pt",
           "crnn_before_render.pt"]:
    p = ROOT / "models" / "ocr" / ck
    if not p.exists():
        print("%-36s MISSING" % ck)
        continue
    c = torch.load(str(p), map_location="cpu", weights_only=False)
    m = build(c.get("num_classes", R.NUM_CLASSES))
    m.load_state_dict(c["model"])
    m.eval()
    outs = []
    for a in arrays:
        x = (np.asarray(a, np.float32)[None, None] / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            lg = m(torch.from_numpy(x))
        outs.append(R.decode_greedy(lg[0].argmax(-1).numpy()))
    same = sum(1 for a, b in zip(outs, onnx_out) if a == b)
    print("%-36s matches ONNX decode: %d/%d" % (ck, same, n))

print("\nfirst 3 ONNX reads:", onnx_out[:3])
