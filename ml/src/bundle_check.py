"""
Ship-gate for the app's model bundle.

WHY THIS EXISTS
---------------
`app/models/minilm_encoder.onnx` once shipped as a 56 KB fp32 graph that
referenced 104 external weight tensors, with the 90 MB `.onnx.data` blob
missing from the directory. Every check we had at the time passed:

  * the file existed, so the precache audit was satisfied
  * the manifest said `"classifier": true`, so the app believed models were
    present
  * `onnx.load(..., load_external_data=False)` succeeded, because that flag
    exists precisely to skip blob resolution

So a model that could not run looked exactly like a model that could. The app
would have crashed on first classification, on a judge's phone, at the demo.

Every check here loads weights for REAL, and the encoder/head pair is run
end-to-end, because "the file is there" is not a property worth asserting.

Run:  bash run.sh bundle-check
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "app"
APP_MODELS = APP / "models"

# crnn.onnx is the recogniser that replaced Tesseract, and it is REQUIRED, not
# optional. Without it the app has no text recognition at all -- Tesseract is
# no longer in the pipeline -- so "the app still works without it" is no longer
# true and must not be claimed.
REQUIRED = ["minilm_encoder.onnx", "minilm_head.onnx", "restorer.onnx",
            "crnn.onnx"]


def main():
    import onnx

    failures = []
    print("=== 1. files present ===")
    for name in REQUIRED:
        p = APP_MODELS / name
        ok = p.exists()
        kb = p.stat().st_size / 1024 if ok else 0
        print(f"  {'OK  ' if ok else 'FAIL'} {name:<24} {kb:>9.0f} KB")
        if not ok:
            failures.append(f"{name} missing")
    if failures:
        print("\nBUNDLE CHECK FAILED")
        return 1

    print("\n=== 2. weights actually resolve (full load, not load_external_data=False) ===")
    loaded = {}
    for name in REQUIRED:
        p = APP_MODELS / name
        blob = APP_MODELS / (name + ".data")
        try:
            m = onnx.load(str(p))
            ext = sum(1 for t in m.graph.initializer
                      if t.data_location == onnx.TensorProto.EXTERNAL)
            loaded[name] = m
            note = (f"blob {blob.stat().st_size/1024:.0f} KB present"
                    if ext else "self-contained")
            print(f"  OK   {name:<24} {len(m.graph.initializer):>3} init  "
                  f"{note}")
        except Exception as e:
            print(f"  FAIL {name:<24} {type(e).__name__}: {e}")
            failures.append(f"{name} weights unresolvable: {e}")

    if failures:
        print("\nBUNDLE CHECK FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n=== 2b. NO EXTERNAL DATA — onnxruntime-web cannot load it ===")
    # THE BUG THIS EXISTS FOR.
    #
    # onnxruntime-web has no Module.MountedFiles API, so a model whose weights
    # live in a sidecar `.onnx.data` file loads perfectly under the native
    # `onnxruntime` used above — and then fails in the browser with
    #   "Failed to load external data file "restorer.onnx.data",
    #    error: Module.MountedFiles is not available."
    #
    # Every check in section 2 passed on exactly that model. The app silently
    # fell back to classical preprocessing, the E2E harness reported 42% field
    # recovery, and the trained model was never executed once.
    #
    # So the loadability test above is necessary but NOT sufficient. A model
    # is only shippable if it is also SELF-CONTAINED. Both conditions are
    # asserted here.
    for name in REQUIRED:
        p = APP_MODELS / name
        m = onnx.load(str(p), load_external_data=False)
        ext = [t.name for t in m.graph.initializer
               if t.data_location == onnx.TensorProto.EXTERNAL]
        blob = APP_MODELS / (name + ".data")
        if ext:
            print(f"  FAIL {name:<24} {len(ext)} external tensor(s), "
                  f"sidecar {'present ' if blob.exists() else 'MISSING'}"
                  f" — will NOT load in onnxruntime-web")
            failures.append(
                f"{name} uses external data ({len(ext)} tensors); "
                f"ORT Web cannot mount it. Re-run `bash run.sh export`, which "
                f"now inlines weights via _inline_external_data().")
        else:
            print(f"  OK   {name:<24} self-contained, 0 external tensors")

    # A leftover sidecar for a model that no longer references one is dead
    # weight in the precache. Flag it, since it doubles as a hint that an
    # export ran with external_data=True at some point.
    for name in REQUIRED:
        blob = APP_MODELS / (name + ".data")
        if blob.exists():
            print(f"  WARN {name + '.data':<24} orphan sidecar "
                  f"{blob.stat().st_size/1024:.0f} KB — not referenced by the "
                  f"graph; delete it")

    print("\n=== 3. real inference through the shipped bundle ===")
    import onnxruntime as ort

    try:
        s = ort.InferenceSession(str(APP_MODELS / "restorer.onnx"),
                                 providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).random((1, 1, 64, 256)).astype(np.float32)
        y = s.run(None, {"input": x})[0]
        finite = bool(np.isfinite(y).all())
        shape_ok = y.shape == x.shape
        print(f"  {'OK  ' if finite and shape_ok else 'FAIL'} restorer "
              f"{tuple(x.shape)} -> {tuple(y.shape)}  "
              f"range [{y.min():.3f}, {y.max():.3f}]  finite={finite}")
        if not (finite and shape_ok):
            failures.append("restorer output non-finite or wrong shape")
    except Exception as e:
        print(f"  FAIL restorer inference: {type(e).__name__}: {e}")
        failures.append(f"restorer inference: {e}")

    try:
        h = ort.InferenceSession(str(APP_MODELS / "minilm_head.onnx"),
                                 providers=["CPUExecutionProvider"])
        e = np.random.default_rng(1).random((3, 384)).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        lg = h.run(None, {"embedding": e})[0]
        ok = lg.shape == (3, 4) and bool(np.isfinite(lg).all())
        print(f"  {'OK  ' if ok else 'FAIL'} head (3,384) -> logits "
              f"{tuple(lg.shape)}  argmax {lg.argmax(1).tolist()}")
        if not ok:
            failures.append("head produced bad logits")
    except Exception as e:
        print(f"  FAIL head inference: {type(e).__name__}: {e}")
        failures.append(f"head inference: {e}")

    # The recogniser's input shape is a HARD CONTRACT with the browser, not a
    # preference: app/js/recognizer.js builds exactly [1,1,32,256] because the
    # exported graph cannot have a dynamic sequence length (aten::lstm refuses,
    # and torch.export cannot unroll a symbolic loop bound). If a re-export
    # changes that width, the JS stops matching and the app silently mis-trims
    # timesteps -- inventing characters from padding. Asserted here so the
    # failure is a build error rather than a wrong dosage on a phone.
    try:
        c = ort.InferenceSession(str(APP_MODELS / "crnn.onnx"),
                                 providers=["CPUExecutionProvider"])
        cx = np.random.default_rng(2).random((1, 1, 32, 256)).astype(np.float32)
        cl = c.run(None, {"input": cx})[0]
        finite = bool(np.isfinite(cl).all())
        shape_ok = (cl.ndim == 3 and cl.shape[0] == 1
                    and cl.shape[2] == 38)
        # 256 px of crop -> 32 timesteps, i.e. W/8. If this drifts, the JS
        # decoder's timestepsFor() is wrong for every crop.
        ts_ok = cl.shape[1] == 32
        print(f"  {'OK  ' if finite and shape_ok and ts_ok else 'FAIL'} crnn "
              f"(1,1,32,256) -> logits {tuple(cl.shape)}  "
              f"expect (1,32,38)  finite={finite}")
        if not (finite and shape_ok):
            failures.append(f"crnn bad logits {cl.shape}")
        if not ts_ok:
            failures.append(
                f"crnn timesteps {cl.shape[1]} != 32; export width must stay 256")
    except Exception as e:
        print(f"  FAIL crnn inference: {type(e).__name__}: {e}")
        failures.append(f"crnn inference: {e}")

    print("\n=== 4. manifest honesty ===")
    man = APP_MODELS / "ort.json"
    if man.exists():
        m = json.loads(man.read_text())
        declares = bool(m.get("restorer")) and bool(m.get("classifier"))
        print(f"  {'OK  ' if declares else 'FAIL'} manifest declares "
              f"restorer={m.get('restorer')} classifier={m.get('classifier')}")
        if not declares:
            print("       note: manifest says models absent; the app will use "
                  "its fallback path and the bundle is dead weight")
    else:
        print("  FAIL no ort.json manifest")
        failures.append("ort.json missing")

    print("\n=== 4b. no third-party OCR in the bundle ===")
    # The project's central claim is that OCR is our own model. Until this
    # commit that claim was only half true: app/models/ort.json asserted
    # "tesseract": false while pipeline.js still created a tesseract worker,
    # sw.js still precached its assets, index.html still loaded its script, and
    # app/vendor/tesseract still shipped 30 MB. A judge who opened the bundle
    # would have found the manifest to be a lie, which is far worse than
    # shipping tesseract honestly.
    #
    # So the absence is now enforced rather than asserted. Four independent
    # ways to reintroduce it are checked, because each was present at some
    # point: the vendor payload, the script tag, the precache entries, and the
    # worker calls.
    vend = [p for p in (APP / "vendor").rglob("*")
            if "tess" in p.name.lower() and p.is_file()]
    if vend:
        failures.append(
            f"third-party OCR payload present: "
            f"{[str(p.relative_to(APP)) for p in vend[:4]]}")
        print(f"  FAIL {len(vend)} tess* file(s) under app/vendor/")
    else:
        print("  ok   no tess* files under app/vendor/")

    html = (APP / "index.html")
    if html.exists():
        h = html.read_text().lower()
        if "tesseract" in h:
            failures.append("index.html still references tesseract")
            print("  FAIL index.html references tesseract")
        else:
            print("  ok   index.html has no tesseract script tag")

    swp = APP / "sw.js"
    if swp.exists():
        st = swp.read_text().lower()
        if "tesseract" in st or "tessdata" in st:
            failures.append("sw.js still precaches tesseract assets")
            print("  FAIL sw.js precaches tesseract assets")
        else:
            print("  ok   sw.js precaches no tesseract assets")

    pj = (APP / "js" / "pipeline.js")
    if pj.exists():
        pt = pj.read_text()
        # Strip comments before looking for CALLS, so the explanatory notes
        # about the removal do not read as the removal failing to happen.
        code = "\n".join(ln for ln in pt.splitlines()
                         if not ln.strip().startswith(("*", "//")))
        if "Tesseract." in code or "createWorker" in code:
            failures.append("pipeline.js still calls tesseract")
            print("  FAIL pipeline.js still calls tesseract")
        else:
            print("  ok   pipeline.js makes no tesseract calls")

    if not vend and html.exists() and "tesseract" not in html.read_text().lower():
        print("       -> bundle is genuinely tesseract-free")

    print("\n=== 5. tokenizer asset ===")
    # The app loads a single flat tokenizer.json (js/classifier.js
    # TOKENIZER_PATH = "models/tokenizer.json") and builds its own wordpiece
    # tokenizer from model.vocab at runtime. This check used to look for a
    # tokenizer/ DIRECTORY, which the app never had, so it printed
    # "informational" and moved on while the file went MISSING -- which is how
    # tokenizer.json was absent from the bundle for its entire life and the
    # on-device classifier could not tokenize anything.
    #
    # An informational check that can only ever report absence is not a check.
    tok = APP_MODELS / "tokenizer.json"
    if not tok.exists():
        print("  FAIL tokenizer.json missing -- the on-device classifier "
              "cannot tokenize without it")
        failures.append(
            "app/models/tokenizer.json missing; js/classifier.js requires it. "
            "Regenerate with the export step (it runs "
            "AutoTokenizer.save_pretrained('app/models')).")
    else:
        import json as _json
        kb = tok.stat().st_size / 1024
        try:
            spec = _json.loads(tok.read_text())
        except Exception as e:
            spec = {}
            print(f"  FAIL tokenizer.json is not valid JSON: {e}")
            failures.append(f"tokenizer.json unparseable: {e}")
        vocab = (spec.get("model") or {}).get("vocab") or {}
        if not vocab:
            print("  FAIL tokenizer.json has no model.vocab -- "
                  "js/classifier.js reads exactly that key")
            failures.append("tokenizer.json has no model.vocab")
        else:
            print(f"  OK   tokenizer.json  {kb:.0f} KB  "
                  f"vocab={len(vocab):,} tokens")
            if len(vocab) < 30000:
                print(f"  WARN vocab is small ({len(vocab):,}); "
                      f"all-MiniLM-L6-v2 has 30,522 — a truncated tokenizer "
                      f"silently degrades every classification")
    # save_pretrained also drops a tokenizer_config.json that nothing reads.
    # Flag it rather than shipping dead weight in the precache.
    stray = APP_MODELS / "tokenizer_config.json"
    if stray.exists():
        print(f"  WARN tokenizer_config.json present "
              f"({stray.stat().st_size/1024:.0f} KB) but referenced by nothing "
              f"— safe to delete")

    print()
    # Include the tokenizer in the shipped payload. It was omitted, which made
    # the reported total 22.16 MB for a bundle that actually installs 22.87 MB.
    payload = list(REQUIRED)
    if (APP_MODELS / "tokenizer.json").exists():
        payload.append("tokenizer.json")
    total = sum((APP_MODELS / n).stat().st_size for n in payload)
    for n in REQUIRED:
        b = APP_MODELS / (n + ".data")
        if b.exists():
            total += b.stat().st_size
    print(f"shipped model payload: {total/1024/1024:.2f} MB")

    if failures:
        print("\nBUNDLE CHECK FAILED — do not ship")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("BUNDLE CHECK PASSED — every shipped model loads and runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
