# ML pipeline — how each model is trained

Every command below is copy-pasteable. Setup once:

```bash
bash setup_env.sh          # torch, opencv, onnx, onnxruntime
bash install_st.sh         # sentence-transformers, transformers
```

Then every entry point goes through the venv wrapper, which strips leaked
`PYTHONPATH` / `VIRTUAL_ENV` / `PYTHONHOME`:

```bash
bash run.sh <command>
```

---

## 0. Order of operations

```bash
# 1. fetch the real-document corpora (~500 MB, one time)
bash run.sh fetch-data

# 2. build the recogniser's training set (~2 min, writes a 90 MB cache)
bash run.sh build-data --receipts 260

# 3. train the CRNN recogniser   (~45 min on 4 CPU cores)
bash run.sh train-ocr 45

# 4. train the restorer          (~22 min on 4 CPU cores)
bash run.sh train-dncnn 40

# 5. train the MiniLM head       (~4 min)
bash run.sh train-clf 60

# 6. export everything to self-contained ONNX
bash run.sh export

# 7. measure, honestly
bash run.sh eval-ocr
bash run.sh eval
bash run.sh results

# 8. ship gates — all must pass
bash run.sh repro-check
bash run.sh bundle-check
bash run.sh privacy
bash run.sh test
```

---

## 1. Real-document data (`fetch-data`, `build-data`)

### Why real photographs

The synthetic renderer (`capture.py`) is still used, and still useful: when you
know exactly what damage was applied, you can test the degradation gate against
a known answer. It is the wrong distribution for **training a recogniser**,
because the gap is in the imaging process, not the degradation model.

| | synthetic | real |
|---|---|---|
| ink | uniform black | thermal, fades across the page |
| paper | flat `#f8f8f8` | pink cast, uneven |
| extras | none | barcode, handwriting, ghost print |
| strokes | one clean font | thin, small, JPEG-ringing |

### Sources

```bash
# ICDAR2019-SROIE: 626 train + 347 test real scanned receipts,
# per-word boxes AND transcriptions. Via the HF mirror.
#   Huang et al., ICDAR 2019 Robust Reading Competition
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"):
    hf_hub_download("jsdnrs/ICDAR2019-SROIE", f, repo_type="dataset",
                    local_dir="data/sroie")
PY

# Express Expense: 198 HAND-PHOTOGRAPHED receipts + 4-corner homographies.
# CC-BY-4.0. Auad et al., SIBGRAPI 2024.
curl -L -o data/handheld/Dataset.zip \
  "https://zenodo.org/api/records/13688441/files/Dataset.zip/content"
cd data/handheld && unzip -q Dataset.zip
```

`bash run.sh fetch-data` does both.

### Building the cache

```bash
bash run.sh build-data --receipts 260
```

Output:

```
crops     30021          # 260 receipts x ~40 words x 3 variants
width     max 320px, height 32px
train     20922 crops   182 receipts
val        4548 crops    39 receipts
test       4551 crops    39 receipts
variants: clean 12233, composite 2354, motion 2251, dark 2227,
          jpeg 2222, defocus 2220, bright 2213, shadow 2160, lowres 2141
```

**The split is by receipt, not by crop.** A receipt contributes ~40 word crops,
so splitting crops would put the same receipt in both train and test and report
memorisation as generalisation.

### Augmentation

Nine variants, each mapped to an observed capture failure:

| variant | models |
|---|---|
| `clean` | the crop as photographed |
| `jpeg` | quality 25–60: WhatsApp / Instagram re-compression |
| `motion` | directional 3–9 px: hand shake |
| `defocus` | circular 1–3 px: lens not focused at arm's length |
| `dark` | ×0.35–0.7 + gamma: dim room, night capture |
| `bright` | ×1.3–1.9: glare / direct light |
| `lowres` | downscale then upscale: small photo enlarged by the OS |
| `shadow` | one-sided gradient: hand shadow across the page |
| `composite` | 2–3 stacked: what a phone actually produces |

`composite` matters most. Real captures are never singly degraded, and a model
trained only on single degradations fails on the combination — which is the only
case that occurs in the field.

Augmentation runs **once, at build time**, deterministically via `seedutil`.
Doing it per-epoch costs real CPU; doing it once makes the dataset
byte-identical across runs, which is what lets `repro-check` mean anything.

### Crop geometry

Height fixed at 32 px (CTC needs a constant time axis), width preserved
(16–320 px) because a 3-character price and a 30-character company name have
very different aspect ratios, and stretching both to the same width destroys the
glyph proportions the model must read.

Boxes are padded 2 px: tight boxes clip ascenders and descenders, and a clipped
`5` is a label the model cannot learn.

---

## 2. The CRNN recogniser (`train-ocr`)

### Architecture

```
[B,1,32,W] → 7 conv blocks → [B,128,1,W/8] → BiLSTM(2×128, bidir)
           → [B,W/8,256] → Linear → [B,W/8,38] → CTC greedy decode
```

Two things about the conv stack were measured rather than reasoned about, and
both were wrong on the first attempt.

**Height must reach exactly 1 row.** The LSTM reads one row; that row is the
time axis. `stride=2` + `pool=2` reduces an axis by **4**, not 2 (conv halves it,
then pool halves it again), which walks 32 → 8 → 2 → 0 and raises
`Calculated output size: (96x0x2)`. The working config decouples the axes: five
height-only pools (`pool=(2,1)`) take 32 → 1, and the last block pools width
only (`pool=(1,2)`) because a height-spanning pool applied after height is 1
gets 0. `forward()` asserts the height.

**Channel widths were profiled.** The first run took 335 s/epoch, and the profile
inverted the expectation — the conv stack was 87% of the step, not the LSTM:

| config | W=320 conv fwd+bwd | conv params |
|---|---|---|
| 256 ch | 1677 ms | 1,089,632 |
| **128 ch (chosen)** | **779 ms** | **272,944** |
| 96 ch | 762 ms | 167,904 |
| 64 ch | 896 ms | 96,320 |

Total model **942,166 params**, 2.2× faster.

### The charset is 37 symbols

10 digits + 26 letters + space, and **no punctuation**. Punctuation is the
highest-error class in practice and contributes nothing to any field this app
extracts. The visible consequence: `9.00` is read as `900`, which is safe here
because `TOTAL 900` and `TOTAL 9.00` match the same field regex. This is written
down so nobody discovers it by accident.

### CTC feasibility

CTC needs `T >= label_length`, where `T = W/8`. Over the 30,021 crops:

```
median 2.16 chars/timestep,  p5 1.48,  only 0.4% below the 1.0 floor
```

`train_ocr.py` **drops** the infeasible samples and reports how many (30, or
0.10%) rather than dropping them silently. `zero_infinity=True` is set as well:
the filter is belt, that is braces.

### Training details

| Setting | Value | Why |
|---|---|---|
| loss | CTC, `zero_infinity=True` | no alignment labels needed |
| optimiser | AdamW, lr 1e-3, wd 1e-4 | |
| schedule | OneCycleLR, 25% warmup | |
| batching | width-bucketed, tol 24 px | a batch must be rectangular |
| grad clip | 5.0 | CTC loss is spiky before blanks are learned |
| **selection** | **validation word accuracy** | loss is dominated by easy crops |

Per-sample `input_lengths` are mandatory. Crops share a padded width, but each
crop's true timestep count depends on its *own* width — passing the batch width
tells CTC that padding is real signal, and it will align characters into it.

Per-variant metrics accumulate `(edits, chars, exact, n)` and divide at the
**end**. Dividing per sample and averaging weights a 3-character crop the same
as a 28-character one, which flatters CER and hides exactly the long labels
(prices, account numbers) that carry the risk.

```bash
bash run.sh train-ocr 45
```

Checkpoints to `models/ocr/crnn.pt`, history to `artifacts/ocr_training.json`.

### Performance note

`timesteps()` runs a real conv forward, and the training loop needs it twice per
sample. Over 30,021 crops that is 60,042 forwards, which pushed the dataset load
from 20 s to over 4 minutes. There are only **299 distinct crop widths**, so it
is memoised on width: the feasibility filter now takes 2.7 s.

---

## 3. Model 1 — SightLineNet (document restoration)

### The problem

The user holds a phone ~30 cm above a document, one-handed, with limited vision.
That physical situation produces a characteristic damage pattern that classical
filters handle badly:

- **Motion blur.** A shaky hand produces *directional* smear, not isotropic
  Gaussian smoothing. A median filter blurs it further; an unsharp mask
  amplifies the smear along with the edges.
- **Defocus.** At close range the lens produces a circular bokeh disc, which
  is not the Gaussian kernel every tutorial uses.
- **Sensor noise.** Poisson-Gaussian, and it scales with ISO — so a dim room
  produces noise *and* low contrast simultaneously.
- **Hand shadow.** A hard-edged shadow from the hand holding the phone, in
  the exact region of the page being read.

A classical stack applies one *fixed* inverse regardless of which of these
actually occurred. SightLineNet learns the inverse of the measured family.

### The architecture

```
Input   1 × 64 × 256   grayscale float32 in [0,1]
        ↓
Conv3x3(1→32) + ReLU
        ↓
Conv3x3(32→32) + GroupNorm + ReLU     ┐
Conv3x3(32→32) + GroupNorm + ReLU     │
Conv3x3(32→32) + GroupNorm + ReLU     ├  5 blocks
Conv3x3(32→32) + GroupNorm + ReLU     │
Conv3x3(32→32) + GroupNorm + ReLU     ┘
        ↓
Conv3x3(32→32) + ReLU
Conv3x3(32→1)
        ↓
Output = input + body(input)          ← residual
```

**37,793 parameters.** Small enough to train in minutes on a laptop CPU and
to run in a browser on a phone.

Architecture selection was measured, not assumed:

| config | params | s/epoch |
|---|---|---|
| 48 ch / 8 blocks | 126,097 | 124.1 |
| **32 ch / 6 blocks (chosen)** | **37,793** | **51.9** |
| 24 ch / 6 blocks | 21,433 | 41.7 |

### Three design decisions worth defending

**1. Residual learning with zero-init output.** The final convolution's weights
and bias are initialised to zero, so at initialisation the network is *exactly*
the identity function — `forward(x) == x` bit-for-bit. Training starts from "do
nothing" and learns to add only the correction that helps. A network predicting
the image directly must learn the identity from scratch and trains markedly
worse on a small corpus. Asserted as a test invariant in `test_model.py`.

**2. Restoration, not denoising.** A classical DnCNN can only remove additive
noise. It *cannot* invert a blur that destroyed stroke detail — the information
is gone. SightLineNet is trained against the clean render, so it learns to
sharpen and denoise jointly.

**3. Charbonnier loss, not MSE.** `sqrt(a² + ε²)` behaves like smooth-L1 and is
far more robust to the heavy-tailed residuals that severe degradation produces.
MSE lets a handful of catastrophic patches dominate each batch's gradient.

### Training data

Rendered documents paired with degraded versions across all ten capture
profiles. **Leakage control:** the train/val split is by **document seed**, not
by patch. Every patch from one document lands entirely on one side.

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, wd 1e-5 |
| Loss | Charbonnier |
| Grad clip | 1.0 |
| Batch | 16 |
| Epochs | 40 |
| Checkpoint | best val PSNR |
| **best val PSNR** | **14.13 dB** |

---

## 4. The degradation gate (`gating`)

Not a trained model — a measured decision rule, kept honest by a cross-language
parity test.

It computes the variance of a Laplacian over `64×256` **tiles** and returns the
**median** tile weight:

```
SHARP_CLEAN = 0.080     SHARP_BAD = 0.008     CONTENT_MIN = 0.12
```

Median, not mean, because blank margins must not dilute a blurred page and one
noisy tile must not force restoration of a clean one.

**This gate is mandatory.** Applied unconditionally the restorer costs
**−55.76 dB** on undamaged patches:

| patches | identity | gated | effect |
|---|---|---|---|
| undamaged (n=71) | −55.76 dB | **+0.00 dB** | clean scans untouched |
| mildly damaged (n=184) | +1.25 dB | ~0.00 dB | neutral |
| badly damaged (n=165) | +5.81 dB | **+4.69 dB** | where the work is |

```bash
bash run.sh gating               # calibrate + report, Python side
node app/tools/gating_parity.js  # 30/30 agreement with the JS port
```

`test_gating.py` carries a **mutation test**: replacing the tiled gate with a
whole-image gate must fail the scale regression and four clean-profile tests.
It does.

---

## 5. Model 2 — MiniLM document classifier

```
Document text
      ↓
sentence-transformers/all-MiniLM-L6-v2      FROZEN
   6 layers · 22,713,216 params · 384-dim output
   mean pooling over the attention mask, then L2 normalise
      ↓
ClassifierHead                             TRAINED
   Linear(384→128) → ReLU → Dropout(0.1) → Linear(128→4)
      ↓
banking │ medical │ legal │ general
```

**49,796 trainable parameters out of 22.7 million.** Only the head learns.

> **MiniLM is a text encoder. It cannot read pixels.** It consumes the output of
> the recogniser. This is not a workaround — it is the reason the CRNN exists.

### Why the encoder is frozen

1. **No catastrophic forgetting.** MiniLM was pretrained on ~1 billion sentence
   pairs. Training 49K head parameters cannot degrade that.
2. **Speed.** The whole run is ~4 minutes on CPU. There is no GPU in the budget.
3. **Size.** The frozen encoder quantises to 22.16 MB for the browser.

### Corpus

`ml/src/corpus.py` holds **174 unique document templates** across 4 categories
(banking 45, medical 50, legal 38, general 41).

**Deduplicated.** An earlier version contained literal duplicates across the
train/val boundary. That is label noise, and it inflated the reported score.
`test_corpus.py` asserts zero duplicates.

**Split three ways, before augmentation.** train 70% / val 15% / test 15%.
Reporting a val score that was also used to pick the checkpoint is optimistic.

**Augmentation is train-only.** Real OCR output has digit/letter confusion
(`0`↔`O`, `1`↔`l`, `5`↔`S`, `8`↔`B`), stray case damage, and random substitutions
inside digit runs. `ocr_noise()` simulates these, applied **after** the split.

### Metrics reported

- Held-out **test** accuracy with a bootstrap 95% CI
- Macro F1, per-category accuracy, full 4×4 confusion matrix
- Cross-validation mean ± std across 20 disjoint folds
- **5-seed spread**, because a single split lied

Result: **mean 95.0%**, range 91.7–100%, σ 3.1%, bootstrap CI [86.7–100%].

**Report the mean and the spread, never the 100% seed.** A single 100% split was
traced to split luck, not model quality.

---

## 6. ONNX export and parity

`ml/src/export_onnx.py` exports four graphs and then **proves** they compute the
same thing as PyTorch:

| Artifact | Input | Output | Parity criterion |
|---|---|---|---|
| `restorer.onnx` | `float32[b,1,H,W]` | `float32[b,1,H,W]` | max abs Δ ≤ 1e-5 |
| `crnn.onnx` | `float32[b,1,32,W]` | `float32[b,T,38]` | max abs Δ ≤ 1e-5 |
| `minilm_encoder.onnx` | `input_ids`, `attention_mask` | `last_hidden_state` | **26/26 held-out class agreement** (INT8) |
| `minilm_head.onnx` | `float32[b,384]` | `float32[b,4]` | max abs Δ ≤ 1e-5 |

Measured: restorer **4.32e-07**, head in isolation **8.94e-08**, encoder worst
embedding delta **2.64e-02** under INT8.

**Two criteria, not one.** Strict parity judges the FP32 exports. The encoder is
*intentionally* quantised, so it is judged on held-out class agreement instead.
A single threshold cannot judge both — an earlier version rejected the INT8
encoder for a delta its own design had introduced.

**Browser models must be self-contained.** ONNX Runtime Web cannot mount
Python-style external `.onnx.data` sidecars:

```
Failed to load external data file "restorer.onnx.data",
error: Module.MountedFiles is not available
```

`_inline_external_data()` inlines them, and `bundle-check` rejects any browser
model containing external tensor references.

---

## 7. Reproducing everything

```bash
bash setup_env.sh          # venv + deps (one time, ~2 min)
bash install_st.sh         # sentence-transformers
bash run.sh fetch-data     # real corpora (~500 MB, one time)
bash run.sh build-data     # build the OCR training cache
bash run.sh train-ocr 45   # recogniser
bash run.sh train-dncnn 40 # restorer
bash run.sh train-clf 60   # classifier
bash run.sh export         # ONNX + parity verification
bash run.sh eval-ocr       # real held-out receipts
bash run.sh eval           # end-to-end, synthetic corpus
bash run.sh test           # pytest suite
```

Every run is seeded via `ml/src/seedutil.py` (`zlib.crc32`, **not** Python's
salted `hash()`), and artifacts carry a seeder stamp so stale results are
labelled stale.

---

## 8. CPU budget

| step | time |
|---|---|
| `fetch-data` | ~5 min (500 MB) |
| `build-data` | ~2 min (one-time cache) |
| `train-ocr 45` | ~45 min |
| `train-dncnn 40` | ~22 min |
| `train-clf 60` | ~4 min |
| `export` | ~2 min |
| `eval-ocr` | ~6 min |
| `eval` | ~12 min |
| gates + tests | ~4 min |

Everything fits a laptop CPU and a hackathon afternoon.

---

## 9. Honest limitations

State these before a judge finds them.

1. **Field accuracy is not 95%.** The 95–100% bar is met by the *classifier*
   (95.0% mean over 5 seeds). End-to-end field extraction on the synthetic
   corpus is **62.2%**, and field extraction is all-or-nothing by design. These
   are different metrics and we do not conflate them.

2. **The classifier's corpus is still synthetic.** 174 rendered templates is not
   10,000 real photographs. The recogniser is trained on real receipts; the
   classifier is not yet.

3. **Blur recovery has a hard ceiling.** A restoration network can sharpen what
   remains and suppress what was added, but it cannot reconstruct a digit stroke
   the lens smeared away entirely. The app reports low confidence instead of
   guessing. That is the designed behaviour.

4. **The 198 hand-photographed receipts are held out for evaluation**, not yet
   folded into training, because their labels are whole-page transcripts rather
   than word boxes. Word-level crops from them would improve the recogniser.

5. **Restoration is patch-local.** A 64×256 tile cannot use global layout, so a
   strong shadow spanning several tiles is restored tile by tile.

6. **English and digits only.** The 37-symbol charset is Latin alphanumerics
   plus space. Adding a script means retraining the output layer.