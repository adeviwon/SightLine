# ECC Code Review — 2026-10-17

Run of the [ECC](https://github.com/affaan-m/ECC) review harness (`/code-review`
local criteria + `/quality-gate` + language review checklists) over the shipped
product: `app/` (PWA), `app/tools/` (dev tooling), `ml/src/` (training/eval).

## Gates at review time

| gate | result |
|---|---|
| `bash run.sh test` | **271 passed, 8 skipped** (0 fail). Skips are dataset-dependent — `data/` was not fetched on the review box, matching the documented skip behaviour. |
| `bash run.sh bundle-check` | **PASSED** — every shipped model loads and runs (26.68 MB payload). |
| `origin/gh-pages` vs `main:app/` | **byte-identical** (blob-hash comparison over the full tree) — the live install serves exactly what `main` ships. |

## Findings

### CRITICAL — none

### HIGH — none in shipped code

The scan flagged 25 `eval()` / 4 `new Function` hits. Every one is either
inside vendored minified libraries (`app/vendor/ort/ort.min.js`,
`app/vendor/pdfjs/pdf.*.js`) or in dev tooling that drives Chrome via CDP
(`app/tools/*` — `cdp.eval` is the tool's own API, not runtime `eval`).
None ship to users in a path that executes attacker-controlled input.

Zero `innerHTML` assignments in first-party JS — every DOM write goes through
`textContent` / `createElement` (XSS-safe by construction). No hardcoded
credentials, no `document.write`, all `fetch()` calls are same-origin static
assets with `cache: "force-cache"` (consistent with the zero-network claim).

### MEDIUM — noted, not changed

- `ml/src/`: several `torch.load(..., weights_only=False)` call sites
  (`build_handheld_consensus.py`, `compare_checkpoints.py`, `eval_ocr.py`,
  `export_onnx.py`). They load the project's own locally-trained checkpoints —
  no network input reaches them — so the unsafe-deserialization risk is
  contained to a developer running repo code. Leave as-is unless checkpoints
  ever come from outside.

### LOW — noted, not changed

- `app/js/pipeline.js` is 992 lines (ECC soft threshold 800). It is cohesive
  (one pipeline); splitting it is churn without a defect.
- `console.log` in `app/tools/*` is expected — those are CLI dev tools.

## Verdict

**Pass.** No CRITICAL or HIGH findings block the ship. The product's own gates
(271 tests, bundle-check, byte-identical deploy) were re-verified during this
review on a fresh environment.
