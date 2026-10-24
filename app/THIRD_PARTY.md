# PP-OCRv5 attribution

Enhanced reading uses PaddlePaddle PP-OCRv5 mobile detector and recognizer,
licensed under Apache License 2.0, exported by bukuroo/PPOCRv5-ONNX at commit
47b3e1b4e90c79737cb71f562a6c85809067c7a5. Both detector and recognition weights remain fp32, selected after a fixed-fixture
precision comparison. See models/ppocr-provenance.json.
Dictionary: PaddleOCR PP-OCRv5 character dictionary. The arithmetic helpers were
adapted from the Aria Reader PP-OCR integration, based on PaddleOCR's DB/CTC routines.
This model is pretrained third-party work, separate from SightLine's own CRNN.
The original CRNN, restorer and classifier are preserved. ONNX Runtime remains 1.20.1.

PaddleOCR source: https://github.com/PaddlePaddle/PaddleOCR
ONNX export source: https://huggingface.co/bukuroo/PPOCRv5-ONNX
Apache-2.0 license: https://www.apache.org/licenses/LICENSE-2.0

Aria Reader integration: `js/aria/fields.mjs` and `taxonomy.mjs` are copied from
local Aria Reader at base commit 273747a plus its existing safety fixes, with
module paths adjusted. `config.mjs` carries only the unchanged 0.75 safety floor.
`receipt.mjs` adapts that pure parser to SightLine; no Tesseract assets were added.
