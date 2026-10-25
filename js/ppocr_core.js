/** PP-OCRv5 mobile adapter. Model weights/dictionary: Apache-2.0; see THIRD_PARTY.md.
 * Runs inside a module worker in the app. Also accepts an injected runtime and
 * model bytes for genuine native WASM integration tests without an HTTP server.
 */
import { resizeLuma, toChw, DET_MEAN, DET_STD, REC_MEAN, REC_STD,
  detInputSize, dbPostProcess, groupIntoLines, splitIntoWords, cropResizeLuma,
  recWidthFor, ctcDecode, buildCharset } from './ppocr_math.js';

export function speechForDocument(lines, confidence, completeRead) {
  return completeRead && confidence >= .75 ? lines.join('\n') : '';
}

export class PaddleReader {
  constructor() { this.det = null; this.rec = null; this.charset = null; this.runtime = null; this.pending = null; }
  async prepare(options = {}) {
    if (this.det && this.rec) return this;
    if (this.pending) return this.pending;
    this.pending = this._prepare(options).finally(() => { this.pending = null; });
    return this.pending;
  }
  async _prepare(options) {
    let det, rec;
    try {
      const ort = options.runtime || await import('../vendor/ort/ort.min.mjs');
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = options.wasmPaths || new URL('../vendor/ort/', import.meta.url).href;
      const settings = { executionProviders: ['wasm'], graphOptimizationLevel: 'all',
        enableCpuMemArena: false, enableMemPattern: false };
      // Recognition crops have many widths. Avoid retaining a memory pattern
      // for each width and let detector/recognizer scratch allocations be reused.
      options.onProgress?.({ status: 'Getting ready to read.' });
      det = await ort.InferenceSession.create(options.detModel || new URL('../models/ppocrv5-mobile-det.onnx', import.meta.url).href, settings);
      rec = await ort.InferenceSession.create(options.recModel || new URL('../models/ppocrv5-mobile-rec.onnx', import.meta.url).href, settings);
      let dictionary = options.dictionary;
      if (dictionary === undefined) {
        const response = await fetch(new URL('../models/ppocrv5_dict.txt', import.meta.url));
        if (!response.ok) throw new Error('The reading files are not available.');
        dictionary = await response.text();
      }
      this.charset = buildCharset(dictionary);
      this.runtime = ort; this.det = det; this.rec = rec;
      return this;
    } catch (error) {
      await Promise.allSettled([det?.release(), rec?.release()]);
      throw error;
    }
  }
  async recognize(image, onProgress = () => {}) {
    if (!this.det || !this.rec) throw new Error('The reader is not ready.');
    const { data, width, height } = image || {};
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1
      || width * height > 16000000 || !data || data.length !== width * height) {
      throw new Error('This picture could not be read. Choose another photo.');
    }
    const started = performance.now(), ort = this.runtime;
    const size = detInputSize(width, height, 960);
    const input = new ort.Tensor('float32', toChw(resizeLuma(data, width, height, size.width, size.height), size.width, size.height, DET_MEAN, DET_STD), [1, 3, size.height, size.width]);
    let outputs;
    let lines;
    onProgress({ status: 'Finding the words on the page.' });
    try {
      outputs = await this.det.run({ [this.det.inputNames[0]]: input });
      const output = outputs[this.det.outputNames[0]];
      const [dh, dw] = output.dims.slice(-2);
      const boxes = dbPostProcess(output.data, dw, dh);
      lines = groupIntoLines(boxes.map(b => ({ x0: b.x0 * width / dw, x1: b.x1 * width / dw, y0: b.y0 * height / dh, y1: b.y1 * height / dh })));
    } finally {
      input.dispose();
      for (const value of Object.values(outputs || {})) value.dispose();
    }
    const words = [], texts = [], safeTexts = [];
    let confidenceSum = 0, completeRead = true;
    for (let li = 0; li < lines.length; li++) {
      const parts = [];
      let safeLine = true;
      for (const lineBox of lines[li]) {
        for (const box of splitIntoWords(data, width, height, lineBox)) {
          const rw = recWidthFor(box.x1 - box.x0, box.y1 - box.y0, 48, 2000);
          const crop = cropResizeLuma(data, width, height, box, rw, 48);
          if (!crop) { safeLine = false; continue; }
          const feed = new ort.Tensor('float32', toChw(crop, rw, 48, REC_MEAN, REC_STD), [1, 3, 48, rw]);
          let results, decoded;
          try {
            results = await this.rec.run({ [this.rec.inputNames[0]]: feed });
            const output = results[this.rec.outputNames[0]];
            if (output.dims[2] !== this.charset.length) throw new Error('The reading files do not match.');
            decoded = ctcDecode(output.data, output.dims[1], output.dims[2], this.charset);
          } finally {
            feed.dispose();
            for (const value of Object.values(results || {})) value.dispose();
          }
          if (decoded.confidence < .75) safeLine = false;
          if (decoded.confidence < .5 || !/[0-9A-Za-z\u3400-\u9fff]/.test(decoded.text)) continue;
          parts.push(decoded.text); confidenceSum += decoded.confidence;
          words.push({ text: decoded.text, confidence: decoded.confidence,
            bbox: { x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1 } });
        }
      }
      if (parts.length) {
        texts.push(parts.join(' '));
        if (safeLine) safeTexts.push(parts.join(' '));
      }
      if (!safeLine) completeRead = false;
      onProgress({ status: 'Reading the words.', done: li + 1, total: lines.length });
    }
    // A missing medicine line can leave its directions or daily maximum behind.
    // Do not build a spoken prescription from fragments of different medicines.
    return { text: texts.join('\n'), fieldText: safeTexts.join('\n'), speechText: speechForDocument(safeTexts, confidenceSum / Math.max(1, words.length), completeRead), confidence: words.length ? confidenceSum / words.length : 0,
      words, lines: texts.length, engine: 'ppocrv5', psm: 'ppocrv5-detect', structure: 0, psmRank: 0,
      ms: Math.round(performance.now() - started) };
  }
  async dispose() {
    const sessions = [this.det, this.rec]; this.det = null; this.rec = null; this.charset = null;
    await Promise.allSettled(sessions.map(s => s?.release()));
  }
}
