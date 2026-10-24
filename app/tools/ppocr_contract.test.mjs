import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { PaddleReader, speechForDocument } from '../js/ppocr_core.js';
import { ctcDecode, buildCharset } from '../js/ppocr_math.js';
import { recognizeImage, reset } from '../js/ppocr.js';

test('model bytes match the recorded conversion and dictionary', () => {
  const metadata = JSON.parse(readFileSync(new URL('../models/ppocr-provenance.json', import.meta.url)));
  for (const [name, expected] of Object.entries(metadata.files)) {
    const bytes = readFileSync(new URL('../models/' + name, import.meta.url));
    assert.equal(bytes.length, expected.bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), expected.sha256);
  }
});
test('CTC uses dictionary offset and preserves decimal punctuation', () => {
  const chars = buildCharset('5\n.\n0\n');
  const indices = [1, 0, 2, 3, 0, 3];
  const probs = new Float32Array(indices.length * chars.length);
  indices.forEach((n, i) => { probs[i * chars.length + n] = 1; });
  assert.equal(ctcDecode(probs, indices.length, chars.length, chars).text, '5.00');
});
test('an incomplete medicine read cannot turn a daily maximum into a spoken dosage', () => {
  const fragments = ['Amoxicillin 500 mg', 'Maximum 1200mg in 24 hours'];
  assert.equal(speechForDocument(fragments, .96, false), '');
  assert.equal(speechForDocument(['Amoxicillin 500 mg'], .74, true), '');
  assert.equal(speechForDocument(['Amoxicillin 500 mg'], .96, true), 'Amoxicillin 500 mg');
});
test('partially initialized models release sessions when recognizer loading fails', async () => {
  let released = 0, creates = 0;
  const runtime = { env: { wasm: {} }, InferenceSession: { create: async () => {
    if (++creates === 1) return { release: async () => { released++; } };
    throw new Error('bad graph');
  } } };
  const reader = new PaddleReader();
  await assert.rejects(reader.prepare({ runtime, dictionary: 'A\n' }), /bad graph/);
  assert.equal(released, 1); assert.equal(reader.det, null); assert.equal(reader.rec, null);
});
test('invalid image dimensions are rejected before inference', async () => {
  const reader = new PaddleReader(); reader.det = {}; reader.rec = {};
  await assert.rejects(reader.recognize({ data: new Uint8Array(), width: 0, height: 1 }), /Choose another photo/);
});
class FakeWorker {
  static instances = [];
  constructor() { FakeWorker.instances.push(this); this.terminated = false; }
  postMessage(message) { this.message = message; }
  terminate() { this.terminated = true; }
  finish(result) { this.onmessage({ data: { id: this.message.id, result } }); }
}
const image = () => ({ data: new Uint8Array(4), width: 2, height: 2 });
test('busy calls cannot overlap; successful result preserves engine identity', async () => {
  const previous = globalThis.Worker; globalThis.Worker = FakeWorker;
  try {
    const first = recognizeImage(image());
    await assert.rejects(recognizeImage(image()), /already being read/);
    FakeWorker.instances.at(-1).finish({ text: 'TOTAL 5.00', engine: 'ppocrv5' });
    assert.equal((await first).engine, 'ppocrv5');
  } finally { reset(); globalThis.Worker = previous; }
});
test('a timeout terminates unfinished WASM and allows a clean retry', async () => {
  const previous = globalThis.Worker; globalThis.Worker = FakeWorker;
  try {
    const first = recognizeImage(image(), null, 5), worker = FakeWorker.instances.at(-1);
    await assert.rejects(first, /too long/); assert.equal(worker.terminated, true);
    const retry = recognizeImage(image());
    const next = FakeWorker.instances.at(-1); assert.notEqual(next, worker);
    worker.onerror(); // a late error must not terminate the new worker
    worker.finish({ text: 'stale' }); // late output from the terminated generation must be ignored
    next.finish({ text: 'new', engine: 'ppocrv5' });
    assert.equal((await retry).text, 'new');
  } finally { reset(); globalThis.Worker = previous; }
});
test('reset cancels an active scan and unavailable workers fail with an action', async () => {
  const previous = globalThis.Worker; globalThis.Worker = FakeWorker;
  try {
    const pending = recognizeImage(image()); reset();
    await assert.rejects(pending, /stopped/);
    globalThis.Worker = undefined;
    await assert.rejects(recognizeImage(image()), /Choose Original reader/);
  } finally { reset(); globalThis.Worker = previous; }
});


test('speech fields cannot turn lookalike characters into an invented dose', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const pipeline = require('../js/pipeline.js');
  assert.equal(pipeline.retainVerbatim([{value:'500mg'}], 'Take S00m9').length, 0);
  assert.equal(pipeline.retainVerbatim([{value:'500mg'}], 'Take 500 mg').length, 1);
  assert.equal(pipeline.retainVerbatim([{value:'5.00'}], 'Amount 500').length, 0);
  assert.deepEqual(pipeline.medicalFieldContext([{label:'Dosage',value:'1200mg'}], 'Maximum 1200mg in 24 hours'),
    [{label:'Warning',value:'Maximum 1200mg in 24 hours'}]);
  assert.equal(pipeline.medicalFieldContext([{label:'Dosage',value:'500mg'}], 'Amoxicillin 500 mg capsules')[0].label, 'Dosage');
});
