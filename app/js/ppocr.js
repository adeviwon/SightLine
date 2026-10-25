/** The worker is the cancellation boundary: timeout and reset terminate WASM,
 * rather than leaving inference alive behind a rejected Promise. No cloud calls.
 */
let worker = null, pending = null, nextId = 0;
function getWorker() {
  if (worker) return worker;
  if (typeof Worker !== 'function') throw new Error('Enhanced reading is unavailable here. Choose Original reader.');
  worker = new Worker(new URL('./ppocr.worker.js', import.meta.url), { type: 'module', name: 'SightLine reader' });
  worker.onmessage = ({ data }) => {
    if (!pending || data.id !== pending.id) return;
    if (data.progress) { pending.onProgress?.(data.progress); return; }
    const job = pending; pending = null; clearTimeout(job.timer);
    if (data.error) job.reject(new Error(data.error)); else job.resolve(data.result);
  };
  const generation = worker;
  worker.onerror = () => {
    if (worker === generation) reset('The reader stopped. Try again or choose Original reader.');
  };
  return worker;
}
export function recognizeImage(image, onProgress, timeout = 60000) {
  if (pending) return Promise.reject(new Error('A page is already being read.'));
  return new Promise((resolve, reject) => {
    try {
      const reader = getWorker(), id = ++nextId;
      pending = { id, resolve, reject, onProgress,
        timer: setTimeout(() => reset('Reading took too long. Try a smaller photo.'), timeout) };
      reader.postMessage({ id, image }, [image.data.buffer]);
    } catch (error) { reset(error.message); reject(error); }
  });
}
export async function runPaddleOCR(candidates, onProgress, structureScore) {
  if (!candidates?.length) throw new Error('Choose a picture to read.');
  // PP-OCRv5 expects continuous-tone input. Read the raw/deskewed candidate once;
  // the CRNN's restored/binarized candidate ladder does not apply to this model.
  const canvas = candidates[0];
  const rgba = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < gray.length; i++) gray[i] = (rgba[i*4]*77 + rgba[i*4+1]*150 + rgba[i*4+2]*29) >> 8;
  const result = await recognizeImage({ data: gray, width: canvas.width, height: canvas.height }, onProgress);
  result.structure = structureScore ? structureScore(result.text) : 0;
  return result;
}
export function reset(message = 'Reading was stopped.') {
  worker?.terminate(); worker = null;
  if (pending) { const job = pending; pending = null; clearTimeout(job.timer); job.reject(new Error(message)); }
}
