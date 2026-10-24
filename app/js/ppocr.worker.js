import { PaddleReader } from './ppocr_core.js';
const reader = new PaddleReader();
let busy = false;
self.onmessage = async ({ data }) => {
  if (busy) { self.postMessage({ id: data.id, error: 'A page is already being read.' }); return; }
  busy = true;
  const progress = value => self.postMessage({ id: data.id, progress: value });
  try {
    await reader.prepare({ onProgress: progress });
    const result = await reader.recognize(data.image, progress);
    self.postMessage({ id: data.id, result });
  } catch (error) {
    await reader.dispose();
    self.postMessage({ id: data.id, error: 'The page could not be read. Try again or choose Original reader.', detail: String(error?.message || error) });
  } finally { busy = false; }
};
