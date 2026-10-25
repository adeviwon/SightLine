// A deliberately failed ESM fetch must offer a real recovery, then run live OCR.
const fs = require('fs'), path = require('path'), http = require('http');
const { chromium, webkit } = require('./pw_core.js')();
const ROOT = path.resolve(__dirname, '../..'), APP = path.join(ROOT, 'app');
const OUT = path.join(ROOT, 'artifacts/bilingual-history');
const types = { '.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.wasm':'application/wasm', '.json':'application/json' };
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://local').pathname;
  const file = path.resolve(APP, '.' + (pathname === '/' ? '/index.html' : pathname));
  if (!file.startsWith(APP + '/') || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store' });
  fs.createReadStream(file).pipe(res);
});
(async () => {
  const checks = [];
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    for (const [name, engine] of [['chromium', chromium], ['webkit', webkit]]) {
      const browser = await engine.launch({ headless: true, ...(name === 'chromium' ? { executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' } : {}) });
      try {
        const context = await browser.newContext({ serviceWorkers:'block', viewport:{width:390,height:844} });
        const page = await context.newPage();
        const requested = [];
        page.on('request', request => requested.push(new URL(request.url()).pathname));
        const check = (label, pass) => { checks.push({ engine:name, label, pass:!!pass }); console.log((pass ? 'PASS ' : 'FAIL ') + name + ' ' + label); if (!pass) throw Error(label); };
        await page.route('**/js/layout.js', route => route.abort('failed'));
        await page.goto('http://127.0.0.1:' + server.address().port);
        await page.waitForFunction(() => window.SightLineApp && window.__SightLineReaderWarmup);
        const warmup = await page.evaluate(() => window.__SightLineReaderWarmup);
        check('module failure is detected before opening the photo picker', warmup.error === 'READER_LOAD_FAILED');
        check('JavaScript warmup does not load weights or compile WASM', !requested.some(url => /\.(onnx|wasm)$/.test(url)));
        await page.locator('#file-input').setInputFiles(path.join(ROOT, 'artifacts/aria-merge/sroie-X00016469612.jpg'));
        await page.waitForFunction(() => document.body.dataset.screen === 'error');
        check('loading failure never blames photo quality or displays developer jargon',
          (await page.locator('#error-recovery').textContent()).includes('Reload the reader') &&
          !(await page.locator('#error-banner').textContent()).match(/ocr_engine|tesseract|hard failure|module script/i));
        check('failed module has a reload action and no stale OCR result',
          (await page.locator('#btn-retry').textContent()) === 'Reload reader' && await page.evaluate(() => SightLineApp.lastResult === null));
        await page.locator('#visible-language').selectOption('zh-HK');
        await page.waitForFunction(() => document.getElementById('btn-retry').textContent === '重新載入閱讀器');
        check('recovery controls are localized', true);
        await page.screenshot({ path:path.join(OUT, `reader-recovery-${name}.png`), fullPage:true });
        await page.unroute('**/js/layout.js');
        await Promise.all([page.waitForNavigation(), page.locator('#btn-retry').click()]);
        await page.waitForFunction(() => window.SightLineApp && window.__SightLineReaderWarmup);
        await page.evaluate(() => window.__SightLineReaderWarmup);
        check('reload recovers the module graph and preserves language', await page.evaluate(() => document.documentElement.lang === 'zh-HK' && document.getElementById('badge').textContent !== '請重新載入閱讀器'));
        await page.locator('#file-input').setInputFiles(path.join(ROOT, 'artifacts/aria-merge/sroie-X00016469612.jpg'));
        await page.waitForFunction(() => window.SightLineApp.lastResult || document.body.dataset.screen === 'error', undefined, {timeout:120000});
        const result = await page.evaluate(() => SightLineApp.lastResult);
        check('a real receipt runs PP-OCRv5 after recovery', result?.engine?.ocr === 'ppocrv5' && result?.category === 'receipt');
        await context.close();
      } finally { await browser.close(); }
    }
    fs.writeFileSync(path.join(OUT, 'reader-loading-checks.json'), JSON.stringify(checks, null, 2));
  } finally { await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
