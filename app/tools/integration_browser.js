/** Real controller/worker/offline regression. Synthetic repository fixtures only.
 * PW_CORE points to Playwright; APP_ROOT and OUT allow isolated before/after runs.
 */
const fs = require('fs'), path = require('path'), http = require('http');
const pw = process.env.PW_CORE ? require(process.env.PW_CORE) : require('./pw_core.js')();
const root = path.resolve(process.env.APP_ROOT || path.join(__dirname, '..'));
const label = process.env.LABEL || 'after';
const out = path.resolve(process.env.OUT || path.join(root, '../artifacts/integration/browser'));
fs.mkdirSync(out, { recursive: true });
const manifest = JSON.parse(fs.readFileSync('/tmp/sightline_samples/manifest.json'));
const types = { '.js':'application/javascript', '.mjs':'application/javascript', '.wasm':'application/wasm', '.html':'text/html', '.css':'text/css', '.json':'application/json', '.png':'image/png', '.txt':'text/plain' };
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const sample = url.pathname.startsWith('/samples/');
  const file = path.resolve(sample ? '/tmp/sightline_samples' : root, '.' + (sample ? url.pathname.slice(8) : url.pathname === '/' ? '/index.html' : url.pathname));
  const base = sample ? '/tmp/sightline_samples' : root;
  if (!file.startsWith(base + path.sep) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type':types[path.extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store' }); fs.createReadStream(file).pipe(res);
});
const checks = [], records = [], errors = [], external = [];
const check = (name, ok, detail) => { checks.push({ name, ok:!!ok, detail }); console.log((ok?'PASS ':'FAIL ') + name + (ok?'':' '+JSON.stringify(detail))); };
function distance(a,b) { let row=Array.from({length:b.length+1},(_,i)=>i); for(let i=0;i<a.length;i++){let prev=row[0];row[0]=i+1;for(let j=0;j<b.length;j++){const old=row[j+1];row[j+1]=Math.min(row[j+1]+1,row[j]+1,prev+(a[i]!==b[j]));prev=old;}}return row[b.length]; }
const norm = s => String(s).toUpperCase().replace(/[^A-Z0-9]+/g,' ').trim().replace(/\s+/g,' ');
let browser;
(async () => {
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
  const base = 'http://127.0.0.1:' + server.address().port;
  browser = await pw.chromium.launch({ headless:true, args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'] });
  const context = await browser.newContext({ viewport:{width:390,height:844}, colorScheme:'light', permissions:['camera'] });
  // Isolate OCR/UI from the optional MiniLM classifier on the bounded CI host.
  // The classifier remains covered by the native bundle inference gate.
  await context.route('**/models/minilm*', route => route.abort());
  const page = await context.newPage();
  page.on('pageerror', e=>errors.push(String(e)));
  page.on('request', r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith(base))external.push(r.url());});
  await page.goto(base, { waitUntil:'load' });
  await page.waitForFunction(()=>!!window.SightLineApp, { timeout:30000 });
  await page.waitForTimeout(1200);
  for(const [w,h] of [[390,844],[320,568],[1280,900]]) {
    await page.setViewportSize({width:w,height:h});
    await page.screenshot({path:path.join(out,`${label}-idle-${w}.png`),fullPage:true});
    const geometry=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,minTarget:Math.min(...[...document.querySelectorAll('button')].filter(e=>e.getBoundingClientRect().height>0).map(e=>e.getBoundingClientRect().height))}));
    check(`${w}px idle has no sideways scroll`,geometry.scroll<=w,geometry);
    if(label==='after') check(`${w}px buttons meet 48px floor`,geometry.minTarget>=48,geometry);
  }
  await page.setViewportSize({width:390,height:844});
  const subset = label==='before' ? manifest.samples.filter(s=>s.profile==='studio_clean'||s.doc==='prescription'&&s.profile==='handheld_light') : manifest.samples;
  async function scan(sample, expectedEngine = "ppocrv5") {
    await page.locator('#skip-classify').evaluate(e => { e.checked = true; });
    const file=path.basename(sample.file), previous=await page.evaluate(()=>window.SightLineApp.lastResult?.timestamp);
    await page.evaluate(()=>{ window.__slBeat={ticks:0,maxGap:0,last:performance.now()};window.__slTimer=setInterval(()=>{const now=performance.now();window.__slBeat.maxGap=Math.max(window.__slBeat.maxGap,now-window.__slBeat.last);window.__slBeat.last=now;window.__slBeat.ticks++;},30); });
    console.log('SCAN START', sample.doc, sample.profile, expectedEngine);
    await page.locator('#file-input').setInputFiles(path.join('/tmp/sightline_samples',file));
    const trace = setInterval(() => page.locator('#progress-label').textContent({timeout:2000}).then(text => console.log('SCAN PHASE', text)).catch(() => {}), 5000);
    try { await page.waitForFunction(old=>{const result=window.SightLineApp.lastResult;const error=document.getElementById('error-banner');return result?.timestamp!==old&&!!result || !error.hidden&&/failed|could not|stopped/i.test(error.textContent);},previous,{timeout:75000}); } finally { clearInterval(trace); }
    const result=await page.evaluate(()=>{clearInterval(window.__slTimer);return {result:window.SightLineApp.lastResult,error:document.getElementById('error-banner').textContent,beat:window.__slBeat};});
    check(`${sample.doc}/${sample.profile} completes`,!!result.result,result.error);
    if(!result.result) throw new Error(result.error);
    const r=result.result, gt=norm(manifest.text[sample.doc]), hyp=norm(r.ocrText);
    const fields=manifest.fields[sample.doc];
    records.push({doc:sample.doc,profile:sample.profile,engine:r.engine.ocr||'crnn',ms:r.ms,cer:distance(gt,hyp)/Math.max(1,gt.length),fieldsPresent:fields.filter(f=>hyp.replace(/ /g,'').includes(norm(f).replace(/ /g,''))).length,expectedFields:fields.length,extracted:r.fields.length,heartbeatTicks:result.beat.ticks,maxHeartbeatGap:result.beat.maxGap});
    if(label==='after')check(`${sample.doc}/${sample.profile} uses selected ${expectedEngine}`,r.engine.ocr===expectedEngine,r.engine);
    return r;
  }
  for(const sample of subset.slice(0, Number(process.env.MAX_SAMPLES) || subset.length)) await scan(sample);
  const first=manifest.samples.find(s=>s.doc==='prescription'&&s.profile==='studio_clean');
  await scan(first);
  for(const [w,h] of [[390,844],[320,568],[1280,900]]) {
    await page.setViewportSize({width:w,height:h});
    const geometry=await page.evaluate(()=>{const first=document.querySelector('.field-row');return {scroll:document.documentElement.scrollWidth,firstFieldY:first?.getBoundingClientRect().top+scrollY,photoHeight:document.getElementById('viewfinder').getBoundingClientRect().height};});
    check(`${w}px results have no sideways scroll`,geometry.scroll<=w,geometry);
    await page.screenshot({path:path.join(out,`${label}-results-${w}.png`),fullPage:true});
  }
  if(label==='after' && !process.env.QUICK) {
    await page.setViewportSize({width:390,height:844});
    await page.waitForFunction(()=>/Ready offline/.test(document.getElementById('badge').textContent),{timeout:60000});
    check('required model files are verified ready offline',true);
    const cached=await page.evaluate(async()=>{const keys=await caches.keys();const cache=await caches.open(keys.find(k=>k.startsWith('sightline-')));return (await cache.keys()).map(x=>x.url);});
    check('both PP-OCR models and worker are in service-worker cache',['ppocrv5-mobile-det.onnx','ppocrv5-mobile-rec.onnx','ppocr.worker.js'].every(f=>cached.some(x=>x.endsWith(f))),cached.length);
    await context.setOffline(true); await page.reload({waitUntil:'load'});
    await page.waitForFunction(()=>!!window.SightLineApp); await scan(first);
    check('cold page reload and a new worker scan succeed offline',true);
    await context.setOffline(false);
    await page.locator('#file-input').setInputFiles({name:'unreadable.png',mimeType:'image/png',buffer:Buffer.from('not an image')});
    await page.waitForFunction(()=>!document.getElementById('error-banner').hidden);
    check('invalid input clears previous results and read-aloud action',await page.evaluate(()=>!window.SightLineApp.lastResult&&document.getElementById('btn-speak').hidden));
    await scan(first); check('valid scan recovers after invalid input',true);
    await page.locator('.reading-options').evaluate(e=>e.open=true);
    await page.locator('#ocr-model').selectOption('crnn');
    await scan(first, 'crnn');
    check('original reader remains explicitly selectable',await page.evaluate(()=>window.SightLineApp.lastResult.engine.ocr==='crnn'));
    await page.locator('#ocr-model').selectOption('ppocrv5');
  }
  check('no uncaught page errors',errors.length===0,errors);
  check('no external runtime requests',external.length===0,external);
})().catch(error=>{check('browser validation completes',false,String(error.stack||error));}).finally(async()=>{
  if(browser)await browser.close();server.close();
  const report={label,scope:'Headless Linux Chromium with synthetic documents; optional MiniLM disabled; no camera/device verification',checks,records,errors,external};
  fs.writeFileSync(path.join(out,`${label}.json`),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({label,pass:checks.filter(x=>x.ok).length,fail:checks.filter(x=>!x.ok).length,scans:records.length}));
  process.exitCode=checks.some(x=>!x.ok)?1:0;
});
