// Capture actual OCR results in both display languages; synthetic label is
// deliberately identified, never substituted for the OCR pipeline.
const fs=require('fs'), path=require('path'), http=require('http');
const {chromium}=require('./pw_core.js')();
const ROOT=path.resolve(__dirname,'../..'), APP=path.join(ROOT,'app'), OUT=path.join(ROOT,'artifacts/medicine-sections');
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm'};
const server=http.createServer((req,res)=>{
  const name=new URL(req.url,'http://local').pathname;
  const file=path.resolve(APP,'.'+(name==='/'?'/index.html':name));
  if(!file.startsWith(APP+'/')||!fs.existsSync(file)||fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});fs.createReadStream(file).pipe(res);
});
(async()=>{
  fs.mkdirSync(OUT,{recursive:true});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_BIN||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const context=await browser.newContext({viewport:{width:390,height:844}});
    const page=await context.newPage();
    await page.goto('http://127.0.0.1:'+server.address().port);
    await page.waitForFunction(()=>window.SightLineApp);
    await page.locator('#file-input').setInputFiles(path.join(ROOT,'artifacts/redesign/fixtures/synthetic-single-label.png'));
    await page.waitForFunction(()=>SightLineApp.lastResult,undefined,{timeout:120000});
    const result=await page.evaluate(()=>SightLineApp.lastResult);
    if(result.engine.ocr!=='ppocrv5'||result.category!=='medical') throw Error('Actual label OCR failed');
    fs.writeFileSync(path.join(OUT,'source-result.json'),JSON.stringify(result,null,2));
    for(const language of ['en-GB','zh-HK']) {
      await page.selectOption('#visible-language',language);
      for(const width of [320,390,768,1440]) {
        await page.setViewportSize({width,height:width<500?844:1000});
        if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)) throw Error('Overflow '+language+' '+width);
        await page.screenshot({path:path.join(OUT,`medicine-${language}-${width}.png`),fullPage:true});
        console.log('PASS sectioned label '+language+' '+width);
      }
    }
    const AxeBuilder=require(process.env.AXE_CORE||'@axe-core/playwright').default;
    const audit=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    fs.writeFileSync(path.join(OUT,'chinese-axe.json'),JSON.stringify(audit.violations,null,2));
    if(audit.violations.length) throw Error('Chinese accessibility violations: '+audit.violations.map(v=>v.id).join(','));
    console.log('PASS Chinese sectioned results automated accessibility');
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
