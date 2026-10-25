// Server-free development harness. Embeds the EXACT shipped files in Blob URLs
// so a file:// review can exercise real browser worker inference without relaxing
// browser security settings. Deployment URL loading / service worker are NOT tested.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
const root=resolve(import.meta.dirname,'../..'),app=resolve(root,'app');
const read=p=>readFileSync(resolve(app,p),'utf8');
const b64=p=>readFileSync(resolve(app,p)).toString('base64');
const strip=s=>s.replace(/^import[\s\S]*?from ['"][^'"]+['"];\n/gm,'').replace(/^export /gm,'');
const core=strip(read('js/ppocr_core.js')), math=strip(read('js/ppocr_math.js'));
const workerSource=read('vendor/ort/ort.min.js')+'\n'+math+'\n'+core+`
const reader=new PaddleReader();let resource;
self.onmessage=async({data})=>{try{
 if(data.assets)resource=data.assets;
 ort.env.wasm.wasmBinary=resource.wasm;
 const mjs=URL.createObjectURL(new Blob([resource.mjs],{type:'application/javascript'}));
 await reader.prepare({runtime:ort,detModel:resource.det,recModel:resource.rec,dictionary:resource.dict,wasmPaths:{mjs},onProgress:progress=>self.postMessage({id:data.id,progress})});
 const result=await reader.recognize(data.image,progress=>self.postMessage({id:data.id,progress}));
 self.postMessage({id:data.id,result});
}catch(error){self.postMessage({id:data.id,error:String(error.stack||error)})}};`;
let bridge=strip(read('js/ppocr.js'));
bridge=bridge.replace("new URL('./ppocr.worker.js', import.meta.url)",'workerURL');
bridge=bridge.replace('reader.postMessage({ id, image }, [image.data.buffer]);', `reader.postMessage({id,image,assets:resources},[image.data.buffer]);`);
const js=`const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const resources={det:decode('${b64('models/ppocrv5-mobile-det.onnx')}'),rec:decode('${b64('models/ppocrv5-mobile-rec.onnx')}'),wasm:decode('${b64('vendor/ort/ort-wasm-simd-threaded.wasm')}'),dict:${JSON.stringify(read('models/ppocrv5_dict.txt'))},mjs:${JSON.stringify(read('vendor/ort/ort-wasm-simd-threaded.mjs'))}};
const workerURL=URL.createObjectURL(new Blob([${JSON.stringify(workerSource)}],{type:'application/javascript'}));
${bridge}
window.addEventListener('load',async()=>{
 const badge=document.getElementById('badge');badge.textContent='Browser worker test';
 document.getElementById('skip-classify').checked=true;document.getElementById('skip-restore').checked=true;
 SightLine.runOCR=(c,p)=>runPaddleOCR(c,p);
 const image=new Image();image.src='data:image/png;base64,${readFileSync('/tmp/sightline_samples/prescription__studio_clean.png').toString('base64')}';
 await image.decode();
 const start=performance.now();
 window.__workerHeartbeat=0;const beat=setInterval(()=>window.__workerHeartbeat++,30);
 try {window.__workerResult=await SightLineApp.runScan(image,'Synthetic prescription');}
 catch(error){window.__workerError=String(error.stack||error);}
 finally{clearInterval(beat);window.__workerElapsed=performance.now()-start;document.title=window.__workerResult?'SightLine • worker scan complete':'SightLine • worker test needs attention';badge.textContent='Browser worker test';}
});`;
let html=read('index.html').replace('<head>',`<head><base href="${new URL('file://'+app+'/').href}">`);
html=html.replace('</body>',`<script type="module">${js.replace(/<\/script/gi,'<\\/script')}</script></body>`);
const target=resolve(root,'artifacts/integration/worker-preview.html');
writeFileSync(target,html);
console.log('Built test-only standalone harness:',target);
