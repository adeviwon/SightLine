// Real WASM precision comparison over fixed synthetic SightLine fixtures.
// No canvas/browser/camera emulation: input is the fixture's grayscale pixels.
import { PaddleReader } from '../js/ppocr_core.js';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { createRequire } from 'node:module';
const pipeline=createRequire(import.meta.url)('../js/pipeline.js');
const root=resolve(import.meta.dirname,'../..'),work=mkdtempSync(join(tmpdir(),'sl-native-'));
const runtimeStart=performance.now();
const runtime=await import('../vendor/ort/ort.min.mjs');
const runtimeImportMs=performance.now()-runtimeStart;
const manifest=JSON.parse(readFileSync('/tmp/sightline_samples/manifest.json'));
const python=`from pathlib import Path\nfrom PIL import Image\nimport json,sys\nm=json.loads(Path(sys.argv[1]).read_text())\nfor s in m['samples']:\n p=Path(s['file']); Image.open(p).convert('L').save(Path(sys.argv[2])/(p.stem+'.pgm'))\n`;
const child=spawnSync('bash',['run.sh','py','-c',python,'/tmp/sightline_samples/manifest.json',work],{cwd:root,encoding:'utf8'});
if(child.status!==0)throw Error(child.stderr);
function norm(s){return String(s).toUpperCase().replace(/[^A-Z0-9]+/g,' ').trim().replace(/\s+/g,' ');}
function distance(a,b){let row=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){let prev=row[0];row[0]=i+1;for(let j=0;j<b.length;j++){const old=row[j+1];row[j+1]=Math.min(row[j+1]+1,row[j]+1,prev+(a[i]!==b[j]));prev=old;}}return row[b.length];}
const comparison=process.argv.find(a=>a.startsWith('--uint8='));
const variants=comparison ? [['uint8',resolve(comparison.slice(8))],['fp32',resolve(root,'app/models/ppocrv5-mobile-rec.onnx')]] : [['fp32',resolve(root,'app/models/ppocrv5-mobile-rec.onnx')]];
const results=[];
try{
for(const[precision,model]of variants){
 const reader=new PaddleReader(),start=performance.now();
 await reader.prepare({runtime,detModel:new Uint8Array(readFileSync(resolve(root,'app/models/ppocrv5-mobile-det.onnx'))),recModel:new Uint8Array(readFileSync(model)),dictionary:readFileSync(resolve(root,'app/models/ppocrv5_dict.txt'),'utf8')});
 const prepareMs=performance.now()-start;
 const samples=[];
 for(const sample of manifest.samples){
  const stem=sample.doc+'__'+sample.profile;
  const buffer=readFileSync(join(work,stem+'.pgm'));
  const match=/^P5\n(\d+) (\d+)\n255\n/.exec(buffer.subarray(0,80).toString('ascii'));
  const output=await reader.recognize({data:new Uint8Array(buffer.subarray(match[0].length)),width:+match[1],height:+match[2]});
  const ref=norm(manifest.text[sample.doc]);
  const literal=s=>s.toUpperCase().replace(/\s+/g,'');
  const expected=manifest.fields[sample.doc];
  const punctuationRef=manifest.text[sample.doc].toUpperCase().replace(/\s+/g,' ').trim();
  const punctuationRead=output.text.toUpperCase().replace(/\s+/g,' ').trim();
  const category=pipeline.understand(output.text).category;
  const understood=pipeline.understand(output.speechText,{category});
  let fields=pipeline.retainVerbatim(understood.fields,output.speechText);
  if(category==='medical')fields=pipeline.medicalFieldContext(fields,output.speechText);
  samples.push({...sample,file:stem,precision,ms:output.ms,cer:distance(ref,norm(output.text))/ref.length,punctuationCER:distance(punctuationRef,punctuationRead)/punctuationRef.length,expectedFields:expected.length,recognitionFieldHits:expected.filter(v=>literal(output.text).includes(literal(v))).length,speechFieldHits:expected.filter(v=>literal(output.speechText).includes(literal(v))).length,extractedFieldHits:expected.filter(v=>fields.some(f=>literal(f.value).includes(literal(v)))).length,words:output.words.length,speechCharacters:output.speechText.length,confidence:output.confidence});
  console.log(precision,stem,output.ms+'ms','CER',samples.at(-1).cer.toFixed(4));
 }
 const warmStart=performance.now();await reader.prepare();const warmPrepareMs=performance.now()-warmStart;
 let invalidInputRejected=false;
 try{await reader.recognize({width:0,height:1,data:new Uint8Array()});}
 catch{invalidInputRejected=true;}
 const blank=await reader.recognize({width:640,height:480,data:new Uint8Array(640*480).fill(255)});
 const memoryBeforeRepeats=process.memoryUsage(),repeats=[];
 const repeatBuffer=readFileSync(join(work,'prescription__studio_clean.pgm'));
 const repeatHeader=/^P5\n(\d+) (\d+)\n255\n/.exec(repeatBuffer.subarray(0,80).toString('ascii'));
 for(let i=0;i<3;i++){
  const output=await reader.recognize({data:new Uint8Array(repeatBuffer.subarray(repeatHeader[0].length)),width:+repeatHeader[1],height:+repeatHeader[2]});
  repeats.push({ms:output.ms,characters:output.text.length,memory:process.memoryUsage()});
 }
 results.push({precision,runtimeImportMs,prepareMs,warmPrepareMs,meanCER:samples.reduce((a,b)=>a+b.cer,0)/samples.length,samples,invalidInputRejected,blank:{characters:blank.text.length,speechCharacters:blank.speechText.length,confidence:blank.confidence,ms:blank.ms},memoryBeforeRepeats,repeats});
 await reader.dispose();
}
writeFileSync(resolve(root, comparison ? 'artifacts/integration/precision-comparison.json' : 'artifacts/integration/native-fp32.json'),JSON.stringify({scope:'Native ORT Web 1.20.1 WASM, fixed synthetic SightLine images, no browser preprocessing',results},null,2)+'\n');
console.log('SUMMARY',JSON.stringify(results.map(({precision,prepareMs,meanCER})=>({precision,prepareMs,meanCER}))));
}finally{rmSync(work,{recursive:true,force:true});}
