import pathlib,json,re
p=pathlib.Path('artifacts/aria-merge');gt=json.loads((p/'ground-truth.json').read_text())
def norm(s):return ' '.join(s.upper().split())
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a):
  nxt=[i+1]
  for j,y in enumerate(b):nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
  row=nxt
 return row[-1]
truth=norm(gt['text']);res={}
for name in ['sightline','aria']:
 r=json.loads((p/(name+'-before.json')).read_text());text=r.get('ocrText',r.get('text',''))
 res[name]={'CER':distance(truth,norm(text))/len(truth),'confidence':r.get('ocrConfidence',r.get('record',{}).get('ocr',{}).get('mean_confidence',0)/100),'characters':len(text),'elapsedMs':r.get('ms',r.get('testElapsed'))}
(p/'accuracy.json').write_text(json.dumps({'scope':'One real public scanned receipt; not a representative accuracy benchmark. CER uses uppercase whitespace-normalized page text, dataset word ordering.','groundTruthEntities':gt['entities'],'measurements':res},indent=2))
print(json.dumps(res))
