const {test} = require('node:test');
const assert = require('node:assert/strict');
const {build} = require('../js/medicine_sections.js');
function label(text, extra={}) {
  return {category:'medical',engine:{ocr:'ppocrv5'},ocrConfidence:.98,speechText:text,ocrText:text,
    entities:[{label:'DRUG',text:'Amoxicillin',confidence:.9}],fields:[],...extra};
}
const rows = layout => layout.sections.flatMap(group => group.fields);
test('single medicine separates form, strength, each dose, schedule and warnings', () => {
  const result=label('AMOXICILLIN\n500 mg capsules\nTake one capsule three times daily\nKeep out of reach of children', {
    fields:[{label:'Dosage',value:'500 mg',confidence:.88},{label:'Frequency',value:'three times daily',confidence:.88},{label:'Warning',value:'Keep out of reach of children',confidence:.88}]
  });
  const before=JSON.stringify(result), layout=build(result), fields=rows(layout);
  for (const [key,value] of [['Medicine type','capsules'],['Strength on the label','500 mg'],['Amount each time','one capsule'],['When to take it','three times daily']])
    assert.ok(fields.some(f=>f.label===key&&f.value===value),key);
  assert.equal(fields.find(f=>f.label==='Strength on the label').confidence,.88);
  assert.equal(fields.find(f=>f.label==='Medicine name').confidence,.9);
  assert.equal(JSON.stringify(result),before);
});
test('strength and tablet count are different observations, not competing doses',()=>{
  const r=label('AMOXICILLIN 500 mg tablets\nTake 2 tablets twice daily with food', {fields:[{label:'Dosage',value:'500 mg',confidence:.88},{label:'Dosage',value:'2 tablets',confidence:.88}]});
  const layout=build(r);
  assert.equal(layout.directionsResolved,true);
  assert.ok(rows(layout).some(f=>f.value==='2 tablets'));
  assert.ok(rows(layout).some(f=>f.value==='twice daily with food'));
});
test('bare amounts are not relabeled as confirmed strength or amount per dose',()=>{
  const layout=build(label('AMOXICILLIN\n500 mg', {fields:[{label:'Dosage',value:'500 mg',confidence:.88}]}));
  assert.ok(layout.sections.some(g=>g.title==='Amount shown on the label'));
  assert.ok(!layout.sections.some(g=>g.title==='Strength on the label'||g.title==='How much at once'));
});
test('daily maximum and negative directions never become an amount to take',()=>{
  const layout=build(label('AMOXICILLIN\nDo not take 2 tablets\nMaximum 1200 mg in 24 hours'));
  assert.ok(!rows(layout).some(f=>f.label==='Amount each time'||f.label==='Strength on the label'));
});
test('multiple medicines never inherit a shared dose or timing',()=>{
  const layout=build(label('AMOXICILLIN\n500 mg capsules\nTake one capsule twice daily\nIBUPROFEN\n400 mg tablets\nTake two tablets every 6 hours', {entities:[{label:'DRUG',text:'Amoxicillin',confidence:.9},{label:'DRUG',text:'Ibuprofen',confidence:.9}]}));
  assert.equal(layout.directionsResolved,false);
  assert.ok(rows(layout).filter(f=>f.label==='Medicine name').length===2);
  assert.ok(!rows(layout).some(f=>f.label==='Amount each time'||f.label==='When to take it'));
});
test('an unverified second medicine still prevents pairing to the first',()=>{
  const layout=build(label('AMOXICILLIN\n500 mg capsules\nTake one capsule twice daily', {entities:[{label:'DRUG',text:'Amoxicillin',confidence:.9},{label:'DRUG',text:'Ibuprofen',confidence:.4}]}));
  assert.equal(layout.directionsResolved,false);
  assert.ok(!rows(layout).some(f=>f.label==='Amount each time'));
});
test('conflicting schedules stay unassociated',()=>{
  const layout=build(label('AMOXICILLIN\nTake one capsule once daily\nTake two capsules twice daily'));
  assert.equal(layout.ambiguous,true);
  assert.ok(!rows(layout).some(f=>f.label==='When to take it'));
});
test('no trusted text, low field confidence and missing instructions never fabricate values',()=>{
  assert.equal(rows(build(label(''))).length,0);
  const layout=build(label('AMOXICILLIN\n500 mg capsules', {fields:[{label:'Dosage',value:'500 mg',confidence:.4}]}));
  assert.ok(!rows(layout).some(f=>f.value==='500 mg'));
  assert.ok(layout.missing.includes('Amount each time'));
  assert.ok(layout.missing.includes('When to take it'));
});
test('a partial numeric substring cannot become an amount on the label',()=>{
  const layout=build(label('AMOXICILLIN\n15 mg', {fields:[{label:'Dosage',value:'5 mg',confidence:.9}]}));
  assert.ok(!rows(layout).some(f=>f.value==='5 mg'));
});
test('saved results rebuild the same sections without changing source data',()=>{
  const source=label('AMOXICILLIN\n500 mg capsules\nTake one capsule three times daily');
  assert.deepEqual(build(JSON.parse(JSON.stringify(source))),build(source));
});
test('strength text after an instruction never becomes a fabricated schedule',()=>{
  const layout=build(label('AMOXICILLIN\nTake one capsule 500 mg'));
  assert.ok(rows(layout).some(f=>f.label==='Amount each time'));
  assert.ok(!rows(layout).some(f=>f.label==='When to take it'));
});
test('zero strengths, doses and durations are not presented as usable directions',()=>{
  const layout=build(label('AMOXICILLIN\n0 mg tablets\nTake 0 tablets for 0 days'));
  assert.ok(!rows(layout).some(f=>['Strength on the label','Amount each time','How long'].includes(f.label)));
});
test('negative source text cannot be stripped into an affirmative frequency',()=>{
  const layout=build(label('AMOXICILLIN\nDo not take twice daily', {fields:[{label:'Frequency',value:'twice daily',confidence:.9}]}));
  assert.ok(!rows(layout).some(f=>f.label==='When to take it'));
});
test('fallback timing retains route and exceptions rather than dropping qualifiers',()=>{
  const line='Take one capsule by mouth once daily except Sunday';
  const layout=build(label('AMOXICILLIN\n'+line, {fields:[{label:'Frequency',value:'once daily',confidence:.88}]}));
  const when=rows(layout).find(f=>f.label==='When to take it');
  assert.equal(when.value,line);
  assert.equal(when.confidence,.88);
});
