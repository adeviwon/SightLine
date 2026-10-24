/** Presentation of verified English label text. Never mutates extraction or
 * speech eligibility, and never pairs directions across different medicines. */
const SightLineMedicineSections = (() => {
  const clean = value => String(value || '').trim();
  const unique = values => [...new Set(values.filter(Boolean))];
  const normal = value => clean(value).toLowerCase().replace(/\s+/g, ' ');
  function contains(line, value) {
    const source = normal(line), wanted = normal(value);
    if (!wanted) return false;
    let from = 0;
    const adjacent = /^\d/.test(wanted) ? /[a-z\d.,]/i : /[a-z\d]/i;
    for (;;) {
      const at = source.indexOf(wanted, from);
      if (at < 0) return false;
      if (!adjacent.test(source[at - 1] || '') && !adjacent.test(source[at + wanted.length] || '')) return true;
      from = at + 1;
    }
  }
  function build(result) {
    if (result?.category !== 'medical') return { sections: [], directionsResolved: false };
    const text = clean(result.speechText);
    const lines = text.split(/\r?\n/).map(clean).filter(Boolean);
    const pageConfidence = Number.isFinite(result.ocrConfidence) ? Math.min(1, Math.max(0, result.ocrConfidence)) : 0;
    const confidence = line => Math.min(pageConfidence, ...(result.ocrWords || [])
      .filter(word => clean(word.text) && normal(line).includes(normal(word.text)))
      .map(word => Number.isFinite(word.confidence) ? word.confidence : 0));
    const verified = line => text && confidence(line) >= 0.75;
    const sections = [];
    const field = (label, value, line) => ({ label, value, sourceLine: line,
      confidence: Math.min(confidence(line), ...(result.fields || []).filter(f => normal(f.value) === normal(value))
        .map(f => Number.isFinite(f.confidence) ? f.confidence : 0),
        ...(result.entities || []).filter(e => normal(e.text) === normal(value))
          .map(e => Number.isFinite(e.confidence) ? e.confidence : 0)) });
    const add = (title, fields) => {
      const safe = fields.filter(f => Number.isFinite(f.confidence) && f.confidence >= 0.75);
      if (safe.length) sections.push({ title, fields: safe });
    };
    const names = unique((result.entities || []).filter(entity => entity.label === 'DRUG')
      .map(entity => {
        const line = lines.find(line => contains(line, entity.text));
        if (!line || !verified(line) || !Number.isFinite(entity.confidence) || entity.confidence < 0.75) return '';
        const at = line.toLowerCase().indexOf(String(entity.text).toLowerCase());
        return line.slice(at, at + String(entity.text).length);
      }));
    const medicine = names.map(name => field('Medicine name', name, lines.find(line => contains(line, name))));
    if (!names.length) return { sections: [], directionsResolved: false, multipleMedicines: false };
    const single = names.length === 1 && unique((result.entities || []).filter(e => e.label === 'DRUG').map(e => normal(e.text))).length === 1;
    const strength = [], forms = [], doses = [], timings = [], durations = [];
    if (single) for (const line of lines.filter(verified)) {
      // A maximum is a warning, never a strength or an amount to take.
      if (/\b(?:do not|maximum|max|not more than|never)\b/i.test(line)) continue;
      const namePattern = names[0].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const labelLine = line.replace(new RegExp('^' + namePattern + '\\b[ :,-]*', 'i'), '');
      const strengthMatch = labelLine.match(/^(?:strength\s*:\s*)?(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu))\s+(tablets?|capsules?|drops?|patch(?:es)?|inhalers?)\b/i)
        || line.match(/^strength\s*:\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu))\s*$/i);
      if (strengthMatch && Number.parseFloat(strengthMatch[1]) > 0) {
        strength.push(field('Strength on the label', strengthMatch[1], line));
        if (strengthMatch[2]) forms.push(field('Medicine type', strengthMatch[2], line));
      }
      const dose = line.match(/^(?:(?:directions|instructions|sig)\s*:\s*)?(?:take|use|swallow|inhale)\s+((?:one|two|three|four|half|\d+(?:\.\d+)?)\s+(?:tablets?|capsules?|ml|drops?|puffs?|sprays?))\b/i);
      if (dose && !/^0(?:\.0+)?\s/.test(dose[1])) {
        doses.push(field('Amount each time', dose[1], line));
        const rest = line.slice(dose.index + dose[0].length).trim();
        if (/^(?:once|twice|(?:three|four|\d+)\s+times|every|at\s+(?:night|bedtime)|in\s+the\s+(?:morning|evening)|with\s+food|before|after|as\s+(?:needed|required)|if|when)\b/i.test(rest))
          timings.push(field('When to take it', rest, line));
      }
      const duration = line.match(/\bfor\s+\d+\s+(?:days?|weeks?|months?)\b/i);
      if (duration && Number(duration[0].match(/\d+/)[0]) > 0) durations.push(field('How long', duration[0], line));
    }
    const distinct = fields => unique(fields.map(f => normal(f.value))).length;
    const dosageFields = (result.fields || []).filter(f => f.label === 'Dosage');
    const mappedDosages = dosageFields.every(f => [...strength, ...doses].some(part => contains(part.sourceLine, f.value)));
    const directionsResolved = single && distinct(strength) <= 1 && distinct(doses) <= 1 && distinct(timings) <= 1
      && (dosageFields.length <= 1 || mappedDosages);
    const dedupe = fields => fields.filter((f, i) => fields.findIndex(other => normal(other.value) === normal(f.value)) === i);
    if (single && distinct(forms) === 1) medicine.push(forms[0]);
    add('Medicine', medicine);
    if (directionsResolved) {
      add('Strength on the label', dedupe(strength));
      add('How much at once', dedupe(doses));
      add('When to take it', dedupe(timings));
      add('How long', dedupe(durations));
    }
    const matchingLines = f => lines.filter(line => verified(line) && contains(line, f.value));
    const existing = (result.fields || []).filter(f => Number.isFinite(f.confidence) && f.confidence >= 0.75
      && matchingLines(f).length && !(/^(Dosage|Frequency)$/.test(f.label) && matchingLines(f)
        .some(line => /\b(?:do not|maximum|max|not more than|never)\b/i.test(line))));
    // Do not relabel a bare mg number as a strength or a dose without context.
    if (directionsResolved && !strength.length && !doses.length) {
      add('Amount shown on the label', existing.filter(f => f.label === 'Dosage')
        .map(f => ({...f, label:'Strength or dose text'})));
    }
    if (directionsResolved && !timings.length) add('When to take it', existing.filter(f => f.label === 'Frequency')
      .map(f => { const line=matchingLines(f)[0]; const item=field('When to take it',line,line); item.confidence=Math.min(item.confidence,f.confidence); return item; }));
    add('Warnings', existing.filter(f => /warning/i.test(f.label)));
    add('Personal details', existing.filter(f => /patient|birth/i.test(f.label)));
    add('Prescriber', existing.filter(f => /doctor|prescriber|pharmacist/i.test(f.label)));
    const found = fields => fields.some(f => f.confidence >= 0.75);
    const missing = single ? [!found(forms) && 'Medicine type', !found(strength) && 'Strength on the label',
      !found(doses) && 'Amount each time', !found(timings) && !existing.some(f => f.label === 'Frequency') && 'When to take it'].filter(Boolean) : [];
    return { sections, directionsResolved, multipleMedicines: !single, ambiguous: single && !directionsResolved, missing };
  }
  return { build };
})();
if (typeof window !== 'undefined') window.SightLineMedicineSections = SightLineMedicineSections;
if (typeof module !== 'undefined' && module.exports) module.exports = SightLineMedicineSections;
