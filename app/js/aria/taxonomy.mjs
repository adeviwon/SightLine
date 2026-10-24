// js/taxonomy.js — THE DOCUMENT MODEL. Pure data, no DOM, no fetch.
//
// WHAT THIS FILE IS FOR
//   This round narrows the app to two document types and turns the flat result
//   list into labelled sections. Both of those are the same question asked twice:
//   *what are the documents, and what is inside them?* Until now that question was
//   answered in three places at once — DOC_TYPES in js/fields.js, fieldLabels in
//   js/strings.js, and the <ul id="fields"> markup in index.html — and the three
//   could disagree without anything noticing.
//
//   So the shape lives here, once. js/fields.js imports it to classify and to
//   extract; js/ui.js imports it to lay out the section blocks; js/strings.js
//   provides the words. Nothing here holds a user-facing string (rule: every
//   user-facing word is in js/strings.js) — a field carries the *key* of its label,
//   and STRINGS.fieldLabels owns the text.
//
// THE TWO RULES THIS FILE ENCODES (BRIEF.md §1, and this round's §3.2)
//   1. Every field belongs to EXACTLY ONE group. `FIELD_GROUPS` is checked against
//      `FIELDS` by a unit test, so a field with no group, or a field listed under
//      two, fails the build rather than rendering under the wrong heading.
//   2. A group renders only when it has at least one FOUND field. That rule is
//      enforced at render time (js/ui.js), not here — this file only fixes the
//      order, so "Personal details" can never appear below "Warnings".
//
// THE SCOPE CUT (docs/DECISIONS.md D34)
//   `banking`, `medical`, `legal` and `general` are retired. A photo of a bank
//   statement must classify as `unrecognised`, never as a receipt or a
//   prescription — that is what the kept bank-statement fixture now proves.

// The two types the app recognises. Order is presentation order for the spoken
// summary's own fallback; it is not a ranking of the fields.
export const DOC_TYPES = ['receipt', 'prescription'];

// The single fallback. A photo of neither type still gets its transcript and a
// plain explanation — it never gets a wrong type label (this round's brief §3.1).
export const UNRECOGNISED_TYPE = 'unrecognised';

// Every value `classifyDocument().type` may take.
export const CLASSIFIABLE_TYPES = [...DOC_TYPES, UNRECOGNISED_TYPE];

// ---------------------------------------------------------------------------
// GROUPS — the section blocks, in document-importance order, per type.
// ---------------------------------------------------------------------------
// The order IS the importance order: a person scanning a prescription wants the
// medicine and the dose first; a person scanning a receipt wants what they paid.
// `FIELDS[key].group` must name one of the groups listed here for that field's
// type (unit-tested).

export const GROUPS_BY_TYPE = {
  prescription: ['personal', 'prescriber', 'medicine', 'dosage', 'supply', 'dates', 'warnings'],
  receipt: ['shop', 'purchase', 'totals', 'payment', 'reference'],
};

// Group id -> the STRINGS key that holds its heading, and the type it belongs to.
// The heading must clear the 20px measured-type floor (docs/UX_ELDER.md §2.2), so
// it is rendered as a real <h3>, not a caption — see js/ui.js and gate G7.
export const GROUPS = {
  personal: { type: 'prescription', label: 'groupPersonalDetails' },
  prescriber: { type: 'prescription', label: 'groupPrescriber' },
  medicine: { type: 'prescription', label: 'groupMedicine' },
  dosage: { type: 'prescription', label: 'groupDosage' },
  supply: { type: 'prescription', label: 'groupSupply' },
  dates: { type: 'prescription', label: 'groupDates' },
  warnings: { type: 'prescription', label: 'groupWarnings' },
  shop: { type: 'receipt', label: 'groupShop' },
  purchase: { type: 'receipt', label: 'groupPurchase' },
  totals: { type: 'receipt', label: 'groupTotals' },
  payment: { type: 'receipt', label: 'groupPayment' },
  reference: { type: 'receipt', label: 'groupReference' },
};

// ---------------------------------------------------------------------------
// FIELDS — one entry per field, in every type.
// ---------------------------------------------------------------------------
//   group : the section it renders inside (must be a group of `type`)
//   type  : 'receipt' | 'prescription'
//   kind  : the NORMALISATION family. Two fields with the same kind normalise the
//           same way, which is what makes a section internally consistent:
//           money to one currency format, dates to one format, dosage to
//           number+unit. The `kind` is also what the speak-or-omit gate reads to
//           decide whether a value is structurally valid (js/fields.js).
//   label : the key into STRINGS.fieldLabels (never the words themselves)

export const FIELDS = {
  // --- prescription --------------------------------------------------------
  patient_name: { type: 'prescription', group: 'personal', kind: 'name', label: 'patientName' },
  date_of_birth: { type: 'prescription', group: 'personal', kind: 'date', label: 'dateOfBirth' },

  prescriber: { type: 'prescription', group: 'prescriber', kind: 'text', label: 'prescriber' },
  clinic: { type: 'prescription', group: 'prescriber', kind: 'text', label: 'clinic' },

  medicine: { type: 'prescription', group: 'medicine', kind: 'text', label: 'medicine' },
  strength: { type: 'prescription', group: 'medicine', kind: 'strength', label: 'strength' },
  // The FORM is its own row inside Medicine — tablet / capsule / inhaler / syrup /
  // drops / patch. It is deliberately NOT a badge on the document-type label: the
  // brief asks for a visible field row, and "the form is a tablet" is a fact about
  // the medicine, not about what kind of page this is.
  form: { type: 'prescription', group: 'medicine', kind: 'form', label: 'form' },

  dose: { type: 'prescription', group: 'dosage', kind: 'dose', label: 'dose' },
  frequency: { type: 'prescription', group: 'dosage', kind: 'frequency', label: 'frequency' },
  route: { type: 'prescription', group: 'dosage', kind: 'route', label: 'route' },
  duration: { type: 'prescription', group: 'dosage', kind: 'duration', label: 'duration' },

  quantity: { type: 'prescription', group: 'supply', kind: 'count', label: 'quantity' },
  repeats: { type: 'prescription', group: 'supply', kind: 'count', label: 'repeats' },

  issued: { type: 'prescription', group: 'dates', kind: 'date', label: 'issued' },
  valid_until: { type: 'prescription', group: 'dates', kind: 'date', label: 'validUntil' },

  // The caution line, verbatim and short. It is text, not a translation: the app
  // must not paraphrase a safety warning it did not write.
  warning: { type: 'prescription', group: 'warnings', kind: 'warning', label: 'warning' },

  // --- receipt -------------------------------------------------------------
  merchant: { type: 'receipt', group: 'shop', kind: 'text', label: 'merchant' },
  shop_address: { type: 'receipt', group: 'shop', kind: 'text', label: 'shopAddress' },
  shop_phone: { type: 'receipt', group: 'shop', kind: 'phone', label: 'shopPhone' },

  // The receipt's "type": the spending category, derived from the merchant string
  // via a small gazetteer. If the merchant is not recognised the row is ABSENT —
  // never "Unknown", which would present a non-finding as a finding.
  category: { type: 'receipt', group: 'purchase', kind: 'category', label: 'category' },
  purchase_date: { type: 'receipt', group: 'purchase', kind: 'date', label: 'purchaseDate' },
  purchase_time: { type: 'receipt', group: 'purchase', kind: 'time', label: 'purchaseTime' },

  subtotal: { type: 'receipt', group: 'totals', kind: 'money', label: 'subtotal' },
  tax: { type: 'receipt', group: 'totals', kind: 'money', label: 'tax' },
  total: { type: 'receipt', group: 'totals', kind: 'money', label: 'total' },

  payment_method: { type: 'receipt', group: 'payment', kind: 'payment', label: 'paymentMethod' },
  // LAST FOUR DIGITS ONLY, EVER. Nothing in this app stores or renders more than
  // these four; the extractor never keeps the full PAN in the first place.
  card_last4: { type: 'receipt', group: 'payment', kind: 'last4', label: 'cardLast4' },

  receipt_number: { type: 'receipt', group: 'reference', kind: 'text', label: 'receiptNumber' },
  transaction_id: { type: 'receipt', group: 'reference', kind: 'text', label: 'transactionId' },
};

/** The field keys of one type, in group order then declaration order. */
export function fieldsForType(type) {
  const groups = GROUPS_BY_TYPE[type] || [];
  const keys = Object.keys(FIELDS).filter((k) => FIELDS[k].type === type);
  return keys.sort((a, b) => {
    const ga = groups.indexOf(FIELDS[a].group);
    const gb = groups.indexOf(FIELDS[b].group);
    return ga !== gb ? ga - gb : keys.indexOf(a) - keys.indexOf(b);
  });
}

/** Group metadata for a type, in order, with the fields each owns. */
export function groupsForType(type) {
  return (GROUPS_BY_TYPE[type] || []).map((id) => ({
    id,
    label: GROUPS[id].label,
    fields: Object.keys(FIELDS).filter((k) => FIELDS[k].type === type && FIELDS[k].group === id),
  }));
}

export default {
  DOC_TYPES,
  UNRECOGNISED_TYPE,
  CLASSIFIABLE_TYPES,
  GROUPS_BY_TYPE,
  GROUPS,
  FIELDS,
  fieldsForType,
  groupsForType,
};
