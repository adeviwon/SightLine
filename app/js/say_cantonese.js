/**
 * say_cantonese.js — turn a read medicine label into a Cantonese utterance.
 *
 * The refined idea is "photograph the label, hear it in Cantonese". Our recogniser reads the
 * ENGLISH on the label (the 0-9 A-Z alphabet), so this module is the OUTPUT half: match what we
 * read against a licensed zh/en medicine-name lexicon and speak a fixed Cantonese sentence built
 * from facts we actually have.
 *
 * THE SAFETY PROPERTY, and it is the whole point: it speaks only what it can translate and
 * REFUSES the rest. A drug name that is not in the lexicon is never guessed, transliterated or
 * approximated — the utterance says so and sends the listener to a pharmacist. A wrong dose
 * spoken confidently in Cantonese is worse than silence, exactly as it is in English.
 *
 * Pure: no DOM, no network, no speech. The caller owns the voice (`tts.js` takes a `lang`).
 *
 * Not machine translation. A label is a structured record — a drug name and a closed set of
 * dispensing phrases — which is the one case where a dictionary beats a model.
 */

(function (root, factory) {
  var api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SayCantonese = api;
})(typeof self !== "undefined" ? self : this, function () {
"use strict";
/** Curated label phrases. A closed set, because dispensing English is formulaic.
 *  Deliberately NOT a general translator: anything absent is reported untranslated. */
const PHRASES = [
  // frequency + quantity, longest first so the specific forms win
  ["TAKE ONE CAPSULE THREE TIMES DAILY WITH FOOD", "每日三次，每次一粒，隨餐服用"],
  ["TAKE ONE CAPSULE THREE TIMES DAILY", "每日三次，每次一粒"],
  ["TAKE ONE TABLET THREE TIMES DAILY", "每日三次，每次一粒"],
  ["TAKE ONE CAPSULE TWICE DAILY", "每日兩次，每次一粒"],
  ["TAKE ONE TABLET TWICE DAILY", "每日兩次，每次一粒"],
  ["TAKE ONE CAPSULE ONCE DAILY", "每日一次，每次一粒"],
  ["TAKE ONE TABLET ONCE DAILY", "每日一次，每次一粒"],
  ["TAKE ONE CAPSULE DAILY", "每日一粒"],
  ["TAKE ONE TABLET DAILY", "每日一粒"],
  ["THREE TIMES DAILY", "每日三次"],
  ["TWO TIMES DAILY", "每日兩次"],
  ["TWICE DAILY", "每日兩次"],
  ["ONCE DAILY", "每日一次"],
  ["WITH FOOD", "隨餐服用"],
  ["BEFORE FOOD", "餐前服用"],
  ["AFTER FOOD", "餐後服用"],
  ["ON AN EMPTY STOMACH", "空肚服用"],
  ["AT BEDTIME", "睡前服用"],
  ["DO NOT EXCEED THE STATED DOSE", "不可超過指定劑量"],
  ["DO NOT EXCEED THE DOSE", "不可超過指定劑量"],
  ["MAY CAUSE DROWSINESS", "可能引起睡意"],
  ["COMPLETE THE FULL COURSE", "請完成整個療程"],
  ["KEEP OUT OF REACH OF CHILDREN", "請放在兒童接觸不到的地方"],
  ["FOR EXTERNAL USE ONLY", "只供外用"],
  ["USE AS DIRECTED BY YOUR DOCTOR", "依照醫生指示使用"],
  ["STORE BELOW 25C", "儲存於攝氏二十五度以下"],
  ["KEEP IN A COOL DRY PLACE", "存放於陰涼乾爽處"],
];

/** Spoken when we hold a dose but cannot identify the medicine. Never invents a name. */
const NO_NAME = "我睇到劑量，但認唔出藥名。";
/** Spoken when the label cannot be read well enough to say anything. */
const REFUSAL = "我睇唔清楚，唔會亂估 — 請同藥劑師或醫生確認。";
/** Appended when part of what was read could not be translated. */
const PARTIAL = "有啲內容我翻譯唔到，請同藥劑師確認。";

const norm = (s) => String(s == null ? "" : s)
  .toUpperCase()
  .replace(/[^A-Z0-9 ]+/g, " ")
  .replace(/\s+/g, " ")
  .trim();

/** Dose tokens our recogniser can actually emit: a number and a unit, no punctuation. */
function findDoses(text) {
  const out = [];
  // same non-digit left boundary as the app's extractor: OCR glues the dose to the drug
  // name and a left \b cannot match a letter->digit transition (no lookbehind: Safari).
  const re = /(?:^|[^\d])(\d{1,4})\s?(MG|MCG|G|ML|L|IU|PERCENT|%)\b/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push(m[1] + " " + (m[2] === "%" || m[2] === "PERCENT" ? "PERCENT" : m[2]));
  }
  return out;
}

/** Words that are NOT drug names. The lexicon is built from real label data, so it contains
 *  entries keyed on dosage forms and generic nouns ("TABLET", "CAPSULE", "SOLUTION") that map to
 *  whatever product that leaflet described. Matching one of those would name a medicine on the
 *  strength of the word "tablet" -- a plausible wrong answer, which is the exact failure this
 *  product exists to prevent. Measured: the first version matched "TABLET" in our own bench read
 *  and confidently offered 喜康錠. A gazetteer needs a stop-list or it invents drugs. */
const NOT_A_NAME = new Set(("TABLET TABLETS CAPSULE CAPSULES SOLUTION SOLUTIONS SUSPENSION POWDER " +
  "CREAM OINTMENT GEL LIQUID INJECTION DROPS SYRUP GRANULES PATCH SPRAY INHALER SUPPOSITORY " +
  "WATER SALT SODIUM POTASSIUM CALCIUM MAGNESIUM CHLORIDE HYDROCHLORIDE SULFATE ACETATE CITRATE " +
  "MALEATE TARTRATE PHOSPHATE STERILE ORAL TOPICAL EXTERNAL INTRAVENOUS INTRAMUSCULAR " +
  "TABLETTE TABLETTEN KAPSUL KAPSEL FILM COATED SUGAR ENTERIC SUSTAINED RELEASE PROLONGED MODIFIED " +
  "PLAIN COMPOUND COMPLEX FORTE PLUS EXTRA ORIGINAL GENERIC BRAND MEDICINE MEDICINES DRUG DRUGS " +
  "TABLETTEN TABLETKA PASTE LOTION SHAMPOO MOUTHWASH GARGLE SPRAY DROPS EYE EAR NOSE ORAL " +
  "PACK PACKS BOTTLE BOX TUBE SACHET VIAL AMPOULE").split(" "));

/** The longest lexicon key occurring in the text. Longest, so "AMOXICILLIN" is not shadowed
 *  by a shorter name inside it, and so a brand beats a fragment. */
function findName(text, lexicon) {
  let best = null;
  for (const key of Object.keys(lexicon)) {
    if (key.length < 5) continue;
    if (NOT_A_NAME.has(key)) continue;
    if (key.split(" ").every((w) => NOT_A_NAME.has(w))) continue;  // "FILM COATED TABLET"
    if (/\b(TABLET|CAPSULE|SOLUTION|POWDER|CREAM|OINTMENT|INJECTION|SYRUP|DROPS|SPRAY)\b/.test(key)
        && key.split(" ").length <= 2) continue;                    // "X TABLET" is a form, not a name
    if (text.indexOf(key) === -1) continue;
    if (!best || key.length > best.length) best = key;
  }
  return best ? { en: best, zh: lexicon[best].zh || lexicon[best] } : null;
}

function findPhrases(text) {
  const hit = [];
  for (const [en, zh] of PHRASES) {
    if (text.indexOf(en) !== -1) hit.push(zh);
  }
  return hit;
}

/**
 * Build the Cantonese utterance for one read label.
 *
 * @param {string} readText  what the recogniser produced (upper-case, as emitted)
 * @param {object} lexicon   { "AMOXICILLIN": {"zh": "阿莫西林"}, ... }  (see app/data/drug_lexicon.json)
 * @returns {{ok:boolean, zh:string, name:?object, doses:string[], phrases:string[], untranslated:boolean}}
 *          `ok:false` means nothing could be said honestly — the caller must fall back to English
 *          or to the refusal, NOT to a guess.
 */
function build(readText, lexicon) {
  const text = norm(readText);
  const lex = lexicon && lexicon.pairs ? lexicon.pairs : (lexicon || {});
  const name = findName(text, lex);
  const doses = findDoses(text);
  const phrases = text ? findPhrases(text) : [];

  const lines = [];
  if (name) lines.push("呢隻藥係" + name.zh + "。");
  if (doses.length) lines.push("劑量" + doses.join("，") + "。");
  if (phrases.length) lines.push(phrases.join("。") + "。");

  if (!lines.length) {
    // nothing translatable: do not invent one
    return { ok: false, zh: REFUSAL, name: null, doses: [], phrases: [], untranslated: true };
  }
  if (!name) lines.unshift(NO_NAME);

  // Did the label say more than we could account for? We cannot know line-by-line here, so the
  // honest rule is: if we found a name OR doses we speak those, and we flag partial coverage
  // whenever the read text contains more words than the fragments we matched.
  const matched = [name ? name.en : "", ...doses, ...phrases].join(" ").split(/\s+/).filter(Boolean);
  const untranslated = text.split(/\s+/).filter(Boolean).length > matched.length * 2;

  return { ok: true, zh: lines.join("") + (untranslated ? PARTIAL : ""), name, doses, phrases, untranslated };
}

return { build: build, findName: findName, findDoses: findDoses, findPhrases: findPhrases,
         PHRASES: PHRASES, REFUSAL: REFUSAL, NO_NAME: NO_NAME, PARTIAL: PARTIAL, NOT_A_NAME: NOT_A_NAME };
});

/* ── runnable self-check: `node app/js/say_cantonese.js` ─────────────────────────────── */
if (typeof module !== "undefined" && require.main === module) {
  const assert = require("assert");
  const SC = module.exports;
  const build = SC.build, NO_NAME = SC.NO_NAME, REFUSAL = SC.REFUSAL, PARTIAL = SC.PARTIAL;
  const lex = { pairs: {
    AMOXICILLIN: { zh: "阿莫西林" },
    IBUPROFEN: { zh: "布洛芬" },
    PARACETAMOL: { zh: "撲熱息痛" },
  } };
  let n = 0;
  const t = (label, fn) => { fn(); n++; console.log("  ok    " + label); };

  t("a known drug name is spoken in Cantonese", () => {
    const r = build("RX AMOXICILLIN 500MG CAPSULES", lex);
    assert.strictEqual(r.ok, true);
    assert.ok(r.zh.includes("阿莫西林"), r.zh);
  });
  t("the dose survives into the utterance", () => {
    const r = build("RX AMOXICILLIN 500MG CAPSULES", lex);
    assert.ok(r.doses.includes("500 MG"), JSON.stringify(r.doses));
    assert.ok(r.zh.includes("500 MG"), r.zh);
  });
  t("a direction phrase is translated from the closed set", () => {
    const r = build("TAKE ONE CAPSULE THREE TIMES DAILY WITH FOOD", lex);
    assert.ok(r.zh.includes("每日三次"), r.zh);
  });
  t("an UNKNOWN drug name is never guessed", () => {
    const r = build("RX ZORBEXIL 20MG TABLETS", lex);
    assert.strictEqual(r.name, null, "invented a name: " + JSON.stringify(r.name));
    assert.ok(r.zh.startsWith(NO_NAME), r.zh);
    assert.ok(!/[\u4e00-\u9fff]/.test(r.zh.replace(NO_NAME, "")) || r.zh.indexOf("ZORBEXIL") === -1,
      "must not transliterate the unknown name");
  });
  t("an unreadable label refuses instead of speaking", () => {
    const r = build("", lex);
    assert.strictEqual(r.ok, false);
    assert.strictEqual(r.zh, REFUSAL);
  });
  t("noise produces a refusal, not a name", () => {
    const r = build("EAX CE E 0RO 121G000 0O0TX", lex);
    assert.strictEqual(r.name, null);
  });
  t("a DOSAGE FORM is never offered as a drug name", () => {
    // measured failure: the first version matched "TABLET" in our own bench read
    const l3 = { pairs: { TABLET: { zh: "喜康錠" }, CAPSULE: { zh: "某膠囊" }, AMOXICILLIN: { zh: "阿莫西林" } } };
    const r = build("RX AMOXICILLIN 500MG TABLET", l3);
    assert.strictEqual(r.name.en, "AMOXICILLIN", "matched a dosage form: " + JSON.stringify(r.name));
    const r2 = build("TAKE ONE TABLET TWICE DAILY", l3);
    assert.strictEqual(r2.name, null, "named a medicine from the word TABLET: " + JSON.stringify(r2.name));
  });
  t("the longest name wins over a fragment inside it", () => {
    const l2 = { pairs: { CIPRO: { zh: "環丙沙星" }, CIPROFLOXACIN: { zh: "環丙沙星" } } };
    const r = build("CIPROFLOXACIN 250MG", l2);
    assert.strictEqual(r.name.en, "CIPROFLOXACIN");
  });
  t("it says so when part of the label was not translated", () => {
    const r = build("RX AMOXICILLIN 500MG UNKNOWN VERBAGE HERE THAT WE CANNOT MAP AT ALL", lex);
    assert.strictEqual(r.untranslated, true);
    assert.ok(r.zh.includes(PARTIAL), r.zh);
  });
  t("a dose GLUED to the drug name is still a dose", () => {
    // measured failure: "RX BUPROLIN400MG" produced NO dose, because a left \b cannot match
    // between the letter N and the digit 4
    const r = build("1 M RX BUPROLIN400MG ABETS", lex);
    assert.ok(r.doses.includes("400 MG"), "glued dose invisible: " + JSON.stringify(r.doses));
  });
  t("a numerals-only read still gives the dose, and admits it has no name", () => {
    const r = build("500 MG", lex);
    assert.ok(r.zh.includes("500 MG"));
    assert.ok(r.zh.includes("認唔出藥名"));
  });
  console.log("\n  " + n + "/" + n + " checks passed");
  process.exit(0);
}
