# SightLine — the refined idea (medicine labels only, spoken in Cantonese)

**Rewritten 2026-10-22.** This narrows the product from "reads any document" to a single,
specific job, and states plainly which half of it ships today and which half is the plan.

---

## 1. The idea, in one paragraph

An elderly person in Hong Kong stands in their kitchen with a box of medicine. Two things
stop them reading it: **they can no longer see it clearly, and what is printed on it is in
English they never learned.** They point a phone at the label and take one photo. The phone
reads the label and **says it back to them in Cantonese** — what the medicine is, how strong
it is, and how to take it. All of it happens **on the phone**: no network, no upload, no
account, nothing about their medication leaves the device, and it keeps working with the
network switched off.

## 2. Why narrowing the idea is stronger, not weaker

"An offline reader for any document" is a category that already exists — Seeing AI, Google
Lookout, Envision, Be My Eyes. "**A medicine label your grandmother can hear in Cantonese,
with the network off**" is a single specific claim with a named user, two compounding
barriers, and a policy that already funds it.

The two barriers compound, and that is the whole insight: an app that only fixes the eyes
leaves the language problem, and one that only translates assumes you can see the text it
is translating. **They need both fixed in one gesture** — and the gesture has to be as simple
as taking a photo, because the user is 78, alone, and holding a box at 11pm.

## 3. What this changes about the machine

The refinement removes the hardest thing I previously thought stood in the way.

| job | needed | status |
|---|---|---|
| See the label | camera + line detection | ✅ ships |
| Read the label | recognise printed text | ✅ ships — and HK medicine labels carry the drug name, strength and directions **in English**, which is exactly the `0-9 A-Z` alphabet the recogniser already has |
| Know what matters | extract drug name, strength, dose | ✅ ships |
| **Say it in Cantonese** | a Cantonese voice + a translation dictionary | ⚠️ **not shipped — this is the work** |
| Refuse rather than guess | confidence + evidence gating | ✅ ships |

**The key realisation:** because the label is in *English*, this design does **not** require
Chinese OCR — the 37-symbol `0-9 A-Z` alphabet reads the label as it is. The gap is only on
the *output* side: speaking Cantonese. That converts what looked like a 3–4,000-character
alphabet-expansion project into a bounded speech-and-lexicon change.

### The Cantonese plan, concretely

1. **The voice already has an entry point.** `app/js/tts.js` exposes `speak(text, {lang})`
   and `pickVoice(lang)`, which resolves any language tag and falls back sensibly. Cantonese
   is `zh-HK` — iOS ships Cantonese voices, so the call is a parameter, not a port.
2. **A translation dictionary, from a licence-clean source.** An English drug name must be
   spoken as its Cantonese name. `twinkle-ai/tw-drug-labels-vision` (CC-BY-4.0, 72,326 rows
   from Taiwan's FDA leaflet archive) pairs `drug_name_en` with `drug_name_zh` — a real,
   shippable, bilingual medicine-name lexicon. Field-level content (strength, form,
   directions) needs a small phrase map, not machine translation: a label is a structured
   record, which is the one case where a dictionary beats a model.
3. **Speak the field, never the blob.** The app speaks the extracted fields in a fixed
   Cantonese sentence rather than translating free text, so the utterance stays short,
   checkable, and impossible to hallucinate.

## 4. The safety position (unchanged, and now more important)

It **refuses instead of guessing**. If the recogniser cannot read the label it says so and
asks for another photo, rather than inventing a dose. A wrong dose spoken confidently in
Cantonese is worse than silence. A phone app is not a regulated medical device: the app
labels everything as read-from-your-label, and a licensed clinician is accountable, never
the software. Until Chinese/Cantonese output is evaluated per-language with a release floor,
**this does not go to patients.**

## 5. How the narrowed idea scores on the six judged criteria

| Criterion | The line that carries it |
|---|---|
| **Creativity** | Not "another OCR app" — it removes **two** barriers (can't see it, can't read it) in one gesture, and speaks the user's own language |
| **Innovation** | Our own CRNN trained from scratch, on-device, no OCR library shipped (a build gate enforces it), and a refusal path designed in rather than bolted on |
| **Feasibility** | Installs from a URL, runs in airplane mode, no app store, no account; **89.7% of Hong Kong's 65+ already carry a smartphone** |
| **Potential Impact** | 199,600 report long-term seeing difficulty — **three in four are 65+**; 188,569 over-65s live alone; **46.4% of older HK people take five or more medicines** |
| **Interdisciplinarity** | M medication safety + refusal · S vision and ageing · T on-device ML · E image restoration and degradation engineering · B deployment through the Gerontechnology Promotion Scheme |
| **Teamwork** | We audited our own numbers and published the ones that embarrass us — including the row where Tesseract beats us |

## 6. Policy anchoring (the 2025 edition — the one the Open Challenge rule names)

| Para | Why it is ours |
|---|---|
| **75** | AI for carer support — "*render carer support through an AI chatbot to give carers more convenient access to information*" — and generative AI in clinical workflows |
| **227** | Healthcare reform for an ageing society, including "*enhancing ophthalmic services*" — the only directly vision-related commitment in either edition |
| **266** | Working Group on Ageing Society Strategies, remit covering "*…gerontechnology*"; "*by 2046, people aged 65 and above will account for 36% of our total population*" |

The 2026 edition is the funded follow-through: ¶113 ($300M, AI for Welfare Lab), ¶398–400
($100M, Gerontechnology Promotion Scheme). Verbatim text: `docs/19_POLICY_ANCHOR.md`.

## 7. What is honestly not done yet

1. **Cantonese speech is not shipped.** The app speaks English. This document is the spec for
   the change, not a description of the build.
2. **No medicine-label corpus.** Everything measured so far was measured on receipts and
   synthetic documents. Transfer to medicine labels is **an untested assumption** — and the
   right way to settle it is now available: the CC-BY-4.0 Taiwan leaflet set (72,326 real
   medicine-label images) can be run through the pipeline to produce a *measured* number
   instead of a promise.
3. **Chinese-only labels remain out of reach.** The alphabet emits no CJK character. A
   Cantonese-*speaking* app can serve an English-labelled box today; a fully Chinese label
   needs the alphabet expansion (a data problem, ~3–4k characters) and we say so.
4. **It has not run on physical hardware**, and there is no clinician or NGO partner yet.

Stating these is not a weakness in a pitch; it is the difference between a number a judge can
break in one question and one they can trust.
