# SightLine — the refined idea (medicine labels only, spoken in Cantonese)

**Rewritten 2026-10-22, updated the same day with the shipped build.** This narrows the
product from "reads any document" to a single, specific job. Both halves now exist: the app
reads the label, and it says it back in Cantonese — as a **first build**, with the limits in §7
stated as plainly as the feature.

---

## 1. The idea, in one paragraph

An elderly person in Hong Kong stands in their kitchen with a box of medicine. Two things
stop them reading it: **they can no longer see it clearly, and what is printed on it is in
English they never learned.** They point a phone at the label and take one photo. The phone
reads the label and **says it back to them in Cantonese** — what the medicine is, how strong it
is, and how to take it. All of it happens **on the phone**: no network, no upload, no
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
| **Say it in Cantonese** | a `zh-HK` voice + a licensed zh/en drug-name lexicon + a closed phrase set | ✅ **ships — a first build, 2026-10-22** |
| Refuse rather than guess | confidence + evidence gating | ✅ ships |

**The key realisation held:** because the label is in *English*, this design does **not**
require Chinese OCR — the 37-symbol `0-9 A-Z` alphabet reads the label as it is. The work was
only on the *output* side: speaking Cantonese. What looked like a 3–4,000-character
alphabet-expansion project turned out to be a bounded speech-and-lexicon change, and it is now
built rather than costed.

### What was built, concretely

1. **The voice was already an entry point, as predicted.** `app/js/tts.js` exposes
   `speak(text, {lang})`, `pickVoice(lang)` and `hasVoice(lang)`. Cantonese is `zh-HK`. The
   app offers it as a spoken-language control (`app/index.html` → `#speak-lang`, default
   `en-GB`, option `zh-HK`), and when the device has no Cantonese voice it **shows the
   Cantonese text instead of faking a voice** with the English one — `hasVoice()` exists so the
   app never silently mispronounces.
2. **A translation dictionary, from a licence-clean source — shipped.** An English drug name
   is spoken as its Cantonese name: `app/data/drug_lexicon.json` holds **18,839 English →
   Cantonese name pairs (1.69 MB)** matched against a real bilingual medicine lexicon. Built by
   `ml/tools/hf_lexicon.py`. Licence and attribution are recorded **inside the file**
   (`_licence`): **CC-BY-4.0**, source `twinkle-ai/tw-drug-labels-vision` (Taiwan FDA leaflet
   archive; 72,326 real medicine-label images). The field content (strength, form, directions)
   is a small **closed set of dispensing phrases**, not machine translation — a label is a
   structured record, which is the one case where a dictionary beats a model.
3. **It speaks the field, never the blob.** `app/js/say_cantonese.js` builds a fixed Cantonese
   sentence from the fields we hold — name, dose, directions — rather than translating free
   text, so the utterance stays short, checkable, and impossible to hallucinate.

## 4. The safety position, now enforced in Cantonese too

It **refuses instead of guessing**, in the same breath and by the same rule in both languages.
The Cantonese output path will not speak a sentence it cannot justify:

- An **unreadable label** produces, in Cantonese, "*I cannot see it clearly, I will not guess
  — please confirm with a pharmacist*" (`say_cantonese.js` → `REFUSAL`), not a sentence.
- An **unknown drug name is never guessed, transliterated or approximated.** If we hold a dose
  but not a name, it says so and names no medicine (`NO_NAME`).
- If part of the read was translated and part was not, it **says which** (`PARTIAL`) and sends
  the listener to a pharmacist rather than letting them infer.

A wrong dose spoken confidently in Cantonese is exactly as dangerous as one spoken in English.
A phone app is not a regulated medical device: everything is labelled as read-from-your-label,
and a licensed clinician is accountable, never the software. Until the Cantonese output is
evaluated per-language with a release floor, **this does not go to patients.**

### A bug this safety property caught in the first version — and it is the same class of error the product already fixed

The lexicon is built from real labels, so it necessarily contains entries **keyed on dosage
forms** — the string `TABLET` appears as a key, mapping to whatever product that particular
leaflet described. The first matcher read our own bench document, found the word `TABLET`, and
**confidently offered a specific medicine name as "the drug".** That is the identical failure
mode as the invented account number this project already refuses to ship: *a plausible-looking
match that is simply wrong, in the medium where wrong matters.* Fixed with a stop-list of
dosage forms and excipients in `say_cantonese.js` (`NOT_A_NAME`) plus a rule that a 1–2 word
key containing a form word is not a name; a regression test pins it (`node
app/js/say_cantonese.js`, 10/10).

### A second bug, about verification rather than logic

The shipped 1.7 MB lexicon was **invisible to git**: `.gitignore` had an unanchored `data/`
which swallowed `app/data/`, so the asset would have deployed as a **404 on the live site**
with every local test still passing. It was caught by **checking the live URLs after
deploying**, not by any test. `data/` is now anchored (`/data/`) and the file is committed.

## 5. How the narrowed idea scores on the six judged criteria

| Criterion | The line that carries it |
|---|---|
| **Creativity** | Not "another OCR app" — it removes **two** barriers (can't see it, can't read it) in one gesture, and speaks the user's own language |
| **Innovation** | Our own CRNN trained from scratch, on-device, no OCR library shipped (a build gate enforces it), and **a refusal path designed in rather than bolted on — now in the spoken output too** |
| **Feasibility** | Installs from a URL, runs in airplane mode, no app store, no account; **89.7% of Hong Kong's 65+ already carry a smartphone** |
| **Potential Impact** | 199,600 report long-term seeing difficulty — **three in four are 65+**; 188,569 over-65s live alone; **46.4% of older HK people take five or more medicines** |
| **Interdisciplinarity** | M medication safety + refusal · S vision and ageing · T on-device ML · E image restoration and degradation engineering · B deployment through the Gerontechnology Promotion Scheme |
| **Teamwork** | We audited our own numbers and published the ones that embarrass us — including the row where Tesseract beats us, and the three honest limits of the Cantonese build in §7 |

## 6. Policy anchoring (the 2025 edition — the one the Open Challenge rule names)

| Para | Why it is ours |
|---|---|
| **75** | AI for carer support — "*render carer support through an AI chatbot to give carers more convenient access to information*" — and generative AI in clinical workflows |
| **227** | Healthcare reform for an ageing society, including "*enhancing ophthalmic services*" — the only directly vision-related commitment in either edition |
| **266** | Working Group on Ageing Society Strategies, remit covering "*…gerontechnology*"; "*by 2046, people aged 65 and above will account for 36% of our total population*" |

The 2026 edition is the funded follow-through: ¶113 ($300M, AI for Welfare Lab), ¶398–400
($100M, Gerontechnology Promotion Scheme). Verbatim text: `docs/19_POLICY_ANCHOR.md`.

## 7. The honest limits — three of them, and a judge should hear them from us

1. **It has not been reviewed by a Cantonese speaker.** The sentence frames are the author's,
   not a native copywriter's. In a product for this audience that is a real gap, and a first
   fix, not a footnote.
2. **The drug-name step is unvalidated on real medicine labels.** Measured on our own
   12-document bench it matched **nothing** honestly — that corpus is synthetic receipts and
   statements, and our own reads garble names anyway. The lexicon is licensed and real; **its
   coverage of what we can actually read is not yet measured.** Transfer to medicine labels
   remains an untested assumption, and the way to settle it is available: run the CC-BY-4.0
   Taiwan leaflet set (72,326 real medicine-label images) through the pipeline for a *measured*
   number instead of a promise.
3. **No Chinese voice ships with the web engine.** WebKit under Playwright has **no Chinese
   voice installed**, so the app **shows the Cantonese text rather than faking a voice** with
   the English one. On a phone the voice must be present in Settings — an Apple-side
   precondition the demo device must be checked for before filming.

Also still true, unchanged: **Chinese-only labels remain out of reach** — the alphabet emits no
CJK character. A Cantonese-*speaking* app serves an English-labelled box today; a fully Chinese
label needs the alphabet expansion (a data problem, ~3–4k characters). And it has **not run on
physical hardware**, and there is **no clinician or NGO partner** yet.

Stating these is not a weakness in a pitch; it is the difference between a number a judge can
break in one question and one they can trust.

## 8. What is measured on this build (the gates a judge can re-run)

| gate | command | result |
|---|---|---|
| the utterance logic, pure | `node app/js/say_cantonese.js` | **10/10** |
| the safety property in the real engine, **against the live site** | `SMOKE_BASE=https://adeviwon.github.io/SightLine/ node app/tools/cantonese_verify.js --engine=webkit` | **14/14** |
| the lexicon contract | `ml/tests/test_drug_lexicon.py` | **7/7** |
| the shipped app, unchanged | `node app/tools/ios_full_check.js --engine=webkit` | **91/91** |

Every one of these asserts the *refusal*, not just the happy path — the reason a wrong dose is
never spoken confidently is that the test asks it to fail correctly.
