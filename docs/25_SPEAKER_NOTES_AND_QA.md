# Grand Final — speaker notes, timing and Q&A

For the live pitch at the Data Technology Hub, Tseung Kwan O, **2026-11-14**. Slides:
`docs/22_GRAND_FINAL_DECK.html` (10 slides; PDF at `docs/22_GRAND_FINAL_DECK.pdf`, which is
what gets emailed to the organisers **48 h ahead**). Judges are listed in `docs/23`.

**Scope, narrowed 2026-10-22:** this is a **medicine-label** tool, not a general document
reader. The user is an elderly person who cannot see the label clearly **and** cannot read the
English printed on it; the task we set the AI is to read the label and **say it back in
Cantonese**. Cantonese is **not shipped** — the app speaks English (`app/js/tts.js` defaults
`en-GB`), and every measurement in this repo was taken on that English build. Whenever we
state the Cantonese goal we state that in the same breath: `docs/26_IDEA_REFINED.md` is the
canonical statement, `docs/26` §7 and the README blockquote carry the honest sentence.

**Assume ~6:45 total.** If the slot is 5 minutes, use the cut list at the bottom.

## Per-slide script and timing

| # | Slide | sec | Say (don't read the slide — the slide IS the number) |
|---|---|---|---|
| 1 | Title | 15 | "We're Team STEMB. This is SightLine: it reads a **medicine label** aloud, on your phone, with no network at all. The goal is Cantonese — today it speaks English, and we'll keep saying that honestly all the way through." |
| 2 | The human | 35 | "She is 78. The pharmacy hands her a box of medicine, and two things stop her at once: her eyes can no longer make out the small print, and the label is printed in **English she never learned**. So the dose gets guessed — or it waits for a neighbour who isn't coming today. That's the whole problem, and it's a person, not a statistic." |
| 3 | Problem at scale | 40 | "It isn't one person. 199,600 people in Hong Kong report long-term seeing difficulty — three in four are over 65, and 188,569 people over 65 live alone. 89.7% of them already carry a smartphone, so this needs no new device. Now add the second barrier: the drug, the strength and the directions are printed **in English**. An app that only magnifies the print leaves half the problem standing, and one that only translates assumes she can see it." |
| 4 | The solution | 35 | "SightLine runs entirely on the phone. Photograph the medicine label: it reads it, names the drug, the strength and the dose, and **refuses to guess** anything it can't read. One honest line on where we are: today it reads that English and **speaks English**; the Cantonese voice is the next build. We say that because every number we quote was measured on the English one." |
| 5 | **LIVE DEMO** | 90 | Say almost nothing. Airplane mode ON, the medicine box held to the camera. Hand the phone to a judge if you can. Run the hard case too and let it refuse — "I could not read that dose clearly, please check it with a pharmacist." Then: "That refusal is the design. A wrong dose spoken confidently is worse than no dose." |
| 6 | How it works | 45 | "Camera, line detection, then our own recogniser — a CRNN with CTC, 942,166 parameters, trained from scratch, a 3.6 MB recogniser graph (26.4 MB of models and tokenizer, about 37.2 MiB installed). No OCR library is shipped; a build gate fails the build if one comes back. Then field extraction, then speech — and speech is the honest gap: English today. Cantonese is a voice parameter plus a lexicon, not a new recogniser, because the label is already English." |
| 7 | Measured results | 50 | **Lead with the honest row: "On a clean page Tesseract reads 95.7% of lines exactly, whole-page, against our 10.6% — and on our own hard captures it beats us too, because the simulator we had been measuring ourselves against turned out to be 17 to 67 times harsher than a real photograph. We found that ourselves and corrected it."** Then: "What ships end to end reads **57.2%** of the words, CER 0.217, 33.3% of the ground-truth fields and 91.7% of the documents classified right — on **twelve synthetic rendered documents, in WebKit**. The 45.9% we used to quote came from a stale set of test images: we keep that row and label it, because 1.7 of the 11.3 points is the benchmark getting more realistic and 9.5 is the retrain — I won't sell those as one gain." Then the case, honestly: "So why ours? One 3.6 MB graph with no second runtime, instead of a second engine plus four megabytes of language data; nothing leaves the phone by construction; and a refusal path — because on a clean page Tesseract read the dose 500mg as 300mg. **On accuracy it beats us everywhere we have measured, including hand-held captures.** A wrong dose spoken confidently is the failure we exist to prevent. Also true: these tables are receipts and synthetic documents, we have no medicine-label corpus yet, and that transfer is an assumption we have not settled." |
| 8 | What we got wrong | 40 | "We found four defects in our own train/serve contract. The width we served truncated 98.8% of lines. Padding wasn't inert in a bidirectional network. The image transform ran at serve and never at train. And our deskew was applied backwards — it was doubling the tilt. Each one now has a gate that fails the build." |
| 9 | Impact and policy | 40 | "This is already funded policy. The 2025 Policy Address — the edition the Open Challenge names — commits to AI carer support at Para 75, ophthalmic services at Para 227, and a working group on gerontechnology at Para 266. The 2026 edition followed through with $300 million and $100 million. A carer hearing their own medicine label, on the phone they already own." |
| 10 | Close | 15 | "It reads the label — or it says it cannot. The goal is that voice in Cantonese; today it is English, and either way nothing leaves the phone. Thank you." |

## Which slide serves which judged criterion

| Criterion | Slide | The line that carries it |
|---|---|---|
| Creativity | 2, 4 | two compounding barriers removed in one gesture — can't see it, can't read the English — and the goal of speaking the user's own language |
| Innovation | 8 | four contract defects found and gated by the team that shipped them |
| Feasibility | 4, 6 | installs from a URL, works in airplane mode, no app store, no account; and the Cantonese gap is a voice parameter + a licence-clean lexicon, not a new recogniser |
| Potential Impact | 2, 3, 9 | named user, verified HK figures, and the scheme that already funds it |
| Interdisciplinarity | 1, 3, 9 | T (on-device ML) · E (image restoration) · S (vision science) · M (medication safety) · B (deployment via the Gerontechnology scheme) |
| Teamwork | 8, 10 | we audited our own numbers and published the ones that embarrass us — including that Cantonese isn't shipped |

## Judge-specific angles

- **Prof Shahid Khan (hepatology, healthcare innovation)** → lead with medication safety, the
  refusal behaviour and who is accountable. State plainly: **a phone app is not a regulated
  medical device; a licensed clinician is accountable, never the software.**
- **Dr Jackie Bell (computing, EDI)** → technical rigour. Have slide 8 ready to dwell on: the
  gates, the mutation-tested failure direction, the Tesseract head-to-head we didn't win. If she
  pushes on the language gap, give Q6 verbatim — the honest version is the stronger answer.
- **Dr Pamela Tin (Research Director, HK Chief Executive's Policy Unit)** → the evidence and the
  policy anchoring. `docs/20_IMPACT_EVIDENCE.md` has every figure with its source and year; the
  definitional caveat on 199,600 is in `docs/13`. Offer the caveat before she finds it.
- **Google Hong Kong judge** → deployment path via the Gerontechnology Promotion Scheme and NGO
  partnerships. **Confirm the name first** — two official Imperial pages disagree (Helen Kwok
  vs Lucas Cheung); do not put a name on a slide until resolved (`docs/13`).

## Q&A — the nine we will be asked

1. **"Why not just use Tesseract?"** **On accuracy it beats us everywhere we have measured** —
   whole-page it reads 95.7% of lines exactly at CER 0.000 and 100% of the fields, and it beats
   us on hand-held captures too. We publish that first, and the earlier "we win 5.5x on real
   captures" claim is WITHDRAWN: it was a property of our old, unrealistically harsh simulator,
   which broke Tesseract's binariser. Our case is not the score — it is a second runtime plus
   4.1 MB of language data (10.6 with osd) against our one 3.6 MB graph; nothing leaving the
   phone by construction; and a **refusal path**, which the direct Tesseract integration has
   none of. On a clean page it read the dose 500mg as 300mg.
2. **"Is it accurate enough?"** Not on arbitrary photographs, and we say so: **57.2%** word
   accuracy end to end — on **12 synthetic rendered documents, WebKit, one record per
   document, current fixtures** — and 1.8% line-exact on 20 real hand-photographed pages. The
   45.9% we used to quote was measured on a **previous, stale fixture generation**; we keep
   that row and label it, because +1.7 of the move is the fixture correction and +9.5 is the
   retrain. The design answer is the refusal. We publish per-condition numbers instead of one
   headline — and those numbers come from receipts and synthetic documents, not from a
   medicine-label corpus.
3. **"What happens if it reads a dose wrong?"** It refuses rather than guesses; it never presents
   a dose as authoritative; a clinician is accountable. The measured failure mode is why the
   document-evidence gate exists (speckle once hallucinated a plausible account number at 0.678
   confidence, inside the real-read band).
4. **"What about Chinese? The label is 繁體中文."** The shipped alphabet is digits + A–Z + space:
   **zero Chinese characters today**, and we say that publicly. A fully Chinese-only label needs
   the alphabet expanded (~3–4k characters) — a data problem we are not claiming to have solved.
   But our user's box is not Chinese-only: HK labels carry the drug name, strength and directions
   **in English**, which is why the language gap sits on the output side, not the input side.
5. **"Does it work on a real phone?"** It installs and runs on iOS and Android as a PWA, verified
   in WebKit emulation; **it has not yet been run on physical hardware**, which is our next step
   and the honest limit of what we can claim today.
6. **"Does it speak Cantonese?"** **No — not yet, and we will not say it does until it is
   measured.** What ships today reads printed English and speaks English (`app/js/tts.js` defaults
   `lang: "en-GB"`); every number in this repo was taken on that build. Here is the whole gap:
   (a) **the voice** — `app/js/tts.js` already exposes `speak(text, {lang})` and `pickVoice(lang)`,
   which resolves any language tag; Cantonese is `zh-HK`, an iOS-shipped voice, so it is a
   parameter change, not a port; (b) **the words** — an English drug name has to be spoken as its
   Cantonese name, and `twinkle-ai/tw-drug-labels-vision` (CC-BY-4.0, 72,326 zh/en drug-name pairs
   from Taiwan's FDA leaflet archive) is a shippable, licence-clean lexicon for exactly that; the
   strength/form/direction phrases are a small phrase map, because a label is a structured record,
   which is the one case where a dictionary beats a model. **And it does not need Chinese OCR:**
   the label is in English, which is the `0-9 A-Z` alphabet the recogniser already has. So this is
   a bounded speech-and-lexicon change, not a 3–4,000-character alphabet expansion. The reason we
   still say "not shipped" is the rule we set ourselves: no language output goes to patients until
   it is evaluated per-language with a release floor (`docs/26` §3–§4).
7. **"So what's left before it speaks Cantonese?"** Three tasks, named: wire `zh-HK` through the
   speech call and the UI; ship the drug-name lexicon plus the phrase map; evaluate the Cantonese
   utterances against the English ones and give them a release floor. Then it can be claimed.
8. **"Is this not just Seeing AI / Google Lens?"** Those upload the page to a server. Ours makes
   no network call after install — provable from the service worker in minutes, and by the
   airplane-mode demo.
9. **"Team across STEMB?"** Answer with the table above, and name who owns which discipline.

## Rehearsal checklist

- [ ] Demo phone charged, app installed from the live URL, **airplane mode verified the same day**
- [ ] Laptop with the local copy as fallback (`bash run.sh serve`), no venue Wi-Fi dependency
- [ ] Slides emailed 48 h ahead, and a copy on a USB stick and in the cloud
- [ ] The refusal case rehearsed until it lands as a design choice, not a failure
- [ ] Q6 rehearsed until "no, not yet — here is exactly what remains" sounds like confidence, not apology
- [ ] Every number said aloud traced to a file (`docs/14`, `docs/24`, `docs/03`)
- [ ] Time it five times; the demo beat always runs long

## If the slot is 5 minutes (cut list)

Cut slide 6 to 20 s (pipeline in one sentence) and slide 9 to 20 s (name the paragraphs, don't
quote them), and drop slide 8's fourth defect. **Never cut the demo, the refusal, or the honest
Cantonese line — a judge who catches the overclaim costs more than the 15 seconds saved.**
