# Grand Final — speaker notes, timing and Q&A

For the live pitch at the Data Technology Hub, Tseung Kwan O, **2026-11-14**. Slides:
`docs/22_GRAND_FINAL_DECK.html` (10 slides; PDF at `docs/22_GRAND_FINAL_DECK.pdf`, which is
what gets emailed to the organisers **48 h ahead**). Judges are listed in `docs/23`.

**Assume ~6:45 total.** If the slot is 5 minutes, use the cut list at the bottom.

## Per-slide script and timing

| # | Slide | sec | Say (don't read the slide — the slide IS the number) |
|---|---|---|---|
| 1 | Title | 15 | "We're Team STEMB. This is SightLine: it reads your documents aloud, on your phone, with no network at all." |
| 2 | The human | 35 | "Mrs. Chan is 78. She lives alone. The pharmacy hands her this — and to her it might as well be blank. She either guesses the dose or waits for a neighbour. That's the whole problem, and it's a person, not a statistic." |
| 3 | Problem at scale | 40 | "It isn't one person. 199,600 people in Hong Kong report long-term seeing difficulty — three in four are over 65. 188,569 people over 65 live alone. And 89.7% of them already carry a smartphone. That last number matters: this needs no new device." |
| 4 | The solution | 35 | "SightLine runs entirely on the phone. Point it at a document: it reads it aloud, names what kind of document it is, speaks the fields that matter — the dose, the account number, the case reference — and refuses to guess anything it can't read." |
| 5 | **LIVE DEMO** | 90 | Say almost nothing. Airplane mode ON, held to the camera. Hand the phone to a judge if you can. Run the hard case too and let it refuse — "I could not read that label clearly, please check it with a pharmacist." Then: "That refusal is the design. A wrong dose spoken confidently is worse than no dose." |
| 6 | How it works | 45 | "Camera, line detection, then our own recogniser — a CRNN with CTC, 942,166 parameters, trained from scratch, 3.6 MB on the device. No OCR library is shipped; a build gate fails the build if one comes back. Then field extraction, then speech." |
| 7 | Measured results | 50 | **Lead with the honest row: "On a clean flat scan, Tesseract beats us — 72.9% line-exact against our 11.7%, and we say that first."** Then: "On a real hand-held capture we win by 5.5 times on word accuracy. And part of its lead is that our 37-symbol alphabet can't emit punctuation. On the clean page it read 500mg as 300mg. We'd rather say 'I can't read this' than be wrong on a dose." |
| 8 | What we got wrong | 40 | "We found four defects in our own train/serve contract. The width we served truncated 98.8% of lines. Padding wasn't inert in a bidirectional network. The image transform ran at serve and never at train. And our deskew was applied backwards — it was doubling the tilt. Each one now has a gate that fails the build." |
| 9 | Impact and policy | 40 | "This is already funded policy. The 2025 Policy Address — the edition the Open Challenge names — commits to AI carer support at Para 75, ophthalmic services at Para 227, and a working group on gerontechnology at Para 266. The 2026 edition followed through with $300 million and $100 million." |
| 10 | Close | 15 | "It reads it — or it says it cannot. Nothing leaves the phone. Thank you." |

## Which slide serves which judged criterion

| Criterion | Slide | The line that carries it |
|---|---|---|
| Creativity | 5 | refusal-over-guessing is a product decision, not a limitation |
| Innovation | 8 | four contract defects found and gated by the team that shipped them |
| Feasibility | 4, 6 | installs from a URL, works in airplane mode, no app store, no account |
| Potential Impact | 2, 3, 9 | named user, verified HK figures, and the scheme that already funds it |
| Interdisciplinarity | 1, 3, 9 | T (on-device ML) · E (image restoration) · S (vision science) · M (medication safety) · B (deployment via the Gerontechnology scheme) |
| Teamwork | 8, 10 | we audited our own numbers and published the ones that embarrass us |

## Judge-specific angles

- **Prof Shahid Khan (hepatology, healthcare innovation)** → lead with medication safety, the
  refusal behaviour and who is accountable. State plainly: **a phone app is not a regulated
  medical device; a licensed clinician is accountable, never the software.**
- **Dr Jackie Bell (computing, EDI)** → technical rigour. Have slide 8 ready to dwell on: the
  gates, the mutation-tested failure direction, the Tesseract head-to-head we didn't win.
- **Dr Pamela Tin (Research Director, HK Chief Executive's Policy Unit)** → the evidence and the
  policy anchoring. `docs/20_IMPACT_EVIDENCE.md` has every figure with its source and year; the
  definitional caveat on 199,600 is in `docs/13`. Offer the caveat before she finds it.
- **Google Hong Kong judge** → deployment path via the Gerontechnology Promotion Scheme and NGO
  partnerships. **Confirm the name first** — two official Imperial pages disagree (Helen Kwok
  vs Lucas Cheung); do not put a name on a slide until resolved (`docs/13`).

## Q&A — the eight we will be asked

1. **"Why not just use Tesseract?"** It's stronger on clean scans and we publish that. It's a
   second runtime plus 4.1 MB of language data (10.6 with osd); ours is one 3.6 MB graph that
   degrades instead of hallucinating, and on a real hand-held capture we read 68.4% word
   accuracy against its 12.4–23.6%.
2. **"Is it accurate enough?"** Not on arbitrary photographs, and we say so: 45.9% word accuracy
   end to end on our 12-document bench, 1.8% line-exact on 20 real hand-photographed pages. The
   design answer is the refusal. We publish per-condition numbers instead of one headline.
3. **"What happens if it reads a dose wrong?"** It refuses rather than guesses; it never presents
   a dose as authoritative; a clinician is accountable. The measured failure mode is why the
   document-evidence gate exists (speckle once hallucinated a plausible account number at 0.678
   confidence, inside the real-read band).
4. **"What about Chinese?"** The shipped alphabet is digits + A–Z + space: **zero Chinese
   characters today**, and we say that publicly. It's the top roadmap item — a data problem, with
   the pipeline, UI and offline guarantee unchanged.
5. **"Does it work on a real phone?"** It installs and runs on iOS and Android as a PWA, verified
   in WebKit emulation; **it has not yet been run on physical hardware**, which is our next step
   and the honest limit of what we can claim today.
6. **"Cantonese speakers?"** Same answer as 4 — the speech layer is ready; the recogniser's
   alphabet is the constraint.
7. **"Is this not just Seeing AI / Google Lens?"** Those upload the page to a server. Ours makes
   no network call after install — provable from the service worker in minutes, and by the
   airplane-mode demo.
8. **"Team across STEMB?"** Answer with the table above, and name who owns which discipline.

## Rehearsal checklist

- [ ] Demo phone charged, app installed from the live URL, **airplane mode verified the same day**
- [ ] Laptop with the local copy as fallback (`bash run.sh serve`), no venue Wi-Fi dependency
- [ ] Slides emailed 48 h ahead, and a copy on a USB stick and in the cloud
- [ ] The refusal case rehearsed until it lands as a design choice, not a failure
- [ ] Every number said aloud traced to a file (`docs/14`, `docs/24`, `docs/03`)
- [ ] Time it five times; the demo beat always runs long

## If the slot is 5 minutes (cut list)

Cut slide 6 to 20 s (pipeline in one sentence) and slide 9 to 20 s (name the paragraphs, don't
quote them), and drop slide 8's fourth defect. **Never cut the demo or the refusal.**
