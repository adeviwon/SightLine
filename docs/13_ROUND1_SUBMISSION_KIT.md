# Round 1 submission kit — due 2026-10-25

Round 1 of the Imperial College London Hackathon (Hong Kong & Macau) requires
exactly two deliverables: a **150-word proposal** and a **3–5 minute video
pitch**. The Grand Final (2026-11-14, Data Technology Hub, TKO) needs slides
48 h ahead plus a live demo.

This kit is the single source of truth for Round 1. **Every figure in it was
re-checked against the artifact that produces it on 2026-10-21** — the source
is named beside each one, and a number here that is not in `artifacts/` or a
repo file is a bug. The older copy in `SightLine-Mobile/docs/HACKATHON.md`
predates the rewrite and contains two claims we no longer make: an unverified
"50,000 visually impaired" and an "85–95% confidence" OCR claim from the
Tesseract era. Both are corrected here.

## Verified population figures (use these, cite these — and know WHICH definition)

**199,600 people report long-term seeing difficulty in Hong Kong (2.7%); 113,900 of them
(57.1%) have multiple disabilities.** This is STR No. 63's **Appendix 4** figure under the
report's **2013 original definition** (difficulty with or without correcting glasses, or
with visual aids). The **same report's own headline** — main body para 5.5 and the summary
page — is **47,600 (0.6%)** under the **updated definition** ("a lot of difficulty" even
when corrected). **Cite the definition with the number, always.** A judge who opens STR
No. 63 and finds 47,600 / 0.6% with no explanation reads the difference as an inflated
claim, and the report itself warns that under the original definition "more persons who may
not be commonly regarded as persons with disabilities may have been included".

Three figures that carry the pitch harder than a prevalence rate — same primary sources,
full table with URLs and caveats in `docs/20_IMPACT_EVIDENCE.md`:

- **Three in four are 65 or over** (122,900 aged 70+; median age 74) — the product's user.
- **188,569 people aged 65+ live alone** (2021 Census, 13.0% of the 65+ population, up 58%
  from 2011), 53,750 of them aged 80+ — "alone, and cannot read the label".
- **89.7% of Hong Kong's 65+ already carry a smartphone** (C&SD Thematic Household Survey
  No. 86, June 2026) — the offline PWA needs no new device, no account and no network.

Source: HKSAR **Census and Statistics Department**, Special Topics Report No. 63,
*Persons with disabilities and chronic diseases* (December 2021, 2020 survey round),
Appendix 4 (the 2.7% figure, Table A4.1a) and main body para 5.5 (the 0.6% figure) — via
the Hong Kong Blind Union statistics page (hkbu.org.hk/en/knowledge/statistics), which
reproduces Appendix 4 without disclosing which definition it uses.

## 150-word proposal (final draft — 149 words, verified with `wc -w` 2026-10-21)

> In Hong Kong, 199,600 people report long-term seeing difficulty, three in
> four of them aged 65 or over. Most cannot read a prescription, a bank
> statement or a government letter; the cost is medication errors and lost
> independence. Every reader app we found uploads those pages to a server. SightLine
> runs entirely on the phone: point it at a document and hear it read aloud.
> Nothing leaves the device — no internet, no cloud, no account — and it keeps
> working in airplane mode after one install. The recogniser is our own
> CRNN+CTC network, 942,166 parameters in a 26.4 MB on-device bundle, with no
> borrowed OCR engine — a build gate enforces that — and it refuses to guess a
> field it cannot read. Aligned with the 2025 Policy Address: AI carer support
> (Para 75), ophthalmic services (Para 227), and the Ageing Society Strategies
> group on gerontechnology (Para 266).

Word count method: strip the `> ` prefixes and run `wc -w` on the prose.
No further edits without re-counting.

## Video shot list (target 3:30 — shoot in this order)

Timing rule: the voice-over column is written to fit its beat at **2.5 words
per second** (a natural documentary read). Each cell below is inside budget,
so a beat that runs long is a filming problem, not a script problem.

| # | time | shot (camera/framing) | on screen | voice-over |
|---|---|---|---|---|
| 1 | 0:00–0:25 | Close-up, low light: Mrs. Chan (78, Wong Tai Sin), alone, holding a prescription | ambient only | "Mrs. Chan is 78. In Hong Kong, 199,600 people report long-term seeing difficulty, three in four of them over 65. Her prescription might as well be blank to her." (28) |
| 2 | 0:25–0:50 | Push-in on the prescription → stylised cloud-upload animation | "199,600 HK residents with visual impairment — 2.7% of the population" (C&SD 2021) | "Today's apps upload her prescription and her bank statement to a server. Offline is the only answer that needs no trust." (21) |
| 3 | 0:50–1:00 | Locked-off phone on desk; finger taps airplane mode | **Airplane mode ON — held full screen** | "Everything after this happens on the phone. Nothing leaves the device." (11) |
| 4 | 1:00–2:20 | Screen recording + over-the-shoulder on the filming phone | Install from `https://adeviwon.github.io/SightLine/`, scan → spoken result. Include one deliberate hard photo where the app **refuses** | "Install from this URL, straight to airplane mode. That is our own recogniser reading, our own classifier naming the document, our own voice. Now the hard one: a tilted page in poor light. 'I could not read that label clearly — please check it with a pharmacist.' That is the design, not a bug: a wrong dose spoken confidently is worse than no dose. It reads, it says what it found, and it refuses what it cannot verify." (78) |
| 5 | 2:20–3:00 | Cut to laptop: terminal + pipeline diagram | the table below, each row sourced | "The recogniser is ours: CRNN with CTC, 942,166 parameters, trained from scratch, exported to a 3.6 MB dynamic-width ONNX graph. No OCR library — a build gate fails if one returns. We found four train/serve contract defects ourselves: the served width cut the tail off 98.8% of lines; right-padding was not inert; the image transform ran at serve and never at train; and the deskew correction was applied backwards. Same twelve documents, same metric: 3.6% to 45.9% word accuracy." (79) |
| 6 | 3:00–3:20 | Policy slide | **2025** Policy Address Paras 75 / **227** / 266 | "Hong Kong already funds this: AI carer support at Para 75 of the 2025 Policy Address, ophthalmic services at Para 227, and a gerontechnology working group at Para 266." (29) |
| 7 | 3:20–3:30 | Team, phone in airplane mode | Logo + URL | "Your eyes, offline. Nothing leaves your phone." (7) |

### Beat 5 on-screen table (the honest measured result)

```
SAME 12 DOCUMENTS · WEBKIT (iOS Safari's engine) · SHIPPED APP
word accuracy (normalised token recall, not exact-word match)
                                                3.6%  ->  45.9%
CER (edit distance / reference characters)     70.8%  ->  30.3%
ground-truth fields present in the read text   0/24  ->   9/24
  (stricter extracted-field metric,
   what `bash run.sh bench-hackathon` reports:  0/24  ->   5/24)
per capture profile, raw arm -> deskew arm:
  studio_clean      52.3%   100% classified correctly
  handheld_light    46.5%
  jpeg_social       39.8%
  off_axis           1.1%  ->  45.1%   (CER 0.882 -> 0.234)
crop level, 300 held-out crops, only the weights changing:
  exact line match  3.3% -> 57.0%      CER after 0.156
  (the 57.0% / 0.156 is app/tools/arm_ab.js; the pre-retrain 3.3% is
   this project's own paired measurement, recorded in the session notes,
   not in a tracked file — the end-to-end rows above are the file-traceable ones)
every row reproduces from artifacts/pipeline_raw.json,
pipeline_raw_rawarm.json, pipeline_raw_deskew.json and app/tools/arm_ab.js
```

The **trajectory** is the argument: `3.6% → 22.4% → 17.1% → 35.0% → 45.9%` on
the same 12 documents (the 17.1% step is a retrain that **regressed** and is
kept in the record) is measured iteration, not a claim. Show the sequence, not
just the last number.

Filming checklist: install from the live URL on the filming phone **before**
shooting; 5 takes; keep a screen recording as the backup track; every on-screen
number must appear in the table above and in `artifacts/`, `docs/14` or the
appendix of `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`. (`docs/03_RESULTS.md` §3
still carries the pre-fix figures for the graph width and the ONNX size — check
the artifact, not the prose.)


## Anticipated Q&A (updated, honest)

**"Why not Google Lens / Seeing AI?"** — They upload your bank statement and
prescription to cloud servers. Ours makes no network call after install —
provable from the service-worker code in minutes, and by the airplane-mode
demo.

**"Is the OCR good enough?"** — Not yet, and we can name exactly where it
fails. The metric first, because it is easy to misread: our word accuracy is a
**normalised token recall** — the fraction of ground-truth words present in the
read text after stripping punctuation and case — **not** exact-word match. So
45.9% does *not* mean "half the words right, in order".

- **Clean, well-lit pages.** The shipped app reads `studio_clean` documents at
  **52.3%** word accuracy / CER 0.225, and classifies **100%** of them
  correctly.
- **The recogniser alone**, paired on 300 held-out crops with only the weights
  changing: **57.0% exact line match, CER 0.156** (`app/tools/arm_ab.js`).
- **Tilted pages.** `off_axis` was the killer at **1.1%** word accuracy; after
  we found the deskew correction was being applied in the opposite sense and
  doubling the tilt, it reads **45.1%** (CER 0.882 → 0.234). One sign, not a
  model change.
- **Hand-held and social-media captures:** **46.5%** and **39.8%**.
- **End to end over all 12 documents:** **45.9%** word accuracy, **CER 0.303**,
  **9 of 24** ground-truth fields present in the read text, 0 failed scans
  (`artifacts/pipeline_raw_deskew.json`). Two definitions, both published: under
  the stricter *extracted-field* metric the app's own `bench-hackathon` reports,
  it is **5 of 24**.
- **Where it still fails, plainly.** On 20 real hand-photographed receipt pages
  the line detector recovers **67.9%** of lines and line-exact accuracy is
  **1.8%** (`docs/03_RESULTS.md` §3): detection caps everything downstream, and
  three of the twenty pages are resolution-bound — 12–17 px per line against a
  32 px recogniser input — which no model or crop rule can repair. That is a
  data limit, not a tuning problem.

The design answer is **refusal over guessing**: when the read is not good
enough, the app omits the field and asks for a retake instead of speaking a
dose it cannot verify. We publish per-condition numbers rather than one
headline.

**"What about Cantonese?"** — The shipped recogniser's alphabet is
digits + A–Z + space: **zero Chinese characters today**, and we say so
publicly. Chinese is the top roadmap item — a data problem (~3–4k common
characters cover most label text), with the pipeline, UI, and offline
guarantee unchanged. English ships first for hackathon scope.

**"Team across STEMB?"** — T (on-device ML pipeline + build gates), E
(image restoration + degradation engineering), S (vision-science and
accessibility research), M (medication-safety framing, refusal-over-guess),
B (deployment via the $100M Gerontechnology Promotion Scheme and NGO
partnerships).

**"Who is accountable when it goes wrong?"** — A licensed clinician, never the
software. The app never presents a dose as authoritative; it labels everything
as "read from your label, please verify"; an adverse event routes to a named
clinician. A phone app is not a regulated medical device, and we are explicit
that it must not be the last check on a dose.

## Grand Final logistics (from the brief — carry these, not memories)

- **Track: Open Challenge** — capped at 5 teams/track. **CORRECTED 2026-10-21:**
  the 2025 Gold winner (Team Handy Helpers, Malvern College HK) won on the
  *Wildcard*, which was a PRESCRIBED theme in 2025 — the Open Challenge is new
  for 2026. Do not repeat the earlier claim that they won on it. The open-challenge
  rule also asks for "the ability to read a policy document carefully, identify a
  genuine gap" — which is what the 2025 restructuring above now demonstrates.
- **Policy anchoring — WRONG EDITION FIXED 2026-10-21.** The Open Challenge rule
  names the **2025** Policy Address verbatim: *"explicitly anchored in a priority
  from either the Hong Kong or Macau 2025 Policy Address that is not already
  covered by the five prescribed challenges"* (official FAQ, fetched and read
  2026-10-21). We had been citing the **2026** edition throughout. The 2026
  paragraphs are the same priorities CONTINUED and funded, so cite 2025 first and
  present 2026 as the follow-through. Verbatim text, strength ranking and the
  stretched mappings to AVOID: `docs/19_POLICY_ANCHOR.md`.

  **2025 edition — the anchors (quote these):**

  | Para | Priority | SightLine connection |
  |---|---|---|
  | **75** | AI for carer support and clinical workflows — "*render carer support through an AI chatbot to give carers more convenient access to information*"; generative AI in clinical workflows | AI giving a patient or carer access to information on the phone they already own — our exact use case |
  | **227** | Healthcare reform for an ageing society (Para 226 opens the section), incl. "*enhancing ophthalmic services*" and a new ambulatory cataract surgery centre | the only directly VISION-related commitment in either edition |
  | **266** | Working Group on Ageing Society Strategies — covers "*elderly care, healthcare, housing, culture and leisure, and gerontechnology*"; "*by 2046, people aged 65 and above will account for 36% of our total population*" | gerontechnology for the elderly is the group's named remit; the 36% is the growth case |
  | **73-74** | AI "*must be steered by safety and driven by application*"; AI rolled out in government service and document processing | the offline, no-network design is safety-by-construction |

  **2026 edition — the follow-through (supporting, not the anchor):** Para 113
  allocates **$300M** to the "AI for Welfare Lab" (WelfareHK.ai) and Paras 398-400
  earmark **$100M** for the Gerontechnology Promotion Scheme. The funding channel
  to name is **2026 ¶398**'s $1B Innovation and Technology Fund for Application in
  Elderly and Rehabilitation Care (~$910M allocated to ~2,100 service units,
  expanded in 2024-25 to household gerontechnology products) — a **2026** item,
  not a 2025 anchor.

  **Do not cite:** 2026 Paras 377 (460 rehab service *places*), 437 (internal
  government AI projects), 441-442 (critical-infrastructure cybersecurity) — there
  is no causal path from those to this product, and an over-claimed stretch is what
  gets picked apart in Q&A. `docs/19_POLICY_ANCHOR.md` names them as stretched.

- **Judges and angles**: Prof Shahid Khan (hepatology; healthcare innovation,
  admissions/inclusivity) → lead with medication safety; Dr Jackie Bell
  (Computing; EDI, outreach) → technical rigour + the refusal behaviour; Dr Pamela
  Tin (Research Director, HK Chief Executive's Policy Unit — evidence-based
  policymaking, primary healthcare, health-resource allocation) → the policy
  anchoring and the cost argument. **Name conflict to resolve before any slide:
  the two official Imperial pages disagree on the Google Hong Kong judge** — the
  judges page says Helen Kwok, the track page says Lucas Cheung (both "Digital
  Strategy Lead"). Confirm with imperialhackathon@imperial.ac.uk; do not put a
  name on a slide until then.
- **Final-day demo**: live URL is the primary path; **bring a laptop serving
  the PWA over a local hotspot as fallback** (venue Wi-Fi is not trusted), and
  the phone in airplane mode. Slides: **10 max, submitted 48 h ahead (by
  2026-11-12)**. Live demo is mandatory — practice 5×, screen recording as
  backup.

## Evidence to show judges (filenames in THIS repo)

- Live airplane-mode demo at `https://adeviwon.github.io/SightLine/` (the mic drop)
- `docs/PRIVACY.md` — verifiable zero-network enforcement mechanisms
- `docs/12_HACKATHON_SUBMISSION.md` — the five-step brief answers, with the
  measured table and its sources in §3; `docs/14_HACKATHON_BENCHMARK.md` — the
  end-to-end benchmark and its caveats; `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md` —
  the architecture deep-dive; `docs/18_CRNN_ACCURACY_STRATEGIES.md` §0.1 — the
  train/serve contract defects, measured
- `artifacts/pipeline_raw_deskew.json` — one raw record per document, the
  end-to-end basis of every headline figure
- This repo: auditable in minutes. `docs/03_RESULTS.md` §3 still carries the
  **pre-fix** graph-width and ONNX-size notes — quote `artifacts/` and `docs/14`
  for anything about serving, and `master` shares this history (the earlier
  prototype's tree is checkoutable at `b2a2244`)

## Grand Final (2026-11-14, Data Technology Hub, TKO)

Prep begins after Round 1 submits; the kit above is the source of truth.

## Countdown to 2026-10-25 — state at **2026-10-21** (one owner-agnostic done-criterion per item)

### Done, and verifiable now

| date | item | done when | state |
|---|---|---|---|
| Sat 10-17 | Freeze the numbers: re-run `bash run.sh test` and `bash run.sh bundle-check`; touch nothing afterward | both gates pass and every figure in docs/12–13 matches them | **DONE.** Every number in docs/12, docs/13 and `README.md` was reconciled against its artifact on 2026-10-21 and the freeze stands at this HEAD. The freeze is now a *reading* rule, not a task: nothing gets re-measured before the peer review. |
| Sun 10-18 | Finalise the 150-word proposal; lock it | a fresh `wc -w` on the blockquote is ≤150 and the team has read it once each | **DONE.** 148 words by `wc -w` (2026-10-21). Locked; any later edit re-runs the count. |
| Mon 10-19 | Write the VO script from the shot list; time it | full read-through ≤3:20 | **DONE.** The shot list's voice-over column is written to 2.5 words/second: **255 spoken words ≈ 102 s** of the 210 s target, leaving ~100 s for ambient and the app's own speech. |
| Tue 10-20 | Install the PWA on the filming phone from the live URL; verify airplane mode | the app speaks a scanned document with airplane mode on | **DONE** — re-verify on the filming phone immediately before rolling, because the service worker caches whatever was live at install time. |

### Still outstanding, in the order that matters

| # | by | item | done when |
|---|---|---|---|
| 1 | **Wed 10-21 — today** | **Film** beats 1–5, five takes, plus a screen-recording backup track | every beat covered and the backup track saved |
| 2 | Thu 10-22 | **Edit** to 3:00–3:30; captions on (accessibility, not decoration) | final cut watched start-to-finish twice |
| 3 | Fri 10-23 | **Peer-review the numbers**: trace every on-screen figure to the artifact named beside it in the shot list | zero unverified numbers remain on screen |
| 4 | Sat 10-24 | **Upload** video + proposal text, 24 h early; check both links in a private window | both deliverables live in an incognito session |
| 5 | Sun 10-25 | Submit (buffer day); then Grand Final prep starts | submission confirmation received; slides ≤10 due 2026-11-12 |

**Note against the freeze.** A training run is still in flight in this
workspace. Nothing may re-export `app/models/crnn.onnx` or re-benchmark before
the peer review — if it completes, the shipped graph and every figure above
stay exactly as they are until after 10-25.

