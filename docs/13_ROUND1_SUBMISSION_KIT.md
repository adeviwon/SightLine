# Round 1 submission kit — due 2026-10-25

Round 1 of the Imperial College London Hackathon (Hong Kong & Macau) requires
exactly two deliverables: a **150-word proposal** and a **3–5 minute video
pitch**. The Grand Final (2026-11-14, Data Technology Hub, TKO) needs slides
48 h ahead plus a live demo.

This kit is the single source of truth for Round 1. The older copy in
`SightLine-Mobile/docs/HACKATHON.md` predates the rewrite and contains two
claims we no longer make: an unverified "50,000 visually impaired" and an
"85–95% confidence" OCR claim from the Tesseract era. Both are corrected here.

## Verified population figure (use this, cite this)

**199,600 people with visual impairment in Hong Kong — 2.7% of the
population.** 113,900 of them (57.1%) have multiple disabilities.

Source: HKSAR **Census and Statistics Department**, Special Topics Report
No. 63, *Persons with disabilities and chronic diseases* (December 2021,
2020 survey round), Appendix 4 — via the Hong Kong Blind Union statistics
page (hkbu.org.hk/en/knowledge/statistics).

## 150-word proposal (final draft — 140 words; re-count with `wc -w` at upload)

> In Hong Kong, 199,600 people — 2.7% of the population — live with visual
> impairment. Most cannot read prescriptions, bank statements, or government
> letters; the cost is medication errors, financial fraud and lost
> independence. Existing apps upload those documents to a cloud server.
> SightLine runs entirely on the phone: point it at a document and hear it
> read aloud. Nothing leaves the device — no internet, no cloud, no account —
> and after one install it keeps working in airplane mode. The recogniser is
> our own trained CRNN+CTC network (942,000 parameters) in a 26.7 MB
> on-device bundle — no borrowed OCR engine, enforced by a build gate — and
> it refuses to guess a field it cannot read. Aligned with the 2026 Policy
> Address: AI in Healthcare (Para 108), AI for Welfare Lab (Para 113),
> Gerontechnology Promotion Scheme (Paras 398–400).

## Video shot list (target 3:30 — shoot in this order)

| # | time | shot (camera/framing) | on screen | voice-over |
|---|---|---|---|---|
| 1 | 0:00–0:25 | Close-up, low light: Mrs. Chan (78, Wong Tai Sin), alone, holding a prescription | ambient only | "Mrs. Chan is 78. This label might as well be blank." |
| 2 | 0:25–0:50 | Push-in on the prescription → stylised cloud-upload animation | "199,600 HK residents with visual impairment" (C&SD 2021) | The problem: cloud apps upload your prescription and bank statement. Offline is the only privacy answer. |
| 3 | 0:50–1:00 | Locked-off phone on desk; finger taps airplane mode | **Airplane mode ON — held full screen** | "Everything after this happens on the phone." |
| 4 | 1:00–2:20 | Screen recording + over-the-shoulder on the filming phone | Install from `https://adeviwon.github.io/SightLine/`, scan → spoken result. Include one deliberate hard photo where the app **refuses**: "could not read that — please check with a pharmacist" — that is the designed behaviour, not a bug | Narrate what is heard; land the refusal as the safety design |
| 5 | 2:20–3:00 | Cut to laptop: terminal + pipeline diagram | Numbers on screen, each cross-checked against `docs/03_RESULTS.md` | Our own CRNN+CTC recogniser trained from scratch, 942k params, 26.7 MB payload, no OCR library — a build gate fails if one returns. Measured on real hand-held photos: 71.6% line recall, 58.4% CER |
| 6 | 3:00–3:20 | Policy slide | Paras 108 / 113 / 398–400 | What the schemes should buy |
| 7 | 3:20–3:30 | Team, phone in airplane mode | Logo + URL | "Your eyes, offline. Nothing leaves your phone." |

Filming checklist: install from the live URL on the filming phone **before**
shooting; 5 takes; keep a screen recording as the backup track; every on-screen
number must appear in `docs/03_RESULTS.md` or the appendix of
`docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`.

## Anticipated Q&A (updated, honest)

**"Why not Google Lens / Seeing AI?"** — They upload your bank statement and
prescription to cloud servers. Ours makes no network call after install —
provable from the service-worker code in minutes, and by the airplane-mode
demo.

**"Is the OCR good enough?"** — On clean, well-lit documents the recogniser
reads held-out receipt text at ~29% exact-word and 11.95% CER. On hard
hand-held photos the ceiling is the line detector: 71.6% of lines found, 58.4%
CER on found lines. The design answer is the refusal behaviour: when input is
unreadable the app says so and asks for a retake instead of guessing a dose.
We publish per-condition numbers rather than one headline.

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

- **Track: Open Challenge** — capped at 5 teams/track (lowest-competition
  route; the 2025 Gold winner, Team Handy Helpers from Malvern College Hong
  Kong, won on this track).
- **Full policy-anchor table** — five verbatim paragraphs from the 2026 Policy
  Address (delivered 2026-09-16). The proposal cites three; the pitch and Q&A
  can use all five:

  | Para | Priority | SightLine connection |
  |---|---|---|
  | 108 | AI Application in Healthcare | on-device AI reads prescriptions/dosages aloud |
  | 113 | AI for Welfare Lab ($300M) | exactly the profile of AI-welfare solution the lab funds |
  | 398-400 | Gerontechnology Promotion Scheme ($100M) | gerontechnology product for elderly with vision loss |
  | 376-377 | Support Persons with Disabilities & Carers | independent document reading without a carer |
  | 441-442 | Cybersecurity & digital resilience | zero-network architecture is secure by design |

- **Judges and angles**: Prof Shahid Khan (healthcare) → lead with medication
  safety; Dr Jackie Bell (computing/EDI) → technical rigour + the refusal
  behaviour; Lucas Cheung (Google HK, digital strategy) → the deployment path
  via the $100M scheme and NGO partnerships.
- **Final-day demo**: live URL is the primary path; **bring a laptop serving
  the PWA over a local hotspot as fallback** (venue Wi-Fi is not trusted), and
  the phone in airplane mode. Slides: **10 max, submitted 48 h ahead (by
  2026-11-12)**. Live demo is mandatory — practice 5×, screen recording as
  backup.

## Evidence to show judges (filenames in THIS repo)

- Live airplane-mode demo at `https://adeviwon.github.io/SightLine/` (the mic drop)
- `docs/PRIVACY.md` — verifiable zero-network enforcement mechanisms
- `docs/01_ARCHITECTURE.md` + `docs/03_RESULTS.md` — pipeline and every
  measured number; `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md` — the architecture
  deep-dive; `docs/12_HACKATHON_SUBMISSION.md` — the five-step brief answers
- This repo: auditable in minutes; `master` holds the earlier prototype,
  preserved untouched

## Grand Final (2026-11-14, Data Technology Hub, TKO)

Prep begins after Round 1 submits; the kit above is the source of truth.

## Countdown to 2026-10-25 (one owner-agnostic done-criterion per item)

| date | item | done when |
|---|---|---|
| Sat 10-17 | Freeze the numbers: re-run `bash run.sh test` and `bash run.sh bundle-check`; touch nothing afterward | both gates pass on `main` and every figure in docs/12–13 matches them |
| Sun 10-18 | Finalise the 150-word proposal (§ above); lock it | a fresh `wc -w` on the blockquote is ≤150 and the team has read it once each |
| Mon 10-19 | Write the VO script from the shot list; time it | full read-through ≤3:20 (10 s spare) |
| Tue 10-20 | Install the PWA on the filming phone from the live URL; verify airplane mode | the app speaks a scanned document with airplane mode on |
| Wed 10-21 | Film shots 1–5, five takes, plus a screen-recording backup track | every beat covered and the backup track saved |
| Thu 10-22 | Edit to 3:00–4:30; add captions (accessibility, not decoration) | final cut watched start-to-finish twice, captions on |
| Fri 10-23 | Peer review: trace every on-screen number to `docs/03_RESULTS.md` or the appendix of `docs/11` | zero unverified numbers remain on screen |
| Sat 10-24 | Upload video + proposal text; check links in a private window | both deliverables live in an incognito session, 24 h before the deadline |
| Sun 10-25 | Submit (buffer day); then Grand Final prep starts | submission confirmation received; slides ≤10 due 2026-11-12 |
