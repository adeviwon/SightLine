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

## 150-word proposal (final draft — count checked at upload time)

> In Hong Kong, 199,600 people, 2.7% of the population, live with visual
> impairment. Most cannot read prescriptions, bank statements, or government
> letters, risking medication errors, financial fraud, and loss of
> independence. Existing apps upload them to the cloud. SightLine
> is a fully offline document reader that runs entirely on the phone: point
> it at any document and hear it read aloud: prescriptions,
> statements, labels. Nothing leaves the device: no internet, no cloud, no
> account, and it keeps working in airplane mode forever after one install.
> The recogniser is our own trained CRNN+CTC network (942,000 parameters),
> shipped as a 26.7 MB on-device bundle: no borrowed OCR engine, enforced by
> a build gate. Aligned with the 2026 Policy Address: AI in Healthcare
> (Para 108), AI for Welfare Lab (Para 113), Gerontechnology Promotion Scheme
> (Paras 398–400).

## Video beat sheet (target 3:30)

| time | beat |
|---|---|
| 0:00–0:25 | Mrs. Chan, 78, Wong Tai Sin, alone with a prescription she cannot read |
| 0:25–0:50 | The problem: 199,600 HK residents (C&SD 2021); cloud apps upload your bank statement and prescription. Offline is the only privacy answer |
| 0:50–1:00 | **Airplane mode ON — held on screen** |
| 1:00–2:20 | Live demo on the filming phone: install from `https://adeviwon.github.io/SightLine/`, scan a document → spoken result. Include one deliberate hard photo where the app **refuses** ("could not read that — please check with a pharmacist"): that is the designed behaviour, not a bug |
| 2:20–3:00 | Technical: our own CRNN+CTC recogniser trained from scratch, 942k params, 26.7 MB payload, no OCR library — a build gate fails if one returns. Measured on real hand-held photos: 71.6% line recall, 58.4% CER |
| 3:00–3:20 | Policy: Paras 108 / 113 / 398–400 — what the $400M should buy |
| 3:20–3:30 | Close: "Your eyes, offline. Nothing leaves your phone." + team |

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

## Checklist

- [ ] Proposal word-count confirmed ≤150, submitted by **2026-10-25**
- [ ] Video 3–5 min, filmed, 5 takes + screen-recording backup
- [ ] Filming phone: PWA installed from the live URL, airplane mode verified
- [ ] Every number on screen cross-checked against `docs/03_RESULTS.md`
- [ ] Grand Final prep begins after submission: slides ≤10, due **2026-11-12** (48 h before 2026-11-14)
