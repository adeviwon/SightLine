# 23 — Pillar gap analysis: where the marks are actually being left

**Purpose.** Round 1 (150-word proposal + 3–5 minute video, due **2026-10-25**)
and the Grand Final (**2026-11-14**, Data Technology Hub, Tseung Kwan O InnoPark,
HKSTP) are judged on six published words. This file says, pillar by pillar,
what we have, what a judge will find missing, and the cheapest thing that fixes
it. It is an analysis, not a build: nothing outside this file was changed.

**How to read it.** Every external claim carries a URL. Anything I could not
confirm is marked **UNVERIFIED** in §7 and never repeated as fact. Anything that
is our own number carries the repo file it comes from.

---

## 1. What is actually on the record

### 1.1 The judging criteria, verbatim

> "Your pitch will be evaluated on its **creativity, innovation and feasibility**,
> as well as its **potential impact, interdisciplinarity** and **how you've
> worked as a team**."

Source: <https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/faqs/>

The same page adds: "Register to receive full details of the judging criteria
for this year." **We do not have the weightings.** They are only issued to
registered participants. So the six criteria are known to be scored; their
relative weight is not.

The six headings the organizers themselves use in the FAQ map to the six pillars
in the brief we were given. Note the ordering — impact and interdisciplinarity
are named after feasibility, and teamwork is named last and separately, which is
where most teams forget it exists.

### 1.2 Round 1 vs Grand Final — what is actually judged when

| stage | date | deliverables | who reviews |
|---|---|---|---|
| Round 1 | submit by 23:59 HKT/MST **2026-10-25** | 150-word written proposal + 3–5 min video pitch | "Imperial academics and industry leaders" (no demo, no slides, no live anything) |
| Grand Final | **2026-11-14**, in person, HK | emailed slide deck (pre-loaded on a shared laptop) + live demo | judging panel + industry experts |

Sources: FAQ page (above); <https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/university-professional/>

Two logistics facts that decide the Grand Final and are load-bearing:

- Slides are **pre-loaded onto a shared presentation laptop** — so the deck must
  survive not having our fonts, our aspect ratio, or our clicker. A live demo
  requires **bringing our own laptop**, and the FAQ says: "We also recommend
  testing any demos beforehand and having a **backup plan in case of technical
  issues**."
- Finalist pitches must be **in person**; "remote participation is not
  available". Every team member presenting must be able to attend.
  Source: FAQ page.

### 1.3 The panel (2026) — four named judges, not three

| judge | role | what they will care about |
|---|---|---|
| **Professor Shahid Khan** | Professor of Practice (Hepatology) & Consultant Hepatologist; admissions, inclusivity, healthcare innovation | medication safety, clinical harm, who is accountable |
| **Dr Jackie Bell** | Senior Teaching Fellow in Computing; EDI, outreach, STEM engagement | technical rigour, whether we understand our own model, accessibility |
| **Helen Kwok** | Digital Strategy Lead, Google Hong Kong; go-to-market, cross-border commercial strategy | deployment path, funding route, scale |
| **Dr Pamela Tin** | Research Director, **Hong Kong Chief Executive's Policy Unit**; evidence-based policymaking, primary healthcare, health-resource allocation, mental health | **the policy anchor and the evidence base** |

Sources: <https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/judges-mentors/>
and <https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/university-professional/>

**Two things this changes.** (a) `docs/13_ROUND1_SUBMISSION_KIT.md` names three
judges and gets one wrong — it says "Lucas Cheung (Google HK)", while the
judges page says **Helen Kwok** for that same role. Do not put a judge's name on
a slide until that is resolved (see §7). (b) **Dr Pamela Tin is missing from our
entire repo.** A Research Director at the CE's Policy Unit is the judge who will
test whether our policy citation is real. It now holds up to that reading — the
anchor cites the **2025** edition the Open Challenge rule names (corrected
2026-10-21; see §1.6).

### 1.4 Prizes — and the detail nobody notices

| award | what it is |
|---|---|
| Gold | HKD 15,000 team cash + trophy + gold medal each + **letter of recommendation signed by an Imperial academic judge** + featured profile on Imperial's Global Hong Kong platform |
| Silver | HKD 1,500 Amazon voucher **per member** + trophy + medals |
| Bronze | HKD 1,000 Amazon voucher per member + trophy + medals |
| Specialist awards | open to all **seven** Grand Final teams; "presented to teams **not placed in the top three**" |

Specialist awards named on the page: **Best translational research** (research →
real-world pathway), **Best innovation** (original, creative, inventive thinking),
**Greatest social impact** (meaningful and lasting difference to communities in
Hong Kong or Macau), **Best STEMB integration** (science, technology,
engineering, medicine, business genuinely integrated), **Best sustainability**.

Source: <https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/university-professional/prizes-awards/> and the FAQ.

Two consequences worth acting on:

1. **"No team goes home empty handed"** — every one of the seven Grand Final
   teams gets a podium or a specialist award. This is a competition where
   *reaching the final is itself most of the prize*, so the marginal return on
   Round 1 effort is enormous.
2. There are **five specialist award names and the page says "four specialist
   awards"**. Flagged in §7. Even on the narrow reading there is no
   "*best teamwork*" award — teamwork is scored inside the six criteria, not
   as a separate prize, which is exactly why it gets forgotten.

### 1.5 The 2025 edition, and the winners we can actually read

**2025 was a secondary-school-only competition.** The 2025 eligibility was
grades 10–12 living in Hong Kong
(<https://www.imperial.ac.uk/study/visit/events/imperial-hackathon/faqs/>).
The 2025 Grand Finale ran 22 November 2025 at HKSTP, and the four 2025 challenges
were: Sky-High Solutions (low-altitude delivery), Silver Tech for Golden Years
(elderly well-being), From Waste to Resource (circular economy), and **Wildcard:
Your idea, your impact**
(<https://www.imperial.ac.uk/events/202423/grand-finale-the-imperial-hackathon-2025/>).

> **Correction to our own assumption.** `docs/13` says the Open Challenge is the
> "lowest-competition route; the 2025 Gold winner, Team Handy Helpers from
> Malvern College Hong Kong, won on this track." The winner fact is right; the
> inference is wrong. 2025's Wildcard was **one of four prescribed challenges**,
> not the 2026 Open Challenge, and it was won by a **secondary-school** team.
> The 2026 university/early-career track and the Open Challenge are both **new
> this year**. Nothing in 2025 evidences that a university Open Challenge entry
> wins.

2025 winners (all secondary school)
(<https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/imperial-hackathon-2025-winners/>):

| award | team | school | challenge |
|---|---|---|---|
| Gold | Handy Helpers | Malvern College HK | Wildcard: Your idea, your impact |
| Silver | PING | Chinese International School | Silver Tech for Golden Years (elderly well-being) |
| Bronze | DefAn | Sha Tin College | From Waste to Resource |
| Best Real-World Impact | The 5 Big Friendly Giants | Kellett School | Wildcard |
| Best Technical Solution | YC Walkers | Yew Chung International School | Silver Tech for Golden Years |
| Most Innovative Concept | D-Cubed Tiles | International Christian School | Silver Tech |
| Best Presentation/Pitch | Helios | German Swiss International School | Sky-High Solutions |
| Best Startup Star | YCIS Bloomwhisperers | Yew Chung International School | Wildcard |
| Community Choice | Aegis | King George V School | Sky-High Solutions |

**Scale:** the 2025 Hong Kong edition is described in a sponsor-side LinkedIn post
as having "80 brilliant teams"
(<https://www.linkedin.com/posts/garyleung5398_imperialhackathon-imperial-icaahk-activity-7394693923660673024-cGam>
— UNVERIFIED, unofficial). The 2024 edition was won by Team Lambda, Independent
Schools Foundation Academy, quoted on
<https://www.imperial.ac.uk/study/visit/events/global-hackathons/>.

**What the winners' projects actually were, where the record exists:**

- Two of the three overall 2025 winners sat on the **elderly well-being** and
  **elderly-adjacent** challenges. Health and ageing is what this competition
  rewards.
- **Helios** — Best Presentation — is described by a team member as "an
  AI-powered smart drone fleet for emergency medical intervention…
  in the form of **simulated prototypes** and a mobile application"
  (<https://www.linkedin.com/in/nicholas-chan-7a9388336>, UNVERIFIED, self-reported).
  A **concept-level** project won a specialist award on presentation. Storytelling
  is scored here, and it is scored against teams that built nothing.
- The Gold winner's project description is **UNVERIFIED** — see §7. The closest
  match found (`https://expatliving.net/hong-kong/role-models-from-hong-kong-sevens-and-philharmonic/`)
  attributes a "personalised arthritis assistance glove" to a Malvern team called
  "Handy Helpers" winning gold at the *Global Natural Science and Innovation
  Competition* U16 at Imperial — a different competition that the hackathon
  feeds into, not the hackathon itself. Same school, same team name; treat as
  likely-but-unconfirmed.

### 1.6 The Open Challenge rule — and a citation problem in our own proposal

The Open Challenge is real, available to both tracks, **capped at five teams per
track**, and explicitly framed as harder, not easier ("The open challenge is not
an easier option. It is a harder one.").

The one rule that cannot be broken, as published:

> "The open challenge allows teams to identify a problem of their own choosing,
> provided it is explicitly anchored in a priority from either the Hong Kong or
> Macau **2025 Policy Address** that is not already covered by the five
> prescribed challenges."

Sources: the FAQ page (§1.1 URL) and a copy of the full university-track brief
retained by another 2026 entrant at
<https://github.com/Peeradonn/Imperial/blob/main/Imperial%20Sources/Challenge%20Briefs.md>
(third-party transcription — see §7). The brief requires the problem statement to
"**quote directly from the relevant policy document**".

**The finding (and the correction, applied 2026-10-21).** Our Round 1 proposal
originally cited the **2026** Policy Address: "Aligned with the 2026 Policy
Address: AI in Healthcare (Para 108), AI for Welfare Lab (Para 113),
Gerontechnology Promotion Scheme (Paras 398–400)." The rule names the **2025**
edition, so the proposal now leads with the 2025 anchors — **¶75** (AI that gives a
carer access to information), **¶227** (ophthalmic services) and **¶266** (Ageing
Society Strategies / gerontechnology) — and presents the 2026 paragraphs as the
funded continuation. Verbatim text of both editions: `docs/19_POLICY_ANCHOR.md`.

I checked what is actually in each edition:

- **The $300M "AI for Welfare Lab" is a 2026 item** (¶113), so it belongs on the
  slide only as the *funding continuation* of the 2025 priority:
  "The Chief Executive announced in the 2026 Policy Address a funding allocation
  of $300 million to set up 'AI for Welfare Lab' (WelfareHK.ai)" —
  <https://www.lwb.gov.hk/en/welfarehk_ai.html>.
- **In the 2025 Policy Address the paragraph numbering is different.** The 2025
  text puts AI and data science at **paras 66–76** ("Expand the Application of AI"
  = 73–76) and healthcare at **paras 226–234** —
  <https://www.policyaddress.gov.hk/2025/en/policy.html>. Paras 108/113 therefore
  do not exist as those items in the 2025 edition.
- **The 2025 anchors we now use** (verified verbatim against the 2025 full PDF,
  SHA-256 `c4ecaf…18f3ec`, in `docs/19_POLICY_ANCHOR.md`): **¶75** — AI to "render
  carer support through an AI chatbot to give carers more convenient access to
  information" and generative AI in clinical workflows
  (<https://www.policyaddress.gov.hk/2025/en/p73.html>); **¶227** — "enhancing
  ophthalmic services … a new high-capacity ambulatory cataract surgery centre"
  (under the "(D) Healthcare" heading that opens at ¶226); **¶266** — the Working
  Group on Ageing Society Strategies, whose remit includes **gerontechnology**, and
  the "By 2046 … 36%" figure
  (<https://www.policyaddress.gov.hk/2025/en/p265.html>); and **¶73–74** — AI
  "steered by safety and driven by application".
- **Two corrections to our own earlier notes.** The $1B Innovation and Technology
  Fund for Application in Elderly and Rehabilitation Care is a **2026** item (¶398),
  *not* a 2025 one — the 2025 text mentions "gerontechnology" only once, in ¶266.
  And the Ageing Society Strategies working group is **¶266, not ¶265** (¶265 is
  the *Silver Economy* working group); the ophthalmic measure is **¶227, not ¶226**.

**Why this matters more than any other item in this file.** The Open Challenge's
"one rule that cannot be broken" is a citation requirement, judged partly by a
Research Director from the Chief Executive's Policy Unit who does policy
evidence for a living. Our proposal used to answer the question the brief did
**not** ask (2026 policy) instead of the one it did (2025 policy); that is now
reversed.

**The 2026 quotes themselves were never the problem — the edition was.** The 2026
paragraphs are all verified verbatim against the official PDF, with a SHA-256 and
a page count, in `docs/19_POLICY_ANCHOR.md`, and they stand as the funded
continuation. `docs/19` is now restructured so the **2025** edition leads and the
2026 edition follows, and `docs/13`'s proposal and shot list cite 2025 first.

**Done (2026-10-21).** The proposal, the video's policy beat and the policy table
in `docs/13` all now lead with the **2025** anchors (¶75 / ¶227 / ¶266), with the
2026 money (¶113 $300M, ¶398–400 $100M) as the follow-through. That converts a
liability into the one thing the brief says the Open Challenge is designed to
test: "the ability to read a policy document carefully, identify a genuine gap,
and make a clear and evidence-based case for why it matters."

### 1.7 What comparable student AI hackathons reward

- **IC Hack (DoCSoc, Imperial, London)** publishes its criteria outright:
  **"Originality / Usefulness / Technical Challenge"** (IC Hack 24, IC Hack 19) and
  **"Creativity / Utility / Technical Achievement"** (IC Hack 26). Sources:
  <https://ichack24.devpost.com/>, <https://ic-hack-19.devpost.com/>,
  <https://www.startupnetworks.co.uk/links/link/29302-ic-hack-26/>.
  The pattern is stable across a decade: **original idea + actually useful +
  technically hard**. SightLine is strong on the last two, weakest on the first.
- IC Hack 2024's sponsor categories included "Health-Tech" (Terra) and
  "Most Magical Moment" (incident.io) — the second is a *demo quality* prize,
  which is evidence that hackathon judging rewards a single memorable,
  verifiable beat over comprehensive coverage
  (<https://ichack24.devpost.com/>).
- IC Hack 2026's sponsor categories include "Best Use of Technology to Create
  Positive Social Impact" (IMC Trading) and "The Occam Award" (Bending Spoons)
  (<https://www.startupnetworks.co.uk/links/link/29302-ic-hack-26/>). The Occam
  Award is the anti-over-engineering prize; our stack (two trained models, ONNX,
  a build gate that fails if an OCR library returns) is defensible on every
  pillar except that one.

---

## 2. The patterns that separate winners from the rest

Read as: what the record shows winners do, then whether we do it.

| # | pattern | do we exhibit it? | the evidence a judge would see |
|---|---|---|---|
| P1 | **A working artefact, not slides.** Round 1 has no room for a build; the Grand Final rewards a demo. | **Half.** The artefact exists and is live — but it does not read the input it promises on a real handheld photo. | Live URL `https://adeviwon.github.io/SightLine/`; 278 tests pass, 1 xfail (`docs/03_RESULTS.md` §6); WebKit battery 191/191 (`docs/04_RUNNING_ON_IOS_AND_ANDROID.md`). Against that: end-to-end **field recovery 12.5%**, **0.0% on `handheld_light` and `off_axis`** (`docs/14_HACKATHON_BENCHMARK.md`), and `docs/03_RESULTS.md` reports the shipped engine recovering **0.0%** of ground-truth fields end to end. |
| P2 | **One demo moment a judge can verify on the spot.** | **Yes — our strongest single asset.** | Airplane mode switched on mid-demo, then a scan of a *fresh* document that works. `docs/PRIVACY.md` §4 Check B; `docs/05_DEMO_RUNBOOK.md` 3:05. It is verifiable, it is memorable, and it cannot be faked by a concept team. |
| P3 | **A real stakeholder or user, not a persona.** The 2025 Gold went to a physical assistive device for real users. | **No.** This is our biggest honest hole. | `docs/12_HACKATHON_SUBMISSION.md` §5: "we have no partnership, no letter of intent, and no pilot." The only "user" anywhere is the invented "Mrs. Chan (78, Wong Tai Sin)" in the `docs/13` shot list. "A named clinician" is claimed as accountable in `docs/12` §4 and never named. |
| P4 | **Policy specificity — quoted, paragraph-numbered, in the required edition.** | **Yes — and now in the right edition.** | The 2025 anchors (¶75, ¶227, ¶266) are quoted verbatim with paragraph numbers, with the 2026 money (¶113 $300M, ¶398–400 $100M) as the follow-through: `docs/19_POLICY_ANCHOR.md` §3–§4. Corrected 2026-10-21. See §1.6. |
| P5 | **Measurable claims with named failure modes.** | **Yes, unusually.** Nobody else will show a reverted improvement. | `docs/03_RESULTS.md` (the page is generated from artifacts, not typed), `docs/07_EVALUATION_METHODOLOGY.md`, `docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md` §6–§7, `docs/14`. Six detector variants "improved" inside the noise, McNemar **p = 1.000**, and we reverted the one that raised the ceiling while worsening CER. **Caveat:** the numbers are not consistent *between* our own documents — see §3, pillar 3. |
| P6 | **Storytelling wins specialist awards even for concept-level work.** | **Partly — we under-use it.** | Helios won Best Presentation on *simulated prototypes* (§1.5). Our own deck (`docs/00_HACKATHON_PLAYBOOK.md` Part 4) runs three consecutive technical slides (models, data, results) in the middle of a 9-slide deck, and the video spends 60 s of 210 s (shot 4) on screen-recording UI. Our story is present but it is not the spine. |
| P7 | **Track selection is strategy, not admin.** | **Yes — with an unverified precondition.** | We chose Open Challenge, capped at 5 university teams versus unlimited prescribed-challenge teams (FAQ). But the cap means **we may not be in it**: places are allocated on an 80-word problem statement due 24 September 2026, "not on a first-come, first-served basis" (third-party brief, §7). Confirm our status in writing. |
| P8 | **Visible cross-disciplinarity, not a claim of it.** | **No artefact.** | The STEMB answer exists only as a Q&A paragraph in `docs/13`. No financial model, no regulatory pathway, no stakeholder map, no risk register, no STEMB map. (The brief's Step 4 asks for a regulatory pathway, a financial model, key stakeholders and a top-three risk register; `docs/12` §5 has a partial cost model and no regulatory pathway.) |
| P9 | **An external endorsement a judge can recognise.** | **No.** | No clinician name, no NGO, no letter, no pilot. The Gold prize itself is a *letter of recommendation* — this competition trades in credible signatures, and we have none. |

---

## 3. The six pillars

For each: what we have and where it lives, the weakness a judge would actually
find, and the single highest-leverage addition at the lowest cost.

### Creativity

**What we have.** Two ownership claims rather than a feature list: we trained the
recogniser ourselves (`docs/11` §3) and we read documents with the network off
(`docs/PRIVACY.md`). The genuinely creative *design* decision is in
`docs/12_HACKATHON_SUBMISSION.md` §4: the product is built to **refuse** —
"An omitted field is safe; a guessed dose is not."

**The weakness a judge will find.** "Read a document aloud for blind users" is a
known category (Seeing AI, Google Lookout, Envision, Be My Eyes). On the 2025
winner list, the originality award went to the "boldest and most original idea".
Our difference is an **engineering** difference (offline, own model), not a
conceptual one — and the deck currently presents it that way, as an architecture
diagram rather than a reframe. A judge can fairly write "competent, derivative
category."

**Highest-leverage addition.** Name and then *demonstrate* the design principle:
**a reading aid whose best answer is sometimes "I don't know."** Concretely: film
a 30-second A/B on the same deliberately-blurry prescription — a cloud/generic OCR
returns a confident wrong number, SightLine returns the refusal. Ten seconds of
screen, no new code, and it converts our weakest pillar from "same category as
everyone" into "the only team that designed for the failure mode". Put the
refusal beat earlier in the video (it currently sits at 1:00–2:20 inside shot 4)
and name it out loud as the innovation.

### Innovation

**What we have.** A 942,166-parameter CRNN+CTC trained from scratch and shipped
as ONNX in WASM (`docs/11` §3–4); non-autoregressive greedy CTC decoding chosen
for the latency budget (`docs/12` §2); ONNX↔PyTorch parity at max |Δ| = 1.94e-05;
a build gate that **fails** if an OCR library returns (`bundle-check`);
documented, mutation-tested failures (`docs/11` §7).

**The weakness a judge will find.** `docs/03_RESULTS.md` §3 says it plainly:
the shipped CRNN reaches **7.4% field accuracy** where the Tesseract path it
replaced reached **62.2%** — "8x worse than the engine this project replaced."
A judge is entitled to read that as "we deleted a working engine and wrote a
worse one for the novelty of it." That is a live objection and we do not
currently have a one-line answer to it.

**Highest-leverage addition.** Make the **negative result** the innovation claim.
"Six detector variants, all within ~4 pp, none statistically significant
(McNemar p = 1.000); the best one raised the ceiling and made the text worse, so
we reverted it, and we publish that." Almost no student team can say this. It
costs one appendix slide and one video beat, converts the weakest number in the
repo from a liability into evidence of research maturity, and speaks directly to
Dr Bell and to the "Best innovation" criterion. It also gives us the honest
answer to the "8x worse" question: we can say what our own recogniser is for
(nothing depends on network or on a 41 MB third-party binary), and we can say
where it is not yet good enough.

### Feasibility

**What we have.** A PWA that installs from a URL on both platforms with no app
store (`docs/04`); zero network calls after install, enforced in `app/sw.js` and
verifiable in minutes (`docs/PRIVACY.md` §4 Check A/C); 26.45 MiB payload (models + tokenizer), ≈37.2 MiB installed with the wasm runtime
(`docs/11` §4); 278 tests plus a 191/191 WebKit battery; a cost model with
**zero marginal inference cost** and under US$1,000 for year one
(`docs/12` §5); a demo runbook with contingencies (`docs/05`).

**The weakness a judge will find — three of them, all fair.**
1. **It has never run on a physical phone.** Our own README says so:
   "⚠️ *not on physical iPhone/Android hardware*". Every proof is emulation.
2. **It does not work on the input the product is sold on.** Field recovery
   12.5% overall and **0.0% on `handheld_light`** (`docs/14`); detection recall
   caps line accuracy at 67.9–71.6% (`docs/03` §3, `docs/11` §6); three of twenty
   real pages are resolution-bound and unfixable by any detector.
3. **Our own numbers disagree with each other.** `docs/03` §3 reports 29.30% /
   11.95% and a 20-page set at 71.6% recall / 58.4% CER / 2.4% line-exact;
   `docs/14` now reports 29.30% / 11.95% and the 20-page (67.9%/1.8%/56.5%) and 193-page set-B (71.6%/2.4%/58.4%) readings — the 60-page set and 1.96%/63.46% it used to cite are gone;
   `docs/12` §3 reports **12.5%** end-to-end field recovery while `docs/03` §3
   reports **0.0%** end to end. Every one of those was measured; none of them
   agree. `docs/13`'s own rule — "every on-screen number must appear in
   `docs/03_RESULTS.md`" — **cannot currently be satisfied as written**, and a
   judge who opens two of our documents finds a contradiction. On a team whose
   differentiator is honesty, an internal inconsistency is more damaging than a
   low number.
   *Status note (2026-10-21): a parallel reconciliation lane is collapsing these
   figures into one authoritative table (§4, action 3). Do not treat any single
   number below as settled until that table lands and every figure here is re-read
   against it.*

**Highest-leverage addition.** One **authoritative results table**, in one file,
with the count of pages and the exact command per row, and every other document
pointing at it — plus the demo speaking to it. This is a reconciliation job, not
a measurement job: ~2–3 hours, and it protects the pillar where we look strongest.
Second, cheap and concrete: get the app onto **one physical phone and one real
handheld photo read correctly end to end**, filmed. One verified cold start on
hardware is worth more to this pillar than any slide.

### Potential Impact

**What we have.** A defensible prevalence base: **199,600 people (2.7% of the
population) with visual impairment**, 113,900 with multiple disabilities (HKSAR
C&SD Special Topics Report No. 63, 2021, via HK Blind Union — `docs/13`);
**22% of the population aged 65+** and **46.4% polypharmacy** (HKSAR LegCo
ISSH32/2024; Lee et al., *Age and Ageing* 2023, doi:10.1093/ageing/afad014 —
`docs/12` §1); WHO's "50% of preventable harm" framing; a named deployment
shortlist (HA pharmacist services, SWD/community elderly centres, Macau Health
Bureau) and a cost model with zero marginal cost per user (`docs/12` §5).

**The weakness a judge will find.** Two, and the second is fatal to the award
this project should be aiming for.
1. **A prevalence statistic is not demand.** Every number we have proves the
   problem exists. None of them proves anyone wants *this*. `docs/12` §5 says it
   in the submission: "we have no partnership, no letter of intent, and no
   pilot." The "Greatest social impact" award asks for "meaningful and lasting
   difference to communities in Hong Kong or Macau" — that needs a community, not
   a statistic.
2. **Our tool cannot read Chinese, in a Hong Kong competition.** The output
   alphabet is 37 symbols with zero CJK (`docs/11` §6, `docs/12` §4) for a
   population that predominantly reads Chinese, in a bilingual dispensing
   environment. We disclose this, which is right — but disclosure caps the
   impact claim rather than repairing it. Add that our own TTS defaults to
   `en-GB`, so the *audio* is wrong-language too, which is a separate and more
   cheaply fixable problem than recognition.

**Highest-leverage addition.** **One external artefact before 2026-11-14.**
Ranked by cost: (a) a usability session with even one elderly or low-vision
participant, consented and photographed, saying what happened — strongest per
hour; (b) an email or letter of interest from the Hong Kong Blind Union or a
district elderly centre; (c) a named clinician or pharmacist willing to be the
accountable reader we already claim exists in `docs/12` §4. Any one of these
changes the pitch from "should help" to "was used". Separately: **the Cantonese output build has shipped** (`app/js/say_cantonese.js` + an
18,839-pair CC-BY-4.0 zh/en medicine-name lexicon) — the drug name and dose are spoken in
Cantonese, with a **refusal** for any name the lexicon cannot translate, which removes the
"English-only aid for a Cantonese-speaking population" objection from the half of the demo a judge
actually hears. What remains is not a one-parameter change: no Cantonese speaker has reviewed the
sentence frames, and the drug-name step is **now measured on real medicine labels** — 150 of them,
name recovered on **10**, **8 verbatim**, a floor on a Taiwanese corpus, not Hong Kong labels
(`docs/26_IDEA_REFINED.md` §7; `ml/src/eval_medicine_labels.py`).

### Interdisciplinarity

**What we have.** A paragraph. `docs/13`'s Q&A maps five letters onto five
activities ("T — on-device ML pipeline + build gates; E — image restoration;
S — vision-science and accessibility research; M — medication-safety framing;
B — deployment via the 2026 $100M Gerontechnology Promotion Scheme, the follow-through of the 2025 ¶266 gerontechnology priority"). No diagram, no
names, no artefact per letter, and no place in the repo where the team's
composition is stated at all.

**The weakness a judge will find.** The repo is a **single-discipline artefact**:
an ML/software repository with tests. The business letter has a cost table and no
regulatory pathway, no go-to-market, no risk register (the brief's Step 4 asks
for all of these for open-challenge teams). The medicine letter has no clinician.
The science letter has no study design or ethics statement. And here is the
eligibility edge: the FAQ requires that "**All teams in both tracks must also
span at least three of the four STEMB disciplines**"
(<https://www.imperial.ac.uk/study/visit/events/global-hackathons/hackathon-hong-kong-macau/faqs/>),
and the third-party brief says four of four with at least one member from each
city. **Our eligibility rests on a claim that is nowhere evidenced in the repo.**
A judge who scores "interdisciplinarity" low is not being unfair — we gave them
nothing to score.

**Highest-leverage addition.** The **STEMB integration map**: one page, four
columns (S / T / E / M / B), each row naming the person, their discipline, and
the artefact that evidences it with a file path. The brief says the map "will be
assessed as a standalone component" and that teams which cannot demonstrate
genuine integration "will not score well on the STEMB integration criterion,
regardless of the technical quality of their solution." Two hours. It is the
single best marks-per-hour item in this document, and it doubles as the Teamwork
slide (§3, Teamwork) and as our eligibility proof.

### Teamwork

**What we have.** Nothing in the repo. Not one artefact.

**The weakness a judge will find.** "How you've worked as a team" is one of the
six published criteria, and we produce **zero** evidence for it. The repo reads
as a single owner's work: one voice in every document, a commit history from one
contributor, no roles, no meeting record, no division of labour, and
`docs/00_HACKATHON_PLAYBOOK.md`'s own submission checklist still carries
"□ Team member names and roles listed" as an unticked to-do. The compounding
problem is that the same absence also weakens Interdisciplinarity (§3) and puts
the 4–5-member and 3-of-4-STEMB eligibility rules in play. We are one bullet in
our own checklist away from evidence.

**Highest-leverage addition.** A single artefact labelled **"Who did what"**:
per member — discipline (the STEMB letter), the module or document they own, and
a link that proves it (file path or commit range) — plus one decision the team
made together and how. One hour's work, and it is the same page as the STEMB map.
This is the cheapest fix in the file and it protects a whole scored criterion.

---

## 4. Ranked actions

**Method.** Each addition is scored as `marks gained (1–5) × probability a judge
notices (0–1) ÷ effort in hours`, using the pillar weights implied by the
published wording (impact and teamwork named explicitly; attribution to a single
pillar where an item serves two is noted). Effort is wall-clock work by one
person. Lead time is tracked separately, because the formula cannot see it — the
stakeholder item (§5 below) ranks outside the top five on effort yet starts
today because it cannot be compressed.

### Ranked table

| rank | action | marks × P ÷ h | score | do it before |
|---|---|---|---|---|
| 1 | **STEMB map + "Who did what" one-pager** — one page: member, discipline, owned artefact with a file path | 5 × 0.9 ÷ 2 h | **2.25** | **both** (video end card + a proposal clause before 10-25; slide + handout before 11-14) |
| 2 | ~~Fix the policy anchor~~ **Done 2026-10-21** — the 2025 edition now leads, the 2026 money is the follow-through | — | — | closed |
| 3 | **Reconcile the numbers** into one authoritative table; point every other doc at it | 4 × 0.85 ÷ 3 h | **1.13** | **2026-10-25** (numbers are on screen in the video) |
| 4 | **Refusal A/B clip + name the design principle** | 4 × 0.7 ÷ 3 h | **0.93** | **2026-10-25** (it is footage; it must be shot this week) |
| 5 | **Cold-start drill on the two presentation phones + backup laptop & hotspot** | 3 × 0.9 ÷ 4 h | **0.68** | **2026-11-14** |
| — | *long-lead parallel track:* external stakeholder artefact + Cantonese TTS default | 5 × 0.9 ÷ 20 h | 0.23 | **start now**, land before **2026-11-14** |

### The top five, as actions

**1. Build the STEMB map and the "Who did what" page.**
*Exact artefact:* one page (Markdown in-repo plus a rendered slide) — five columns
S/T/E/M/B, one row per member: name, discipline, what they own, and the file path
or commit range that proves it. Include the eligibility statement ("our team
spans N of the four STEMB disciplines") and one team-level decision with its
rationale.
*Where:* a new section in `docs/12_HACKATHON_SUBMISSION.md` §4, plus the end card
of the Round 1 video, plus one Grand Final slide.
*Why the formula ranks it first:* it is the only item that moves **two** scored
pillars (Interdisciplinarity and Teamwork) and simultaneously evidences an
eligibility rule. Two hours.

**2. ~~Fix the policy citation before the proposal is frozen.~~ Done 2026-10-21.**
*What changed:* the 150-word proposal and `docs/13`'s policy-anchor table now
**lead with the 2025 edition** and present the 2026 paragraphs as the follow-through.
The anchors, verbatim with paragraph numbers: **¶75** (AI carer support / access to
information, `p73.html`), **¶227** (ophthalmic services, under (D) Healthcare),
**¶266** (Working Group on Ageing Society Strategies incl. gerontechnology; "By 2046 …
36%", `p265.html`) and **¶73–74** (AI "steered by safety and driven by application");
the 2026 money — ¶113 ($300M "AI for Welfare Lab") and ¶398–400 ($100M Gerontechnology
Promotion Scheme / the ITF) — is the continuation. Every quote is re-checked against
<https://www.policyaddress.gov.hk/2025/public/pdf/policy/policy-full_en.pdf>
(SHA-256 `c4ecaf…18f3ec`) and the 2026 PDF (SHA-256 `f24973cd…49246b`) in
`docs/19_POLICY_ANCHOR.md`. **Still open:** email `imperialhackathon@imperial.ac.uk`
to confirm in writing that the Open Challenge is assessed against the 2025 edition,
and to confirm our placement in the capped track.

**3. Reconcile the numbers into one table.**
*Exact artefact:* one Markdown table (add it to `docs/03_RESULTS.md` or a short
`docs/19`-style file that `docs/03` links to) with columns: metric, value, which
corpus (20 pages vs 193 pages vs 12 rendered docs vs 17,945 words), the exact
command, and the date measured. Then correct or cross-reference `docs/12` §3,
`docs/13`'s shot list, `docs/14` and `docs/11` §6 so no two documents disagree.
*Done when:* `grep -rn` for `29.3` / `29.7` / `58.4` / `63.46` / `12.5%` / `0.0%`
returns rows that all point at the same table with the same corpus labels.
Three hours, and it makes `docs/13`'s own on-screen rule satisfiable for the
first time.

**4. Shoot the refusal A/B and name the principle.**
*Exact artefact:* a ~30-second clip — the same blurred prescription through a
generic/cloud OCR reading a confident wrong figure, then SightLine saying "I
could not read that — please check with a pharmacist." Add one VO line naming it:
"a reader designed to say *I don't know*". Move it to roughly 0:50–1:20 in the
video so it lands before the results beats. One slide in the Grand Final deck,
immediately after the problem slide.

**5. Grand Final demo drill.**
*Exact artefact:* a written, timed rehearsal record — cold start from a
force-quit (not a warm tab) on **both** presentation phones, in airplane mode,
with the model precache done; a laptop running `python3 -m http.server -d app`
plus a personal hotspot as the fallback origin; a screen recording as the last
resort. Note the trap in `docs/04` §2: a LAN-IP load is **not a secure context**,
so `getUserMedia` is refused — a LAN fallback can demo the "Choose a Photo" path
only, never the camera. Four hours, and it is the difference between a demo and
an apology.

**Long-lead parallel track (start today, not before 10-25).** Get one external
stakeholder artefact, and switch TTS to `zh-HK`. The formula ranks it low only
because it costs days; it carries the highest marks-per-notice of anything in the
file.

---

## 5. The one differentiator, and the one exposure

**The one thing we have that no other team is likely to have.**

**A refusal.** Every competitor will demo their system succeeding. We can demo
ours **declining**, on purpose, in front of a judge, and explain why that is the
correct behaviour for a dose — because the whole architecture was built so the
worst outcome is a refused answer rather than a wrong one. It is backed by
artefacts nobody else will have: a published negative result (six detector
variants, none statistically significant, the best one reverted), a build gate
that fails if a third-party OCR library returns, and an airplane-mode moment that
is verifiable on the judge's own timeline. The closest thing in the record is
Helios winning Best Presentation on simulated prototypes — the bar for a
memorable entry is lower than we treat it, and our memorable entry is real.

**The one thing we are most likely to be marked down for.** *(Honest answer.)*

**It does not read the document it says it reads.** Our own documents measured
it: field recovery **12.5%** end to end and **0.0%** on `handheld_light` and
`off_axis` (`docs/14`); the line detector caps everything at **67.9–71.6%** recall
(`docs/03` §3, `docs/11` §6); line-exact accuracy on real hand-photographed pages
is **1.8–2.4%**; and three of twenty real pages are resolution-bound and
unfixable. There is **no Hong Kong medicine-label corpus, and none for training** — the only real
medicine-label corpus we hold is the CC-BY-4.0 Taiwan leaflet set, and on it the drug-name step now
has a measured floor: **10 of 150 recovered, 8 verbatim** (`docs/12` §1, `ml/src/eval_medicine_labels.py`,
`docs/11` §6).
So the pillar the wording names second — potential impact, the reason the product
exists — is carried by a prevalence statistic and a clean-scan demo we control.
A judge who photographs their own prescription is not being adversarial; they are
doing exactly what the Grand Final invites, and what they will see is a refusal
or a misread. Our refusal design turns that into a defensible answer, but only if
we script it first and never pretend the ceiling is higher than it is. The
mitigation is positioning, not improvement: **say the number first, make the
refusal the demonstrated feature, and keep the scan in our hands for the first
three minutes** (`docs/05`).

**And the second thing.** Zero Chinese. A 37-symbol alphabet with no CJK for a
Hong Kong audience, and an `en-GB` TTS default, is the criticism a local judge
will reach for fastest — and unlike the accuracy gap it cannot be reframed as
integrity, only as scope. The 2025 winners' projects all sat on locally defined
problems; a Hong Kong medicine aid that cannot read a Hong Kong medicine label is
the fastest way to lose the "Greatest social impact" award. Fix the audio
language; state the recognition roadmap as a data problem with a number
(~3–4k common characters) and a timeline.

**And the third, which is a compliance question rather than a judging one.**
`docs/13`'s claim that the Open Challenge is our best route rests on a 2025
win by a secondary-school team in a Wildcard challenge that no longer exists; that
caution stands (§1.5). The policy-edition half of this item is **fixed** — the
anchor now cites the **2025** edition the rule names, with the 2026 money as the
follow-through (§1.6, corrected 2026-10-21).

---

## 6. Demo risks, and the mitigation for each

Ordered by probability × damage. Every mitigation is something we can do before
2026-11-14, and none of them require new model capability.

| # | risk | why it is real | mitigation |
|---|---|---|---|
| 1 | **A judge photographs a document our model cannot read** | detection recall caps line accuracy at 67.9–71.6%; 0% on `handheld_light`; 3/20 real pages resolution-bound (`docs/03` §3, `docs/11` §6) | Pre-empt it: state the ceiling and the sample size in the demo *before* the first scan. Keep three prepared documents and their fixed order (`docs/05`). Script the refusal as the designed behaviour, not a failure. Do not invite an unscripted document in the first three minutes; if one arrives, narrate the detection ceiling and hand the retake flow. |
| 2 | **Venue Wi-Fi / first load** | the PWA needs one successful online load (~40 MB precache) before airplane mode works; the primary demo origin is an online GitHub Pages URL | Precache on both phones the night before **and** again on site; carry a laptop serving `app/` on a personal hotspot as the fallback origin; keep a screen recording as the last resort. Know the trap: a LAN-IP load is not a secure context, so a LAN fallback can only demo "Choose a Photo", never the camera (`docs/04` §2). |
| 3 | **iOS Safari quirks / version floor** | WASM SIMD requires **iOS 16.4+**; a lookbehind regex once made the app completely dead on older Safari while every Chromium test passed (`docs/04` §2) | Confirm both demo phones are ≥16.4. Run `bash run.sh ios-check` before travelling. Never demo in private browsing (service workers disabled). Never demo from a plain LAN-IP load. |
| 4 | **Phone battery / memory pressure** | three ORT sessions in the wasm heap; Safari jettisons suspended tabs under pressure (`docs/04` §2) | Charge both phones above 80% plus a power bank (`docs/05` pre-flight). Rehearse a force-quit-and-recover. Know the escape hatch: the **Skip classifier** toggle drops the 22.9 MB encoder from the pipeline. |
| 5 | **No installable app / judge cannot use it on their own phone** | it is a URL, not an App Store listing; iOS install requires **Safari** (not Chrome), Share → Add to Home Screen (`docs/04` §2) | Hand over our own already-installed phone. Carry a printed QR to the HTTPS URL for judges who want it on their device. Never ask a judge to install during the pitch. |
| 6 | **Our numbers contradict each other under questioning** — LARGELY FIXED 2026-10-21 | the docs/03-vs-docs/14 rows were reconciled (29.30%/11.95% is word-level; docs/14 now carries the same row) and the end-to-end 12.5%/0.0% rows were replaced by the 45.9%/0.303 shipped build; the remaining exposure is the epoch-39-vs-29 lineage and the un-artifacted SROIE row (see docs/03's provenance note) | Action 3 in §4 — one authoritative table, every other doc pointing at it. Do not answer a numbers question from memory; answer from the table. |
| 7 | **Deck fails on the shared presentation laptop** | finalist slides are pre-loaded onto a shared machine, 48 h ahead, and the live demo needs our own laptop (FAQ) | Send the PDF **and** a font-embedded copy 48 h early; verify it on a machine without our fonts; assume no clicker and no speaker notes on screen. |
| 8 | **Open Challenge placement / eligibility** | capped at 5 university teams, allocated on an 80-word problem statement due 24 Sept 2026 (third-party brief); the anchor must be a **2025** Policy Address priority *not already covered by challenges 1–5*, and Challenge 1 is "Agetech… gerontechnology… ageing-in-place" — which is arguably where "AI reads a prescription to an elderly visually-impaired user" already sits | Confirm our placement and our track in writing with `imperialhackathon@imperial.ac.uk`. Prepare a two-sentence gap argument for why vision loss plus document reading plus medical-safety refusal is not covered by an ageing-in-place challenge, or be ready to present the same project under Challenge 1. |
| 9 | **Absent team at the Grand Final** | in-person pitches only; no remote participation; all presenting members must attend (FAQ) | Lock attendance now, including travel from Macau. Nominate a second speaker so any single absence does not remove a STEMB discipline from the room. |

---

## 7. What I could not verify — do not present any of this as fact

1. **~~Which Policy Address edition the Open Challenge is assessed against~~ —
   RESOLVED 2026-10-21.** The published FAQ text and the brief both say the anchor
   must come from "the Hong Kong or Macau **2025** Policy Address", so the 2025
   edition leads and the 2026 edition is the funded continuation (`docs/19` §1).
   Verified first-hand against the official full texts: the 2025 PDF (SHA-256
   `c4ecaf…18f3ec`) carries ¶75, ¶227, ¶266 and ¶73–74; the $300M "AI for Welfare
   Lab" (¶113) and the "$100 million … Gerontechnology Promotion Scheme" (¶400) are
   confirmed in the 2026 PDF (SHA-256 `f24973cd…49246b`), and the 2026 ¶398 names
   the Innovation and Technology Fund for Application in Elderly and Rehabilitation
   Care. **Still UNVERIFIED (and it matters): no written confirmation from Imperial
   that the Open Challenge is assessed against the 2025 edition, nor that SightLine
   was accepted into the capped track — email `imperialhackathon@imperial.ac.uk`.**
   (The earlier note that "a 'Gerontechnology Promotion Scheme' at paras 398–400
   could not be confirmed in any edition" is now **confirmed** in the 2026 text.)
2. **The full challenge brief itself.** The open-challenge and prescribed-challenge
   text I quote (the 80-word problem statement due 24 September 2026, the sprint
   deadline of 24 October 2026, the "scientific poster", the one-page STEMB
   integration map, "minimum ten peer-reviewed sources", the regulatory pathway /
   financial model / risk register asks) comes from a **third-party transcription**
   retained by another 2026 entrant at
   <https://github.com/Peeradonn/Imperial/blob/main/Imperial%20Sources/Challenge%20Briefs.md>.
   It paraphrases the official page structure and quotes the official source
   line, but I could not verify it against the official brief (registration is
   closed and the brief is only issued to registered participants).
   **Treat as UNVERIFIED.** Two specifics conflict with the official FAQ: the
   transcription says "**All four** STEMB pillars must be represented in every
   team, and **at least one member from each city** is required", while the
   official FAQ says teams "must also span **at least three of the four** STEMB
   disciplines" with no city requirement. **The official FAQ wins until the brief
   says otherwise — but check, because a Macau-member requirement would be an
   eligibility problem.**
3. **The judge list.** Two official pages disagree about the Google Hong Kong
   judge: the judges page names **Helen Kwok** (Digital Strategy Lead), the track
   page names **Lucas Cheung** (Digital Strategy Lead). Our `docs/13` repeats
   Lucas Cheung. **Do not name a judge on a slide until resolved.**
4. **"Four specialist awards" that lists five** on the prizes page. Which five
   (or four) are actually on offer, and whether "Best sustainability" is limited
   to green challenges, is **UNVERIFIED**.
5. **The 2025 Gold winner's project.** `expatliving.net` attributes an arthritis
   assistance glove to a Malvern team called "Handy Helpers" winning gold at the
   **Global Natural Science and Innovation Competition U16** at Imperial — a
   different competition from the HK hackathon. Same school, same team name,
   different event named. **UNVERIFIED as the hackathon-winning project.**
6. **Competition scale.** "80 brilliant teams" in 2025 is from an unofficial
   sponsor-side LinkedIn post. **UNVERIFIED.**
7. **University-track entry counts for 2026, and whether SightLine was accepted
   into the capped Open Challenge.** Nothing is published either way.
   **UNVERIFIED — and this one is load-bearing for our whole pitch.**
8. **Whether a university-track edition of the Hackathon published winners for
   2025.** The 2025 winners page lists secondary schools only, consistent with the
   2025 FAQ restricting entry to grades 10–12 in Hong Kong; I found no evidence
   of a 2025 university-track winner list. If it exists, it is not public.

---

## 8. Acceptance checks

```bash
test -f docs/23_PILLAR_GAP_ANALYSIS.md && wc -l docs/23_PILLAR_GAP_ANALYSIS.md
grep -c 'https://' docs/23_PILLAR_GAP_ANALYSIS.md          # every external claim has a URL
grep -c 'UNVERIFIED' docs/23_PILLAR_GAP_ANALYSIS.md        # the honest list is non-empty
grep -nE '^(#|###) (Creativity|Innovation|Feasibility|Potential Impact|Interdisciplinarity|Teamwork)' docs/23_PILLAR_GAP_ANALYSIS.md | wc -l   # six pillars
grep -nE '2026-10-25|2026-11-14' docs/23_PILLAR_GAP_ANALYSIS.md | wc -l    # both deadlines appear
```
