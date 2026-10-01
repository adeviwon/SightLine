# ICHack Winning Strategy Guide: Offline AI Document Scanner for the Visually Impaired

## Project: **SightLine**
### *Your eyes, offline. Nothing leaves your phone.*

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Understanding ICHack: What Actually Wins](#2-understanding-ichack-what-actually-wins)
3. [Pitch Refinement: Problem, Market, Impact](#3-pitch-refinement-problem-market-impact)
4. [What Judges Look For at Imperial/IHack](#4-what-judges-look-for-at-imperialichack)
5. [Presenting a Privacy-First Offline AI Product](#5-presenting-a-privacy-first-offline-ai-product)
6. [Demo Strategy for a 3-Minute Pitch](#6-demo-strategy-for-a-3-minute-pitch)
7. [Differentiators That Make It Top 1](#7-differentiators-that-make-it-top-1)
8. [Pre-Hackathon Preparation Checklist](#8-pre-hackathon-preparation-checklist)
9. [Q&A Anticipation Sheet](#9-qa-anticipation-sheet)
10. [Prize Category Targeting](#10-prize-category-targeting)

---

## 1. Executive Summary

SightLine is a fully offline mobile camera app that scans banking details, medical prescriptions, and legal documents using local OCR (Tesseract/ONNX), reads them aloud with TTS (espeak/pyttsx3), and never sends a single byte to the cloud. Target users: blind and visually impaired individuals.

This guide gives you a concrete, research-backed strategy to win at ICHack (Imperial College London's hackathon — the UK's largest student-run hackathon, organized by DoCSoc). It is based on actual ICHack judging criteria, past winners, sponsor prize categories, and the specific dynamics of the event.

**The thesis: offline-first + accessibility + privacy is a winning combination at ICHack.** Two of the top three IMC Trading "Tech for Good" projects at ICHack 26 were health/privacy-focused (Smart ePCR won, Relay — an offline-first medical dashboard — was runner-up). At ICHack 25, the accessibility category winner (Seen_ary) was also a visually impaired navigation tool. Your project sits at the intersection of three proven winning themes.

---

## 2. Understanding ICHack: What Actually Wins

### 2.1 The Official Judging Criteria

ICHack uses three judging pillars (confirmed from Devpost):

| Criterion | What It Means | How to Score High |
|---|---|---|
| **Creativity** | Is the idea novel or a fresh angle on a known problem? | Don't just build "another OCR app." Frame it as a *privacy-preserving accessibility revolution* — the first offline document reader that treats banking, medical, and legal documents as first-class, sensitive categories requiring zero-cloud architecture. |
| **Utility** | Does it solve a real problem people actually have? | Show a user who needs this today. 2 million people in the UK have sight loss (RNIB). This is not a hypothetical problem. |
| **Technical Achievement** | How hard was this to build? Did you do something technically interesting? | Emphasize the engineering challenge: running OCR + TTS + image preprocessing entirely on-device, with sub-second latency, on a phone. That is genuinely hard. |

### 2.2 What Past Winners Look Like

**ICHack 25 Overall Winner: "InCite"** (UCL team)
- AI-powered research tool that maps how academic papers cite each other
- Used Anthropic's Claude for NLP
- Won because: clear problem (literature reviews are painful), clean demo, real utility, strong technical execution

**ICHack 25 Accessibility Winner: "Seen_ary"**
- Computer vision app that helps visually impaired users identify the correct bus
- Used YOLOv8 + PaddleOCR + speech recognition + Google Maps API
- Won because: accessibility-first design (high contrast, large text), real user need, clean demo of the "bus arrives → audio alert" moment

**ICHack 25 Runner-up (Best Use of Claude): "BrailleReader AI"**
- Raspberry Pi camera captures Braille → AI pipeline → Claude refines accuracy → ElevenLabs TTS reads it
- Won because: hardware + AI + accessibility + tangible user benefit. Accuracy improvement from 60% to ~100% was a memorable "wow" moment.

**ICHack 26 IMC Trading Tech for Good Winner: "Smart ePCR"**
- Automated real-time clinical reporting for volunteer medics
- FastAPI/Python backend, SentenceTransformers for semantic search, speech-to-clinical-protocol mapping
- Won because: clear social impact (saves medics time so they can save lives), real technical depth, high-concurrency architecture

**ICHack 26 IMC Trading Tech for Good Runner-up: "Relay"**
- Decentralized, offline-first medical dashboard for emergency response
- Won because: offline-first was the core thesis. Privacy + crisis scenarios. Directly relevant precedent for your project.

### 2.3 The Pattern

Winning ICHack projects share these traits:
1. **A demo that tells a story** — one moment where the judge goes "oh, that's cool"
2. **A problem the judge can feel** — not abstract, but visceral ("a blind person can't read their prescription" hits harder than "OCR on mobile")
3. **Technical depth visible in the demo** — not just "we used an API," but "we solved a hard engineering problem"
4. **Scope done well, not scope done broadly** — one perfect flow beats five half-working features
5. **Social impact or real utility** — ICHack has a strong "Tech for Good" culture (IMC Trading sponsors the Positive Social Impact prize)
6. **Accessibility wins at Imperial** — multiple past winners targeted visually impaired users specifically

---

## 3. Pitch Refinement: Problem, Market, Impact

### 3.1 Problem Framing — The Hook

**Do NOT say:** "We built an OCR app that reads documents aloud."

**DO say:** "Imagine you're blind. A letter arrives from your bank. It could be a fraud alert, a statement, or a notice about your mortgage. You can't read it. Your options are: ask a stranger to read your private financial details aloud, or use a cloud-based app that photographs your banking information and sends it to a third-party server. SightLine gives you a third option: scan it, hear it, and know that your data never left your phone."

#### The problem in three sentences (memorize this):

> "43 million people worldwide are blind. 2 million people in the UK live with sight loss. Every day, they face a choice between independence and privacy when reading sensitive documents — banking statements, medical prescriptions, legal contracts. SightLine eliminates that choice."

#### Problem framing structure:
- **Start with the human**: A blind person receiving a letter they can't read
- **Escalate to the stakes**: It's not just any letter — it's banking, medical, legal. These are documents where privacy is legally protected (GDPR, HIPAA, banking secrecy).
- **Expose the false choice**: Current solutions force you to trade privacy for independence. Cloud OCR apps send your prescription or bank statement to a server. That is a privacy violation most users wouldn't accept if they could see what was happening.
- **Present the resolution**: SightLine makes the trade-off disappear. Full independence, full privacy.

### 3.2 Market Size

Use these real, citable statistics in your pitch:

| Statistic | Source | Use In Pitch As |
|---|---|---|
| 2.2 billion people globally have near or distance vision impairment | WHO (2024) | Global market size — "This isn't a niche. 2.2 billion people." |
| 43 million people are blind worldwide; 295 million have moderate-to-severe visual impairment | Lancet Global Health / IAPB Vision Atlas | TAM — "43 million blind, 295 million visually impaired" |
| 2 million people in the UK live with sight loss | RNIB | UK-specific relevance — "2 million people right here in the UK" |
| 322,638 registered blind and partially sighted people in the UK | RNIB (2022-23) | "Over 320,000 registered blind in the UK alone" |
| By 2050, ~60 million people will be blind globally | Lancet Global Health | Growth trajectory — "This problem is growing, not shrinking" |
| 90% of people with vision loss live in low- and middle-income countries | IAPB | "Offline isn't just a privacy feature — it's an accessibility feature for the 90% who don't have reliable internet" |

#### The market argument:
- **TAM**: 43M blind globally + 295M MSVI = 338M potential users
- **SAM**: 2M UK sight loss population (immediate, English-language, NHS-relevant)
- **SOM**: The subset who need document reading — everyone who receives physical mail, prescriptions, or printed legal documents. That is nearly all of them.
- **The "why now"**: GDPR enforcement is rising, cloud AI privacy concerns are headline news, and on-device AI has crossed the threshold where Tesseract + ONNX can run in real-time on commodity phones.

### 3.3 Impact Framing

Frame impact across three axes — this is how judges think about "Utility" and "Tech for Good":

1. **Dignity & Independence**: "A blind person can read their own bank statement without telling a stranger their account balance." This is the line that makes judges remember you.

2. **Privacy as a Right, Not a Luxury**: "Every existing solution — Google Lens, Seeing AI, Be My Eyes — sends data to the cloud. SightLine is the only approach that treats a blind person's medical and financial information with the same privacy a sighted person expects."

3. **Global Accessibility**: "90% of visually impaired people live in low- and middle-income countries where internet is unreliable. Offline isn't a feature — it's the difference between usable and useless." (This is a powerful argument because it reframes your tech constraint — offline — as a global advantage.)

---

## 4. What Judges Look For at Imperial/IHack

### 4.1 The Three Official Criteria (Score Against Each)

#### Creativity
Judges have seen 30+ demos by the time they reach you. "Another OCR app" will score low. Your creativity angle:
- **Category-aware document handling**: SightLine doesn't just OCR — it recognizes *what kind* of document it's reading (banking vs. prescription vs. legal) and adjusts its reading behavior. A prescription might emphasize dosage and drug names. A bank statement might read out totals first. This is a creative layer above basic OCR.
- **Audio-first UX**: Instead of just dumping OCR text into TTS, design the audio experience. Detect document structure (headers, amounts, dosages) and narrate intelligently: "Bank statement. Account ending 4521. Current balance: £3,247. Would you like me to continue reading?"
- **The privacy-architecture-as-feature framing**: Making "zero data leaves the device" a core selling point, not a footnote.

#### Utility
- Cite RNIB statistics. Name the user. Show the daily reality.
- Have a slide or moment that says: "This is what a blind person does today without SightLine" (ask a stranger, skip the document, or risk cloud privacy).
- Show what they do *with* SightLine (scan → hear → done, privately).

#### Technical Achievement
This is where you win or lose. Judges are Imperial computing students, alumni, and sponsor engineers. They know what's easy and what's hard.

**What is genuinely hard in your stack (emphasize these):**
- Running Tesseract OCR at usable speed on-device with acceptable accuracy — this requires image preprocessing (OpenCV: deskew, denoise, threshold, perspective correction)
- ONNX Runtime integration for model inference without cloud dependency
- Real-time audio feedback loop: scan → process → speak in one fluid gesture
- Document classification without an LLM (you can't call GPT — you're offline). This means a lightweight classifier (ONNX model, keyword heuristics, or a small local model)
- Making TTS sound natural with espeak/pyttsx3 (these engines are robotic by default — if you add any prosody tuning or speech rate adjustment, mention it)

**What is easy and you should NOT overclaim:**
- Basic Tesseract OCR on a clean image — this is a known quantity
- pyttsx3 TTS — straightforward library call
- Camera capture — standard

### 4.2 Sponsor Category Alignment

ICHack has sponsor-specific prizes. Your project maps to multiple categories — **enter the right ones**.

#### Primary target: IMC Trading — "Best Use of Technology to Create Positive Social Impact"
- **Why you fit**: Accessibility + privacy + document reading for blind users is textbook "positive social impact"
- **Precedent**: ICHack 26 winner (Smart ePCR) and runner-up (Relay, an offline-first medical dashboard) both won here. Your project is arguably a stronger social impact case than either.
- **How to win this category**: Emphasize the user. The IMC judges are looking for "does this make a real difference in someone's life?" Your answer: "A blind person can independently read their own bank statement, medical prescription, and legal documents for the first time without compromising privacy."

#### Secondary target: Bending Spoons — "The Occam Award"
- This prize rewards elegant simplicity — the simplest solution that solves a real problem well
- Your pitch: "No cloud, no servers, no API keys, no internet needed. One app, one camera, one speaker. That's it."

#### Possible: DoCSoc categories (if accessibility or developer tooling categories exist in your year)
- ICHack 25 had a dedicated "Accessibility" prize (won by Seen_ary). Check whether your year has one and enter it.
- If "Best Developer Tooling" exists, frame SightLine as a tool that developers can extend for other accessibility use cases.

#### Do NOT enter: "Best Use of Claude" or "Best Use of ElevenLabs"
- Your project is offline-first. Using cloud APIs contradicts your core thesis. Don't compromise the narrative for a sponsor prize.

### 4.3 The Imperial Hackathon Culture

- **Imperial DoCSoc runs ICHack**. The culture values technical depth — this is one of the UK's top CS departments. Don't dumb down the tech; show the architecture.
- **24-hour format**: Judges know you had 24 hours. They don't expect production polish. They expect a working demo with a clear story.
- **Sponsor judges vs. DoCSoc judges**: Sponsor judges (IMC, Marshall Wace, etc.) care about impact and polish. DoCSoc judges (Imperial students/alumni) care about technical achievement and creativity. Pitch to both.
- **Mentor rounds**: Sponsors and mentors walk the floor during the hackathon. Have a 30-second elevator pitch ready and a demo that works at any moment. Talk to every mentor who passes by — their feedback shapes what judges expect.

---

## 5. Presenting a Privacy-First Offline AI Product

### 5.1 The Core Message Architecture

Your privacy thesis has three layers. Present them in this order:

**Layer 1 (Emotional):** "A blind person shouldn't have to choose between reading their prescription and keeping it private."

**Layer 2 (Technical):** "Every existing OCR app sends images to the cloud. Google Vision API, Microsoft Cognitive Services, AWS Textract — all cloud-based. SightLine runs Tesseract OCR and ONNX inference entirely on-device. Zero network calls. Zero external APIs."

**Layer 3 (Evidence):** Show it. During the demo, turn on airplane mode or show a network monitor showing zero outbound traffic. This is your "wow" moment.

### 5.2 How to Make "Offline" Visually Compelling

"Offline" is invisible. You need to make it visible. Here are concrete tactics:

1. **Network monitor on screen during demo**: Run a live network traffic indicator (even a simple Python script showing `0 bytes sent, 0 bytes received`) next to the app. When the document is scanned and read aloud, the monitor stays at zero. This is your proof.

2. **Airplane mode demo**: Put the phone in airplane mode before the demo. Scan a document. It reads aloud. Turn off airplane mode — nothing changes. The point: the app doesn't need the internet, period.

3. **"Zero data leaves your device" diagram**: One slide showing the architecture: Camera → OpenCV preprocessing → Tesseract OCR → ONNX classifier → pyttsx3 TTS → Speaker. No arrows pointing to any cloud. Make it a clean, memorable diagram.

4. **Comparison table**: Show competitors (Seeing AI, Be My Eyes, Google Lens) and their data flow (image → cloud → result) vs. SightLine (image → on-device → result). The visual contrast is powerful.

### 5.3 Handling the "Why Not Just Use Cloud OCR?" Objection

Judges will ask this. Have a crisp answer:

> "Three reasons. First, privacy: our users are scanning bank statements, medical prescriptions, and legal contracts. Under GDPR and HIPAA, sending this data to a third-party server is a compliance risk. Second, accessibility: 90% of visually impaired people live in regions with unreliable internet. An app that stops working when the connection drops is useless to them. Third, trust: blind users can't verify what a cloud app is doing with their data. With SightLine, the proof is architectural — there's no network code to misuse data that never leaves the device."

### 5.4 Privacy as Technical Achievement (Not Just Marketing)

Frame the offline architecture as a technical accomplishment:

- **Model optimization**: Running OCR on-device means managing memory, CPU, and battery. Mention if you quantized the ONNX model, used int8 inference, or tuned Tesseract parameters for mobile.
- **No external dependencies**: The entire pipeline (preprocessing → OCR → classification → TTS) runs in one process. No API calls, no rate limits, no service costs.
- **Deterministic and reproducible**: Cloud OCR results vary. On-device OCR is deterministic — the same document always produces the same output.

---

## 6. Demo Strategy for a 3-Minute Pitch

### 6.1 The 3-Minute Structure

Based on research from hackathon judging panels (JetBrains, Devpost, AngelHack), here is the optimal structure for a 3-minute ICHack pitch:

| Time | Section | Content | Goal |
|---|---|---|---|
| 0:00–0:20 | Hook | "43 million people are blind. When a bank letter arrives, they can't read it. Their options are: ask a stranger, or send it to a cloud server. SightLine gives them a third option — and nothing leaves their phone." | Make the judge feel the problem in 20 seconds |
| 0:20–0:30 | Solution reveal | "SightLine. Fully offline document scanner with text-to-speech for visually impaired users." One line. Show the app name and logo on screen. | Crystal clear what you built |
| 0:30–2:00 | Live demo | Run the actual app. Scan a real document. Hear it read aloud. Show the network monitor at zero. | This is the core — the demo IS the pitch |
| 2:00–2:30 | Technical depth | Quick architecture slide. Show the pipeline: Camera → OpenCV → Tesseract → ONNX → TTS. Highlight what was hard. Mention you turned on airplane mode and it still works. | Prove technical achievement |
| 2:30–2:50 | Impact & market | "2 million people in the UK. 43 million globally. GDPR-compliant by design. Works without internet." Show the comparison table (SightLine vs. cloud apps). | Prove utility and scale |
| 2:50–3:00 | Close | "SightLine. Your eyes, offline. Nothing leaves your phone." Say it, hold the slide, stop talking. | Memorable exit line |

### 6.2 The Demo Script (Exact Words)

Practice this. Time it. Deliver it like a person, not a robot.

> **[0:00 — standing at the front, phone in hand, screen mirrored]**
> "43 million people worldwide are blind. When a letter arrives from their bank, they can't read it. Their options today: ask a stranger to read their private financial details, or use a cloud app that photographs their bank statement and sends it to a server they can't see."
>
> **[0:20 — hold up the phone]**
> "We built SightLine. A fully offline document scanner that reads any document aloud — banking, medical, legal — and nothing, nothing, leaves your phone."
>
> **[0:30 — scan the document]**
> "Let me show you. I have a real bank statement here. I'm going to point the camera at it."
> *[Point camera at document. App auto-captures.]*
> "Notice — I've had airplane mode on since we started."
> *[Show airplane mode icon on screen.]*
> "SightLine detects the document, corrects the perspective, runs Tesseract OCR, and..."
> *[App speaks: "Bank statement. Account ending 4521. Opening balance: £2,340. Closing balance: £3,247." ]*
> "It reads it aloud. And this —" *[point to network monitor showing 0 bytes]*
> "— is the network monitor. Zero bytes sent. Zero bytes received. This prescription, this bank statement, this legal contract — it never left this device."
>
> **[2:00 — switch to architecture slide]**
> "The pipeline: OpenCV handles image preprocessing — deskewing, denoising, perspective correction. Tesseract runs OCR on-device. An ONNX model classifies the document type so we can read it intelligently — a prescription gets the dosage emphasized, a bank statement reads totals first. pyttsx3 handles text-to-speech. The entire stack runs in one process, zero network calls."
>
> **[2:30 — switch to impact slide]**
> "2 million people in the UK have sight loss. 43 million globally. Every existing solution — Seeing AI, Be My Eyes, Google Lens — sends images to the cloud. SightLine is the first approach where a blind person can read their own medical prescription and know, architecturally, that their data is private."
>
> **[2:50 — final slide]**
> "SightLine. Your eyes, offline. Nothing leaves your phone."

### 6.3 Demo Risk Mitigation

Hackathon demos fail. Plan for it.

| Risk | Mitigation |
|---|---|
| Camera doesn't focus / bad lighting | Pre-stage the document under good lighting. Practice the exact hand position. Have a backup pre-captured image the app can load if live capture fails. |
| OCR returns garbage | Pre-test with the exact document you'll demo. Choose a document with large, clear print. Have a backup document. |
| TTS is too quiet on venue speakers | Bring a portable Bluetooth speaker. Test volume before the pitch. Alternatively, route audio through the laptop that's mirroring the screen. |
| Phone runs out of battery | Charge to 100% before the pitch. Bring a power bank. Disable all non-essential apps. |
| Screen mirroring fails | Have a pre-recorded video of the demo on a USB stick as a backup. Practice switching to it seamlessly: "Let me show you a recording of the same demo." |
| App crashes mid-demo | Have a second phone with the app loaded and ready. If the first crashes, say "let me grab the backup" and switch phones. Practice this transition. |
| Time runs out | Practice with a timer. If you're at 2:30 and haven't hit the demo, skip the architecture slide and go straight to the close. The demo is non-negotiable; everything else is. |

### 6.4 What to Pre-Stage

Before the pitch:
- **Document ready**: A real bank statement (can be a mock one with realistic data) on a flat, well-lit surface
- **Airplane mode ON**: Already enabled before you walk up
- **App open and ready**: Camera screen loaded, one tap from scanning
- **Screen mirroring working**: Test with the venue's display system before the pitch
- **Network monitor visible**: A small overlay or second window showing 0 bytes transmitted
- **Backup video recorded**: 90-second screen recording of a successful demo, on USB
- **Speaker connected and tested**: If using external audio
- **Architecture slide ready**: Clean diagram of the pipeline
- **Comparison slide ready**: SightLine vs. cloud OCR apps

---

## 7. Differentiators That Make It Top 1

### 7.1 The Differentiators (Ranked by Judge Impact)

#### Differentiator 1: **Zero-Cloud Architecture as the Core Thesis (Not a Feature)**

Every other accessibility app sends data to the cloud. You don't. This isn't a checkbox feature — it's the entire identity of the product. Frame it that way.

- **Why judges care**: Privacy is a hot topic. GDPR, HIPAA, AI Act. A product that is *architecturally* private (not just "we promise not to look") is technically interesting and socially important.
- **Why it's defensible**: Cloud OCR apps can't easily pivot to offline — their business models rely on server-side processing. You've built something they structurally can't.

#### Differentiator 2: **Category-Aware Reading (Not Just OCR → TTS)**

Basic OCR apps dump raw text into a speech engine. SightLine recognizes document categories and reads them intelligently:
- **Banking statement**: Reads account number, opening/closing balance, and flags unusual transactions first
- **Medical prescription**: Emphasizes drug name, dosage, frequency, and warnings
- **Legal document**: Reads section headers and key clauses, skips boilerplate

This is the "Creativity" differentiator. It elevates the project from "OCR wrapper" to "intelligent document assistant."

- **Implementation**: A lightweight ONNX classifier (or even keyword-based heuristics — "dosage," "mg," "account," "balance," "hereby," "agreement") that routes the OCR output to category-specific reading templates. This is achievable in 24 hours.

#### Differentiator 3: **Audio-First UX Designed for Blind Users**

Most OCR apps are sighted apps with TTS bolted on. SightLine is designed for someone who will never see the screen:
- **Audio feedback before scan**: "Document detected. Hold steady. Scanning... Scan complete." — guide the user through the capture without needing to see
- **Document summary before full read**: "Bank statement detected. 3 pages. Account ending 4521. Shall I read the full statement or just the summary?" — let the user control the experience
- **Voice commands**: "Read again," "Skip to next section," "What's the total?" — hands-free interaction
- **No visual UI needed**: The app should be usable with eyes closed. If you can demo it blindfolded, that's a top-1 moment.

#### Differentiator 4: **Works Without Internet — Global Accessibility**

90% of visually impaired people live in low- and middle-income countries (IAPB). Offline isn't just a privacy feature — it's a global equity feature. Frame it:

> "Cloud-based accessibility apps don't work for the 90% of blind people who live in regions without reliable internet. SightLine works on a phone with no SIM card, no Wi-Fi, in a rural clinic in sub-Saharan Africa. That's not a feature — that's justice."

This reframes a technical constraint as a moral advantage. Judges at a "Tech for Good" category will feel this.

#### Differentiator 5: **Open-Source, Reproducible, Verifiable Privacy**

Your entire stack is open-source:
- Tesseract OCR (Apache 2.0)
- ONNX Runtime (MIT)
- OpenCV (Apache 2.0)
- pyttsx3/espeak (GPL)
- No proprietary models, no black-box APIs

This means: **privacy claims are auditable**. A security researcher can read every line of code and confirm no data leaves the device. You can't say this about a cloud app. Mention this — Imperial judges (CS students) will appreciate it.

#### Differentiator 6: **The "Airplane Mode Demo" Moment**

This is your memorable moment. The one thing judges will tell other judges about an hour later. Practice it until it's flawless:
1. Turn on airplane mode (or have it already on)
2. Scan a document
3. App reads it aloud
4. Show network monitor: zero bytes

No other accessibility app can do this. It's a 10-second moment that proves everything.

### 7.2 What NOT to Claim as Differentiators (Avoid These)

| Don't Say | Why |
|---|---|
| "We use AI" | Everyone uses AI. It's not a differentiator at a hackathon. Say "we run AI on-device, offline" instead. |
| "Our OCR is more accurate than Google" | It's not, and judges know it. Don't claim technical superiority where you don't have it. Claim architectural superiority (privacy, offline). |
| "We're the first to do this" | VDScan exists. Be My Eyes exists. Your differentiator isn't "first" — it's "first fully offline, open-source, category-aware." |
| "We'll add X feature later" | Judges score what you built, not what you'll build. Mention future directions in Q&A, not in the pitch. |

---

## 8. Pre-Hackathon Preparation Checklist

### 8.1 Before the Hackathon Starts

- [ ] **Install and test the full stack locally**: Tesseract, OpenCV, ONNX Runtime, pyttsx3/espeak. Confirm OCR → TTS pipeline works end-to-end on your development machine.
- [ ] **Prepare demo documents**: Print 3 documents — a mock bank statement, a mock prescription, a mock legal letter. Use clear, large print. Test OCR on all three.
- [ ] **Build the pipeline skeleton**: Camera capture → preprocessing → OCR → TTS. Even if it's rough, have the end-to-end flow working before you arrive.
- [ ] **Assign team roles**: One person on image preprocessing (OpenCV), one on OCR integration (Tesseract/ONNX), one on TTS + audio UX, one on the demo + pitch. Four people, four lanes.
- [ ] **Write the 3-minute pitch script** (use Section 6.2). Practice it out loud at least 3 times.
- [ ] **Prepare backup demo video**: Record a 90-second screen capture of a working demo. Save to USB. This is your insurance policy.
- [ ] **Research the specific ICHack prize categories** for your year on the Devpost page. Identify which 2-3 categories you'll enter.
- [ ] **Prepare the architecture diagram slide**: Clean, simple, showing the pipeline with "no cloud" emphasized.

### 8.2 During the Hackathon (24 Hours)

| Hour | Focus |
|---|---|
| 0–2 | Set up environment, confirm pipeline works end-to-end with a simple test |
| 2–6 | Image preprocessing (OpenCV: deskew, denoise, perspective correction). This is where OCR accuracy lives or dies. |
| 6–10 | Tesseract OCR integration + tuning. Test on all three document types. |
| 10–14 | ONNX document classifier (even a simple keyword-based one). Category-aware reading templates. |
| 14–16 | TTS integration + audio UX (audio prompts, document summary, voice commands if time allows). |
| 16–18 | Polish the demo flow. Make it one smooth gesture: open app → scan → hear. |
| 18–20 | Build the pitch slide deck (3-4 slides max: problem, demo, architecture, impact). |
| 20–22 | Practice the pitch out loud. Time it. Fix the flow. Practice the airplane mode moment. |
| 22–24 | Final testing, backup video recording, charge devices, prepare for judging. |

### 8.3 Pitch Slides (Keep to 4)

1. **Title slide**: "SightLine — Your eyes, offline. Nothing leaves your phone." + team names
2. **Architecture slide**: Pipeline diagram with "no cloud" callout. Include the comparison table (SightLine vs. cloud apps).
3. **Impact slide**: Key statistics (2M UK, 43M global, 90% in low-income countries) + the privacy claim
4. **Closing slide**: "SightLine. Your eyes, offline." + GitHub repo link

---

## 9. Q&A Anticipation Sheet

Judges will ask questions after your demo. Prepare answers for these:

### Q: "Why not just use existing apps like Seeing AI or Be My Eyes?"
> "Those are excellent apps, but they send images to the cloud. Be My Eyes connects you to a human volunteer who can see your documents. Seeing AI uses Microsoft's cloud OCR. For reading a restaurant menu, that's fine. For reading your bank statement or your HIV test results, it's a privacy violation. SightLine is for the documents that are too sensitive to send to a server."

### Q: "Tesseract accuracy is lower than cloud OCR. How do you handle that?"
> "Tesseract accuracy on well-preprocessed images is 85-95%, which is usable for document reading. We invest in image preprocessing — deskewing, denoising, perspective correction with OpenCV — which is where most accuracy gains come from. And we trade a few percentage points of accuracy for a fundamental privacy guarantee. For a blind person reading their own bank statement, 'good enough and private' beats 'perfect but exposed.'"

### Q: "What about documents with complex layouts or handwriting?"
> "In this 24-hour prototype, we handle printed text documents — bank statements, prescriptions, typed letters. Handwriting and complex layouts are a known limitation of Tesseract and would require a fine-tuned ONNX model, which is a natural next step. But the 90% use case — printed financial, medical, and legal documents — is solvable today with Tesseract."

### Q: "How would you monetize this?"
> "Open-source core with a freemium mobile app. The base scanning and TTS would be free. Premium features — document-specific reading modes, multi-language support, PDF export — would be a one-time purchase, similar to VDScan's $9.99 model. No subscription, because that would require a server and break the offline thesis."

### Q: "Who are your users, really? How do blind people use a camera app?"
> "Blind users are adept at using phone cameras — apps like Seeing AI and KNFB Reader have trained this behavior for years. The key UX challenge is aiming the camera correctly without visual feedback. We address this with audio guidance: 'Document detected. Hold steady. Scanning...' — spoken prompts that guide the user through capture. This is a solved UX pattern in the accessibility community."

### Q: "What was the hardest technical challenge?"
> "Making the entire pipeline run in real-time on-device. Tesseract OCR on a phone is CPU-intensive. We had to tune image resolution, preprocessing parameters, and Tesseract's page segmentation mode to get acceptable latency. We also had to build the document classifier without any LLM — we're offline, so no Claude, no GPT. That constraint forced us to build a lightweight ONNX classifier, which was actually a better engineering decision."

### Q: "Is this actually new? VDScan already does on-device OCR for blind users."
> "VDScan is a great app and validates that the market exists. Our differentiators are: (1) fully open-source — VDScan is proprietary and closed, so you can't verify its privacy claims; (2) category-aware reading — we don't just dump OCR text into TTS, we recognize document types and read them intelligently; (3) zero-cost and globally accessible — VDScan costs $9.99, which is out of reach for the 90% of blind people in low-income countries. SightLine is free and open-source."

### Q: "What would you build next if you had more time?"
> "Three things: (1) A fine-tuned ONNX OCR model to improve accuracy on complex layouts and handwriting. (2) Multi-language TTS — espeak supports many languages, so the infrastructure is there. (3) Voice command integration — 'read the total,' 'skip to the next section,' 'what's the dosage' — for fully hands-free document navigation."

---

## 10. Prize Category Targeting

### 10.1 Category Priority Matrix

| Category | Fit | Strategy | Effort Required |
|---|---|---|---|
| **IMC Trading — Positive Social Impact** | ★★★★★ | Primary target. Lead with the user story, the dignity argument, and the global accessibility angle. | Emphasize impact in pitch and Q&A |
| **Bending Spoons — The Occam Award** | ★★★★ | Emphasize simplicity: "One app, one camera, one speaker. No cloud, no servers, no complexity." | Add a line about elegant minimalism |
| **DoCSoc — Accessibility** (if offered) | ★★★★★ | Direct fit. If this category exists in your year, it's yours to win. | Self-evident |
| **DoCSoc — Best Hardware Hack** | ★★ | Only if you use a Raspberry Pi or external camera. Don't force it. | Would require hardware pivot |
| **DoCSoc — Best Developer Tooling** | ★★ | Frame as an extensible accessibility toolkit. Stretch fit. | Would need to show extensibility |
| **Best Use of Claude / ElevenLabs** | ★ | Do not enter. Contradicts offline thesis. | Would compromise core narrative |

### 10.2 The Multi-Category Strategy

You can enter multiple categories. Enter:
1. **IMC Trading — Positive Social Impact** (primary)
2. **Bending Spoons — The Occam Award** (secondary)
3. **DoCSoc Accessibility** (if available)

Do not enter more than 3 — it dilutes your focus. For each category submission, tailor the one-line project description to match the category's emphasis:
- IMC Trading: "SightLine gives blind users the ability to independently read sensitive documents — banking, medical, legal — with zero data leaving their device."
- Occam Award: "SightLine is the simplest possible solution to document accessibility: point, scan, hear. No cloud, no API, no internet."
- Accessibility: "SightLine is a fully offline, open-source document scanner that reads banking, medical, and legal documents aloud for visually impaired users."

---

## Appendix: Key Statistics Reference Card

Print this. Keep it at your table. Use it in your pitch and Q&A.

| Stat | Source | When to Use |
|---|---|---|
| 2.2 billion people have vision impairment globally | WHO, 2024 | Opening hook |
| 43 million people are blind worldwide | IAPB Vision Atlas / Lancet GH | Market size |
| 295 million have moderate-to-severe visual impairment | IAPB Vision Atlas | Market size |
| 2 million people in the UK live with sight loss | RNIB | UK relevance |
| 322,638 registered blind/partially sighted in the UK | RNIB 2022-23 | UK specificity |
| 90% of people with vision loss live in low/middle-income countries | IAPB | Offline-as-equity argument |
| 60 million people will be blind by 2050 | Lancet Global Health | Growth trajectory |
| ICHack judging criteria: Creativity, Utility, Technical Achievement | ICHack Devpost | Frame your pitch around these three |
| ICHack 26 "Tech for Good" winner was offline-first medical dashboard (Relay) | ICHack 26 Devpost | Proves offline-first wins at ICHack |
| ICHack 25 accessibility winner was a visually impaired navigation app (Seen_ary) | ICHack 25 Devpost | Proves accessibility wins at ICHack |

---

## Final Word

Your project has three things that win hackathons at Imperial:
1. **A problem judges can feel** — a blind person who can't read their own bank statement
2. **A technically interesting solution** — fully offline OCR + TTS + document classification
3. **A memorable demo moment** — airplane mode, zero bytes, document reads aloud

Nail those three. Keep the scope tight. Practice the pitch until it's muscle memory. And when you walk up to present, remember: the demo is the pitch. Everything else is support.

Go win.