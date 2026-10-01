# Demo runbook

The demo is where the hackathon is won. This is the exact sequence, timed, with
the failure mode for each step called out.

---

## Before you present (T−15 minutes)

```
□  Phone charged above 80%, airplane mode toggle accessible
□  App installed and open once — confirm "ENGINE READY"
□  Three physical test documents on the table, in this order:
      1. prescription   (clean — the reliable win)
      2. bank statement (slightly crumpled)
      3. prescription   (deliberately blurry — the ML story)
□  Backup phone with the app already installed
□  Screen recording running on the laptop (insurance)
□  Slide deck open, results table visible on slide 6
```

**Order matters.** Start with the clean document. A judge who sees one
confident success in the first 20 seconds will watch the rest. Do not open with
the hard case.

---

## The demo (4 minutes)

### 0:00 — The problem (15 s, no slides)

> "Thirty-one million blind people can't read their own medication label.
> Every app that helps them uploads the document to a cloud server. For a
> prescription, that's a privacy line they shouldn't have to cross just to be
> independent."

### 0:15 — Show the app, empty state (20 s)

Open SightLine. Point at the camera.

> "This runs entirely on the phone. No account, no server."

### 0:35 — Scan #1: the clean prescription (45 s)

Hold the phone over the document. Tap **Scan**.

> "It restores the image, reads it, works out it's a prescription, pulls out
> the medication and the dosage, and reads it back."

Then **tap Listen**. Let the audio finish. Do not talk over it.

> "Amoxicillin, 500 milligrams, one capsule three times daily, for seven days."

**This is the moment the room decides you are real.** Do not rush it.

### 1:20 — Scan #2: the bank statement (45 s)

> "Same thing on a bank statement — account number, sort code, balance."

Let it extract. This proves it is not a single-purpose demo.

### 2:05 — Scan #3: the blurry prescription (60 s)

**The ML story.** Hold the phone shakily, or use the deliberately-blurred print.

> "That one is blurry — which is what actually happens when you hold a phone
> over a document one-handed, which is the situation our users are in every
> time. We trained a small PyTorch network to undo exactly that."

If it recovers the dosage — say:

> "That's the model working. Thirty-eight thousand parameters, running on the
> phone."

**If it does not recover it, that is also a good demo:**

> "It can't. There's a hard limit — you can't reconstruct a stroke the lens
> smeared away. So instead of guessing the number, it tells you it couldn't
> read it and asks you to retake. For someone who can't see the label,
> a wrong dosage is the worst possible outcome."

Judges reward knowing your model's limits more than a lucky demo.

### 3:05 — Airplane mode (40 s)

Turn on airplane mode on the phone.

> "Let me show you the part that matters."

Scan again. It works.

> "It's not caching the result — this is a document we haven't scanned. Every
> stage is on the device: the restoration network, the OCR engine, the
> classifier. Nothing is sent anywhere, because there is nowhere to send it.
> You can verify that — the service worker has no network fallback."

### 3:45 — The proof (15 s)

If you have it ready, show the packet capture on your slide:

```
$ sudo tcpdump -i en0 -n | grep -i sightline
   (nothing — for the entire session)
```

Otherwise say: "The service worker source is in the repo. It precaches
everything and never falls back to the network. It's about forty lines."

---

## Contingencies

| What goes wrong | What you do |
|---|---|
| Camera will not open | Do not debug on stage. Say "let me use the prepared sample" and use a gallery image. |
| App is slow / hung | Force-quit, reopen. The models reload from cache in ~2 s. |
| Wi-Fi dropped | Airplane mode still works. **Turn it on deliberately** and sell it as the feature. |
| First scan reads nothing | Retake. Hand-shake is normal. If it fails twice, move to scan #2. |
| Demo phone died | Switch to the backup phone mid-sentence: "Let me use the other phone." Do not apologise at length. |
| Judge interrupts with a question | Answer it fully, then say "and then it reads it aloud" and continue. |

**Rule: never spend more than 20 seconds debugging on stage.** Switch to the
backup path immediately.

---

## The 90-second version

If you get only 90 seconds:

```
0:00  Problem: blind users can't read prescriptions; every app uploads them.
0:20  Demo: scan a prescription → it reads the dosage aloud.
0:50  Airplane mode → it still works. Everything is on-device.
1:10  "Two trained models. 38K params for restoration, MiniLM for
       classification. Here's the held-out accuracy."
```

---

## The slide that must be ready

**Slide 6 — results.** Keep it open during Q&A. Every number on it must be
reproducible with `bash run.sh eval`. If a judge asks "how do you know?",
the answer is one command and a file path.

---

## After the demo

Do not stop talking. The single most useful thing you can do in the 10 seconds
after a demo is:

> "If you'd like to run it — `git clone`, `bash setup_env.sh`, `bash run.sh all`.
> It retrains both models in about forty minutes on a laptop CPU and prints the
> same numbers I'm showing you."

That sentence converts a judge into a person who can verify your claims.
