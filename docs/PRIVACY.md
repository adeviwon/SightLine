# Privacy: how SightLine keeps its promise

SightLine's central claim is that document images never leave the device. This
document explains how that is enforced, how you verify it, and where the
boundary of the claim is.

---

## 1. The claim

> After the first successful load, SightLine makes **zero** network requests.
> No document image, no OCR text, no extracted field, and no identifier is
> transmitted anywhere.

This is stronger than "we don't have a backend." There is no backend to
receive anything. The claim is architectural.

---

## 2. Why it matters for these specific documents

SightLine targets three document types, and each one is sensitive in a way
that makes cloud OCR a genuine problem:

| Document | What a cloud OCR learns | Consequence of disclosure |
|---|---|---|
| **Prescription** | The medication, therefore the condition | Discriminatory treatment by insurers, employers, landlords. In many jurisdictions, protected health information. |
| **Bank statement** | Income, balance, debts, overdraft use | Financial discrimination; a data broker learns a vulnerable person's financial state before they apply for anything. |
| **Legal correspondence** | Disputes, tenancy problems, proceedings | Exposure to a counterparty's lawyer, insurer, or court. |

The users most in need of this tool are the users with the least bargaining
power to consent to data collection. Every existing alternative asks them to
trade privacy for independence. SightLine removes the trade.

---

## 3. Six enforcement mechanisms

### 3.1 Service-worker precache, cache-first

`app/sw.js` precaches every asset during installation and serves **all**
subsequent requests from the cache. There is no `fetch` handler that falls
back to the network.

```javascript
// The whole network policy, in essence:
self.addEventListener('fetch', (event) => {
  event.respondWith(caches.match(event.request));   // cache only, no network
});
```

If the app cannot find an asset in the cache, it fails visibly rather than
reaching out.

### 3.2 No analytics, no telemetry, no crash reporting

There is no measurement SDK, no error-reporting service, and no usage ping.
This is unusual and it is deliberate: any of them would transmit document
metadata to a third party.

**No fonts from Google Fonts.** All type is system-stack. A web font request
leaks the user's IP, user agent, and page path to a third party on every load.

### 3.3 Everything runs in the browser

No server-side inference, no cloud OCR fallback, no "send us the hard ones"
escape hatch. The models are ONNX graphs executed by WebAssembly in the page's
own context.

### 3.4 No storage of document content

Documents are processed in memory and released. SightLine does not persist
scans, OCR text, or extracted fields to disk. Refreshing the page loses
everything — which is the correct default for medical and financial documents.

### 3.5 Camera stays in the tab

`getUserMedia` gives the page a video stream. The frames go to a `<canvas>`,
then to the pipeline. There is no upload target, because there is nowhere to
upload to.

### 3.6 No analytics manifest entries

`app/manifest.json` declares no third-party endpoints. No `connect-src`, no
external origins, no beacons.

---

## 4. How to verify it yourself

Do not take the project's word for it. Three independent checks:

### Check A — read the source

```
grep -rn "fetch(\|XMLHttpRequest\|sendBeacon\|WebSocket\|EventSource" app/
```

Every hit should be either a **precache** fetch during install, or a **local
file** read. None should construct an external URL.

```bash
grep -rnoE "https?://[a-zA-Z0-9.-]+" app/js/ app/index.html app/sw.js | \
  grep -v "www.w3.org"     # w3.org are SVG/XML namespaces, not network calls
```

Expect: no third-party hosts.

### Check B — break the network

1. Load the app online. Wait for `ENGINE READY`.
2. Turn on **airplane mode**.
3. Force-quit the app and reopen it.
4. Scan a document.

It works. This is the demo moment.

### Check C — capture the packets

On macOS, while scanning in airplane mode (or online):

```bash
sudo tcpdump -i en0 -n 'tcp or udp' | grep -viE 'arp|mdns|multicast'
```

Scan three documents. SightLine generates **no packets**. The traffic you see
is the operating system's background chatter, not the app.

In Chrome DevTools: **Network tab → filter by `fetch`/`xhr` while scanning.**
Empty.

---

## 5. Where the claim ends

Honesty requires stating the boundaries:

**First load requires network.** The OCR engine and English language data are
~41 MB. They are downloaded once and cached. After that, no network.

**iOS may evict the cache.** If the app is not opened for weeks, iOS can purge
the service worker cache. One online reload restores it. The app tells the
user when this happens rather than failing silently.

**The device is the trust boundary.** If the phone is compromised at the OS
level, no app-level guarantee holds. SightLine cannot defend against a rooted
device or malicious OS.

**Hosting is still a choice.** If you serve SightLine from a third-party host,
that host sees which IP fetched the app. That reveals *someone* installed a
document reader — not what they scanned. For the strongest guarantee, host it
yourself or bundle it.

---

## 6. Threat model

| Threat | Mitigation | Residual risk |
|---|---|---|
| Document image exfiltrated | No network code path at all | None in-app |
| OCR text leaked | Processed in memory, never stored | Memory forensics on a compromised device |
| Third-party script reads the canvas | No third-party scripts | Supply-chain compromise of a vendored file |
| Model weights stolen | MIT licence, publicly available | N/A — they are free to copy |
| Fingerprinting via fonts | System font stack only | Browser-level fingerprinting outside app control |
| Update channel as an attack vector | No remote update mechanism | Cannot ship security fixes without a redeploy |

That last row is a real trade-off, and worth naming: **no update channel means
no silent patching.** If a vulnerability were found, users would need to
reinstall. For a hackathon project that is the right default — a silent update
mechanism is exactly the kind of thing that quietly reintroduces network calls.

---

## 7. Comparison

| | SightLine | Cloud OCR apps | Volunteer-based (Be My Eyes) |
|---|---|---|---|
| Document image leaves device | **Never** | Yes, to vendor servers | Yes, to a volunteer's phone |
| Needs internet after install | **No** | Yes | Yes |
| Account required | **No** | Usually | Yes |
| Data retention | None | Vendor policy | Volunteer device |
| Verifiable offline | **Yes — airplane mode** | No | No |
| Who can read your prescription | **Nobody** | The vendor | A stranger |

The last row is the one that matters. For a blind user photographing a
prescription, "a volunteer saw my medication" is not a privacy policy detail.
It is the whole problem.
