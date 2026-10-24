const { test } = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path");
const Plan = require("../js/speech_plan.js");
function transport(
  voices = [{ name: "Offline English", lang: "en-GB", localService: true }],
) {
  const synth = new EventTarget();
  let list = voices;
  synth.getVoices = () => list;
  synth.calls = [];
  synth.cancel = () => {
    synth.cancels = (synth.cancels || 0) + 1;
  };
  synth.resume = () => {};
  synth.speak = (u) => synth.calls.push(u);
  global.window = {
    speechSynthesis: synth,
    SpeechSynthesisUtterance: class {
      constructor(text) {
        this.text = text;
      }
    },
  };
  delete require.cache[require.resolve("../js/tts.js")];
  return {
    tts: require("../js/tts.js"),
    synth,
    voices(next) {
      list = next;
      synth.dispatchEvent(new Event("voiceschanged"));
    },
  };
}
test("receipt fields speak independently of unrelated unclear page lines", () => {
  const r = {
    category: "receipt",
    engine: { ocr: "ppocrv5" },
    speechText: "",
    fields: [
      {
        label: "Shop",
        value: "BOOK STORE",
        confidence: 0.86,
        sourceLine: "BOOK STORE",
      },
      {
        label: "Total",
        value: "90.00",
        confidence: 0.6,
        sourceLine: "Total 90.00",
      },
    ],
    receiptDiagnostics: { rejected: [{ key: "total" }] },
  };
  const p = Plan.plan(r);
  assert.equal(p.eligible, true);
  assert.match(p.text, /Shop: BOOK STORE/);
  assert.ok(!p.text.includes("90.00"));
  assert.match(p.text, /unclear/);
});
test("medical partial pages remain refused despite high OCR confidence", () =>
  assert.equal(
    Plan.plan({
      category: "medical",
      engine: { ocr: "ppocrv5" },
      speechText: "",
      ocrText: "Maximum 1200mg",
      ocrConfidence: 0.99,
    }).eligible,
    false,
  ));
test("confirmed complete prescription retains original medicine-dose relationships", () => {
  const text = "Amoxicillin 500 mg. Take one capsule twice daily.";
  assert.equal(
    Plan.plan({
      category: "medical",
      engine: { ocr: "ppocrv5" },
      speechText: text,
    }).text,
    text,
  );
});
test("every chunk has handlers before speak and early errors reject", async () => {
  const { tts, synth } = transport();
  synth.speak = (u) => {
    assert.equal(typeof u.onerror, "function");
    u.onerror({ error: "audio-hardware" });
  };
  await assert.rejects(tts.speak("Receipt."), { code: "audio-hardware" });
  assert.equal(tts.getState().status, "error");
});
test("stop settles without browser events and late events cannot affect retry", async () => {
  const { tts, synth } = transport();
  const one = tts.speak("First read.");
  const old = synth.calls[0];
  tts.stop();
  assert.equal((await one).status, "stopped");
  const two = tts.speak("Second read.");
  old.onend();
  old.onerror({ error: "interrupted" });
  assert.equal(tts.speaking(), true);
  const next = synth.calls.at(-1);
  next.onstart();
  next.onend();
  assert.equal((await two).status, "ended");
  assert.equal(tts.getState().status, "ended");
});
test("all chunks must end; earlier chunks cannot silently fail", async () => {
  const { tts, synth } = transport();
  const p = tts.speak("A readable sentence. ".repeat(25));
  synth.calls[0].onstart();
  synth.calls[0].onend();
  assert.equal(tts.speaking(), true);
  assert.equal(synth.calls.length, 2);
  synth.calls[1].onerror({ error: "synthesis-failed" });
  await assert.rejects(p, { code: "synthesis-failed" });
});
test("voice discovery can complete after an initially empty list", async () => {
  const { tts, synth, voices } = transport([]);
  const p = tts.speak("Receipt.", { voiceWaitMs: 80 });
  voices([{ name: "English", lang: "en-US", localService: true }]);
  await new Promise((r) => setImmediate(r));
  synth.calls[0].onstart();
  synth.calls[0].onend();
  assert.equal((await p).status, "ended");
});
test("missing local voice gives a specific retryable error, never network speech", async () => {
  const { tts, synth } = transport([
    { name: "Online", lang: "en-GB", localService: false },
  ]);
  await assert.rejects(tts.speak("Receipt."), { code: "no-local-voice" });
  assert.equal(synth.calls.length, 0);
});
test("Mandarin does not masquerade as a Cantonese voice", () => {
  const { tts } = transport([
    { name: "Mandarin", lang: "zh-CN", localService: true },
  ]);
  assert.equal(tts.hasVoice("zh-HK"), false);
});
test("missing start events fail instead of pretending playback finished", async () => {
  const { tts } = transport();
  await assert.rejects(tts.speak("Receipt.", { startTimeoutMs: 10 }), {
    code: "start-timeout",
  });
  assert.equal(tts.getState().status, "error");
});

test("chunking preserves decimal amounts and medicine strengths", () => {
  const { tts } = transport();
  const text = "Total: £9.00. Dose: 2.5 mg.";
  assert.equal(tts.chunkText(text, 180).join(" "), text);
});

test("an empty voice catalog reports discovery delay and allows retry", async () => {
  const { tts, synth, voices } = transport([]);
  await assert.rejects(tts.speak("Receipt.", { voiceWaitMs: 5 }), {
    code: "voices-not-ready",
  });
  voices([{ name: "Offline English", lang: "en-GB", localService: true }]);
  const retry = tts.speak("Receipt.");
  synth.calls.at(-1).onstart();
  synth.calls.at(-1).onend();
  assert.equal((await retry).status, "ended");
});

test("native iOS bridge discovers voices and completes speech by native events", async () => {
  const messages = [];
  global.window = {
    webkit: {
      messageHandlers: {
        sightlineSpeech: { postMessage: (message) => messages.push(message) },
      },
    },
  };
  delete require.cache[require.resolve("../js/tts.js")];
  const tts = require("../js/tts.js");
  assert.equal(tts.available(), true);
  const voiceLoad = tts.loadVoices();
  assert.equal(messages.at(-1).action, "voices");
  tts.nativeEvent({
    type: "voices",
    voices: [
      { name: "Siri English", lang: "en-GB", localService: true },
      { name: "Sinji", lang: "zh-HK", localService: true },
      { name: "Mandarin", lang: "zh-CN", localService: true },
    ],
  });
  await voiceLoad;
  assert.equal(tts.hasVoice("zh-HK"), true);
  assert.equal(tts.pickVoice("zh-HK").name, "Sinji");
  const reading = tts.speak("The receipt total is clear.", { lang: "en-GB" });
  const command = messages.at(-1);
  assert.equal(command.action, "speak");
  assert.equal(command.text, "The receipt total is clear.");
  tts.nativeEvent({
    type: "started",
    id: command.id,
    outputKind: "Speaker",
    outputName: "iPhone speaker",
    outputVolume: 0.6,
  });
  assert.equal(tts.getState().status, "speaking");
  assert.equal(tts.getState().outputKind, "Speaker");
  assert.equal(tts.getState().outputVolume, 0.6);
  tts.nativeEvent({ type: "finished", id: command.id });
  assert.equal((await reading).status, "ended");
  const cantonese = tts.speak("呢張收據。", { lang: "zh-HK" });
  const cantoneseCommand = messages.at(-1);
  assert.equal(cantoneseCommand.lang, "zh-HK");
  tts.nativeEvent({ type: "started", id: cantoneseCommand.id });
  tts.nativeEvent({ type: "finished", id: cantoneseCommand.id });
  assert.equal((await cantonese).status, "ended");
});

test("native bridge refuses Mandarin substitution and stop settles for replay", async () => {
  const messages = [];
  global.window = {
    webkit: {
      messageHandlers: {
        sightlineSpeech: { postMessage: (message) => messages.push(message) },
      },
    },
  };
  delete require.cache[require.resolve("../js/tts.js")];
  const tts = require("../js/tts.js");
  const voiceLoad = tts.loadVoices();
  tts.nativeEvent({
    type: "voices",
    voices: [{ name: "Mandarin", lang: "zh-CN", localService: true }],
  });
  await voiceLoad;
  assert.equal(tts.hasVoice("zh-HK"), false);
  await assert.rejects(tts.speak("廣東話。", { lang: "zh-HK" }), {
    code: "no-local-voice",
  });

  tts.nativeEvent({
    type: "voices",
    voices: [{ name: "English", lang: "en-GB", localService: true }],
  });
  const reading = tts.speak("Retry this reading.");
  tts.stop();
  assert.equal((await reading).status, "stopped");
  assert.equal(messages.at(-1).action, "stop");
});

test("native empty voice catalog is queried again when iOS voices arrive late", async () => {
  const messages = [];
  global.window = {
    webkit: {
      messageHandlers: {
        sightlineSpeech: { postMessage: (message) => messages.push(message) },
      },
    },
  };
  delete require.cache[require.resolve("../js/tts.js")];
  const tts = require("../js/tts.js");
  const initial = tts.loadVoices();
  tts.nativeEvent({ type: "voices", voices: [] });
  assert.deepEqual(await initial, []);

  const retry = tts.loadVoices();
  assert.equal(messages.filter((message) => message.action === "voices").length, 2);
  tts.nativeEvent({
    type: "voices",
    voices: [{ name: "Sinji", lang: "zh-HK", localService: true }],
  });
  await retry;
  assert.equal(tts.hasVoice("zh-HK"), true);
});

test("iOS host connects the speech bridge to its WKWebView for replies", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../../ios/Sources/SightLineApp.swift"),
    "utf8",
  );
  assert.match(source, /navigation\.speechBridge\.webView\s*=\s*view/);
});
