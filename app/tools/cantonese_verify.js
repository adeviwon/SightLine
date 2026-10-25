#!/usr/bin/env node
/**
 * cantonese_verify.js — prove the Cantonese output path works IN THE REAL BROWSER.
 *
 * WHAT THIS IS FOR. `node app/js/say_cantonese.js` proves the module's logic, but not that the
 * app loads it, not that the control is wired, and not that the SAFETY property survives being
 * served through the service worker. This drives WebKit (the engine the product targets) and
 * asserts the three things that matter:
 *
 *   1. the module is reachable from the app's own script order,
 *   2. a KNOWN drug name becomes a Cantonese sentence,
 *   3. an UNKNOWN drug name and a noise read are REFUSED, not guessed -- the failure mode this
 *      product exists to prevent, asserted in the oil the user actually runs.
 *
 * Usage: SMOKE_BASE=http://127.0.0.1:8123/ node app/tools/cantonese_verify.js [--engine=webkit]
 */
const { chromium, webkit } = require("./pw_core.js")();
const BASE = process.env.SMOKE_BASE || "http://127.0.0.1:8123/";
const engine = (process.argv.find((a) => a.startsWith("--engine=")) || "--engine=webkit").split("=")[1];

(async () => {
  const browser = await (engine === "chromium" ? chromium : webkit).launch();
  const page = await browser.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(String(e)));
  await page.goto(BASE, { waitUntil: "load" });
  await page.waitForTimeout(600);

  let pass = 0, fail = 0;
  const check = (name, ok, detail) => {
    if (ok) { pass++; console.log("  ok    " + name); }
    else { fail++; console.log("  FAIL  " + name + (detail ? "  -- " + detail : "")); }
  };

  // 1. reachable from the app's own script order
  const hasMod = await page.evaluate(() => typeof window.SayCantonese === "object" && typeof window.SayCantonese.build === "function");
  check("the module is loaded by the app itself (not just by node)", hasMod);

  // 2. the controls exist
  check("a spoken-language control exists", await page.evaluate(() => !!document.getElementById("speak-lang")));
  check("it offers Cantonese (zh-HK)", await page.evaluate(() => {
    const s = document.getElementById("speak-lang");
    return !!s && [...s.options].some((o) => o.value === "zh-HK");
  }));
  check("an on-screen element exists for the Cantonese text", await page.evaluate(() => !!document.getElementById("cantonese")));
  check("the Cantonese control starts hidden until there is a result", await page.evaluate(() => {
    const s = document.getElementById("speak-lang");
    return !!s && !s.checkVisibility();
  }));

  // 3. the safety property, in the browser
  const lex = { pairs: { AMOXICILLIN: { zh: "\u963f\u83ab\u897f\u6797" } } };
  const known = await page.evaluate((l) => window.SayCantonese.build("RX AMOXICILLIN 500MG CAPSULES", l), lex);
  check("a known name is spoken in Cantonese", known.ok === true && known.zh.indexOf("\u963f\u83ab\u897f\u6797") !== -1, JSON.stringify(known));
  check("its dose survives into the utterance", (known.doses || []).indexOf("500 MG") !== -1, JSON.stringify(known.doses));

  const unknown = await page.evaluate((l) => window.SayCantonese.build("RX ZORBEXIL 20MG TABLETS", l), lex);
  check("an UNKNOWN name is refused, never guessed", unknown.name === null, JSON.stringify(unknown.name));
  check("and the refusal names no drug", unknown.zh.indexOf("ZORBEXIL") === -1, unknown.zh);

  const noise = await page.evaluate((l) => window.SayCantonese.build("EAX CE E 0RO 121G000", l), lex);
  check("a noise read is refused outright", noise.ok === false, JSON.stringify(noise.zh));

  const form = await page.evaluate((l) => window.SayCantonese.build("TAKE ONE TABLET TWICE DAILY", { pairs: { TABLET: { zh: "\u559c\u5eb7\u9320" } } }), null);
  check("a dosage form is never offered as a drug name", form.name === null, JSON.stringify(form.name));

  // 4. does this engine even have a Cantonese voice? (records the demo's real constraint)
  const voices = await page.evaluate(() => (window.speechSynthesis && window.speechSynthesis.getVoices() || []).map((v) => v.lang));
  const zh = voices.filter((l) => String(l).toLowerCase().indexOf("zh") === 0);
  console.log("  note  voices in this engine: " + voices.length + ", Chinese: " + (zh.length ? zh.join(",") : "NONE -- the app will show the Cantonese text instead of faking a voice"));

  // 5. the honest copy is still honest
  // Privacy/refusal information now lives in the explicit Help disclosure.
  await page.locator("#btn-help").click();
  await page.locator("#help-dialog details summary").click();
  // innerText only sees ACTIVE panels; the consumer-UI redesign moved this copy into a
  // panel that is hidden until a scan, so the promise was in the shipped DOM and invisible
  // to the assertion. Read the whole document for the copy check, keep innerText for
  // anything that must be VISIBLE.
  const body = await page.evaluate(() => document.body.innerText);
  const dom = await page.evaluate(() => document.documentElement.outerHTML);
  check("the UI no longer promises Cantonese is only 'next' (it is built)", !/Cantonese is next/.test(body));
  // Assert the PROPERTY, not one wording of it: the consumer-UI redesign rephrased this copy
  // ("can't read reliably"), and a gate that pins the exact sentence fails on a reword rather
  // than on the promise being dropped.
  check("the UI still states it does not guess",
    /refus|won.t guess|will not guess|can.t read reliably|not filled in details/i.test(dom));
  check("no page errors", errs.length === 0, errs.join(" | "));

  console.log("\n  " + pass + "/" + (pass + fail) + " checks passed" + (fail ? "  -- " + fail + " FAILED" : ""));
  await browser.close();
  process.exit(fail ? 1 : 0);
})().catch((e) => { console.error("harness error:", e && e.message); process.exit(1); });
