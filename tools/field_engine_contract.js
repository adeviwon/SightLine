/**
 * SightLine — the field engine's contract.
 *
 * app/js/field_engine.js is a COMPOSITION: field_spot + field_guard, wired into
 * the path Lane N measured. Its two halves already have golden-vector parity
 * proofs of their own (app/tools/field_spot_contract.js, 2175 vectors;
 * app/tools/field_guard_contract.js, 898). Those prove the PARTS. This file
 * proves the WIRING: the composition on synthetic pages whose lines are known, so
 * a failure names the behaviour rather than hiding behind a vector table.
 *
 * Every case here has an OUTCOME, not just "no exception":
 *   - a label with a valid value SPEAKS, with the value the page printed
 *   - a label with an impossible value REFUSES, and says why
 *   - a label with no value on its line and none on the next REFUSES
 *   - a line with no label yields NOTHING
 *   - an empty page yields NOTHING and does not crash
 *   - '15 January 2024' (the date class of bug the app reads correctly today)
 *     is NOT refused
 *   - free text is OFF by default, and the same label SPEAKS when it is opted in
 *   - the box geometry direction works, and refuses when the next line is ABOVE
 *   - the regex path reaches the policy (a contradiction REFUSES both values)
 *
 * Run:  node app/tools/field_engine_contract.js      (exit 0 = all passed)
 */

"use strict";

const E = require("../js/field_engine.js");
const FG = require("../js/field_guard.js");

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log("  ok    " + name); }
  else { fail++; console.log("  FAIL  " + name + (detail === undefined ? "" : "  -> " + detail)); }
}

/** The decision for one field, or null. */
function dec(res, field) {
  const hits = res.decisions.filter((d) => d.field === field);
  return hits.length ? hits[0] : null;
}

function show(v) { return JSON.stringify(v); }

console.log("field engine contract — synthetic pages, known lines\n");

// ── 1. a label with a valid value speaks ─────────────────────────────────────
{
  const r = E.extract(["Sort Code: 40-11-04"], { docType: "banking" });
  const d = dec(r, "sort_code");
  check("valid value speaks (Sort Code: 40-11-04)",
    !!d && d.decision === "speak" && d.value === "40-11-04",
    d ? d.decision + " " + show(d.value) + " why=" + d.why : "no decision");
  check("  ...and the read carried its own validation",
    r.fields.sort_code && r.fields.sort_code.ok === true &&
    r.fields.sort_code.provenance === "same-line-right",
    show(r.fields.sort_code));
}

// ── 2. a label with an impossible value refuses, WITH THE REASON ─────────────
{
  const r = E.extract(["Sort Code: 40-11-0"], { docType: "banking" });
  const d = dec(r, "sort_code");
  check("impossible value refuses (5-digit sort code)",
    !!d && d.decision === "refuse" && d.value === "40-11-0",
    d ? d.decision + " " + show(d.value) : "no decision");
  check("  ...and the reason is the validator's, not a generic one",
    !!d && /6 digits required, got 5/.test(d.why),
    d ? d.why : "no decision");
}

// ── 3. a label with no value on the line and none on the next refuses ────────
{
  const r = E.extract(["Account Holder:", "   "], { docType: "banking" });
  const d = dec(r, "account_holder");
  check("label with no value on the line and none on the next refuses",
    !!d && d.decision === "refuse" && d.value === "",
    d ? d.decision + " " + show(d.value) + " why=" + d.why : "no decision");
  check("  ...and the refusal names the missing value",
    !!d && /no value present/.test(d.why),
    d ? d.why : "no decision");
  check("  ...and the read itself says why it found nothing",
    r.fields.account_holder && r.fields.account_holder.ok === false &&
    /no value found after label/.test(r.fields.account_holder.reason),
    show(r.fields.account_holder));
  check("  ...and nothing was spoken", r.spoken.length === 0, show(r.spoken));
}

// ── 4. a line with no label yields nothing ───────────────────────────────────
{
  const r = E.extract(["The quick brown fox jumps over"], {});
  check("a line with no label yields nothing",
    r.spots.length === 0 && r.decisions.length === 0 && r.spoken.length === 0,
    "spots=" + r.spots.length + " decisions=" + r.decisions.length);
}

// ── 5. an empty page yields nothing and does not crash ───────────────────────
{
  let ok = true, detail = "";
  for (const [label, page] of [["[]", []], ["''", ""], ["null", null],
                               ["['']", [""]], ["whitespace", ["   ", "\t"]]]) {
    try {
      const r = E.extract(page, { docType: "banking" });
      if (r.decisions.length !== 0 || r.spoken.length !== 0) {
        ok = false; detail += label + " produced " + show(r.spoken) + " ";
      }
    } catch (e) { ok = false; detail += label + " threw " + (e && e.message) + " "; }
  }
  check("an empty page yields nothing and does not crash", ok, detail);
}

// ── 6. the DATE CLASS OF BUG must not be refused ─────────────────────────────
// '15 January 2024' is a value app/js/pipeline.js reads correctly today (its own
// legal Date pattern emits it), so refusing it would refuse a value the app gets
// right — the regression that motivated field_guard's long-form date branch.
{
  const value = "15 January 2024";
  check("the guard itself accepts '15 January 2024'",
    FG.validate("date", value)[0] === true, show(FG.validate("date", value)));
  const r = E.extract(["Date: " + value], { docType: "legal" });
  const d = dec(r, "date");
  check("the engine SPEAKS '15 January 2024' (date class of bug)",
    !!d && d.decision === "speak" && d.value === value,
    d ? d.decision + " " + show(d.value) + " why=" + d.why : "no decision");
  // and the neighbours it must still reject, so this is not a blanket accept
  check("  ...while an impossible date still refuses",
    E.extract(["Date: 31/02/2026"], { docType: "legal" }).decisions
      .some((d) => d.decision === "refuse" && /not a calendar date/.test(d.why)),
    show(E.extract(["Date: 31/02/2026"], { docType: "legal" }).decisions));
}

// ── 7. free text is OFF by default ───────────────────────────────────────────
{
  check("free text is OFF by default (the engine's own default)",
    E.DEFAULT_OPTS.speakFreeText === false, show(E.DEFAULT_OPTS));
  const off = E.extract(["Patient: Jane Doe"], { docType: "prescription" });
  const dOff = dec(off, "patient");
  check("a free-text field is REFUSED under the default",
    !!dOff && dOff.decision === "refuse" && /free text/.test(dOff.why),
    dOff ? dOff.decision + " why=" + dOff.why : "no decision");
  check("  ...and it is refused though it validated fine",
    off.fields.patient && off.fields.patient.ok === true, show(off.fields.patient));
  const on = E.extract(["Patient: Jane Doe"],
                       { docType: "prescription", speakFreeText: true });
  const dOn = dec(on, "patient");
  check("  ...and the SAME page SPEAKS it when free text is opted in",
    !!dOn && dOn.decision === "speak" && dOn.value === "Jane Doe",
    dOn ? dOn.decision + " " + show(dOn.value) : "no decision");
}

// ── 8. the box geometry direction ────────────────────────────────────────────
{
  const down = E.extract([
    { text: "Sort Code:", box: [0, 0, 80, 10] },
    { text: "40-11-04", box: [0, 14, 80, 24] },
  ], { docType: "banking" });
  const dDown = dec(down, "sort_code");
  check("a value on the line BELOW the label is read across the break",
    !!dDown && dDown.decision === "speak" && dDown.value === "40-11-04" &&
    down.fields.sort_code.provenance === "next-line",
    dDown ? dDown.decision + " " + show(dDown.value) +
            " provenance=" + (down.fields.sort_code || {}).provenance : "no decision");

  const up = E.extract([
    { text: "40-11-04", box: [0, 0, 80, 10] },
    { text: "Sort Code:", box: [0, 14, 80, 24] },
  ], { docType: "banking" });
  const dUp = dec(up, "sort_code");
  check("  ...and a line ABOVE the label is NEVER taken as its value",
    !!dUp && dUp.decision === "refuse" && dUp.value === "",
    dUp ? dUp.decision + " " + show(dUp.value) : "no decision");

  // The app's own lines have NO box (the OCR layer returns a joined string), so
  // the reading-order fallback is what ships. Pinned here as a documented
  // property of the input, not discovered in production.
  const noBox = E.extract(["Sort Code:", "40-11-04"], { docType: "banking" });
  const dNoBox = dec(noBox, "sort_code");
  check("a box-less line falls back to the next line in reading order",
    !!dNoBox && dNoBox.decision === "speak" && dNoBox.value === "40-11-04",
    dNoBox ? dNoBox.decision + " " + show(dNoBox.value) : "no decision");
}

// ── 9. the regex path reaches the policy ─────────────────────────────────────
// A contradiction between the two paths must REFUSE both — that is field_policy's
// R3, and it only fires if regexFields is actually handed to decide().
{
  const r = E.extract(["Sort Code: 40-11-04"],
                      { docType: "banking", regexFields: { sort_code: "40-22-99" } });
  const d = dec(r, "sort_code");
  check("a regex/spot contradiction refuses BOTH values (R3)",
    !!d && d.decision === "refuse" && /disagree/.test(d.why),
    d ? d.decision + " why=" + d.why : "no decision");
  check("  ...and agreement is not refused",
    dec(E.extract(["Sort Code: 40-11-04"],
                  { docType: "banking", regexFields: { sort_code: "401104" } }),
        "sort_code").decision === "speak",
    show(dec(E.extract(["Sort Code: 40-11-04"],
                       { docType: "banking", regexFields: { sort_code: "401104" } }),
             "sort_code")));
}

console.log("\nfield engine contract: " + (pass + fail) + " checks, " +
            pass + " passed, " + fail + " FAILED");
process.exit(fail ? 1 : 0);
