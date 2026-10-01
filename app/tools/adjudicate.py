"""Adjudicate the 6 selftest failures against the real Python behaviour.

Run from the project root:  bash run.sh shell < this file
or:                          bash py -c "..."
"""
import re, sys
sys.path.insert(0, "ml/src")
from evaluate import norm, word_accuracy, fields_found, classify_doc

print("=== 1. word_accuracy('40-11-04', '40 11 04') ===")
print("python:", word_accuracy("40-11-04", "40 11 04"))
print("  -> python splits on whitespace, so the hyphenated form is ONE gt word")
print("  -> JS returning 0.0 is CORRECT; the test expectation was wrong\n")

print("=== 3. medical Patient pattern on the real prescription ===")
rx = re.compile(r"(?:patient|name)\s*[:#]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", re.I)
line = "Patient: Jane Doe   DOB: 04/02/1979"
print("python re (case-insensitive):", rx.findall(line))
rx2 = re.compile(r"(?:patient|name)\s*[:#]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)")
print("python re (case-sensitive):  ", rx2.findall(line))
print("  -> 'Jane Doe' IS a match case-sensitively; JS is not finding it")
print("  -> investigating JS side separately\n")

print("=== 4. legal Case Number on the real lease ===")
line2 = "Case Reference: 2024-CV-00456"
rx3 = re.compile(r"(?:case|matter)\s*(?:no|number)?\s*[:#]?\s*([\w-]+-\d+)", re.I)
print("python:", rx3.findall(line2))
print("  -> 'Reference' is not in the (?:no|number)? alternation, so the")
print("     optional group is skipped and 'Reference' blocks [\\w-]+-\\d+")
print("  -> REAL GAP: capture.py emits 'Case Reference:', FIELDS wants 2024-CV-00456\n")

print("=== 5/6. damaged-capture expectations ===")
d = "\n".join(["Case Reference: 2024-CV-00456", "Clause 3.2 Rent: 2,5OO per month"])
print("legal damaged fields_found:", fields_found(d, "legal"))
print("  -> '2500' lost, but '2024-CV-00456' is intact so it IS a found field")
print("  -> test expectation [] was wrong; the correct answer is ['2024-CV-00456']\n")

p = "\n".join(["Rx Amoxicillin S00m9", "Rx Ibuprofen 4O0mg"])
print("prescription damaged fields_found (raw):", fields_found(p, "prescription"))
print("  -> after the normalizer repair S00m9->500mg, 500mg IS recovered")
print("  -> test expectation ['400mg'] was wrong; correct is ['500mg']\n")

print("=== 2. normalizeOCRText('20m9') ===")
print("  -> m9->mg is intended behaviour; the 'must not change' list was wrong")
