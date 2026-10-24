# Aria document parser source

`fields.mjs` and `taxonomy.mjs` were copied from `~/aria-reader`'s
existing working tree (base commit `273747a`) on 2026-10-24. Only import suffixes
were changed. That working tree already included OCR confidence caps and local
word evidence. Its unrelated files and completed work remain untouched.

`config.mjs` exports the only setting this pure parser reads: the unchanged
0.75 confidence floor. No Aria application controller, UI, model weights,
Tesseract dependency or vendor runtime is imported.

`../receipt.mjs` is the SightLine adapter. It prioritises strong bank statement
evidence, gates fields against complete clear source lines, refuses unlabelled
phone numbers and conflicting totals, and retains literal decimal/currency text.
MiniLM's four-class category order is unchanged.
