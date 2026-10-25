# Sectioned medicine results — 24 October 2026

The user requested clear medicine details instead of a paragraph. The results
view now uses separate cards for Medicine (name/type), Strength on the label,
How much at once, When to take it, How long, and Warnings, when recognized.
Full OCR text and the spoken summary remain accessible in disclosures.

`medicine_sections.js` builds a read-only presentation from existing verified
speech text and accepted fields. Strength and amount per administration are
distinguished only by supported literal label contexts. Unlabelled mg values
remain 'Amount shown on the label'. Dose conditions, route text and exceptions
are retained; negative instructions and daily maximums never become doses.
Conflicting schedules or multiple medicines remain unassociated. Missing
important details are explicitly listed rather than invented. Extracted
fields, OCR text, confidence and speech eligibility are not mutated.

Supported labels, forms, exact dose-count expressions and durations have
Traditional Chinese presentation mappings. Unsupported medical wording remains
English with the existing untranslated indication and source comparison.
History results derive the new layout from stored source data without a schema
migration or another OCR pass. The OCR model and speech policy are unchanged.

UI/UX Pro Max's verified heading-hierarchy guidance informed semantic sections,
short value cards, accessible labels and secondary-text disclosure. Single-value
cards avoid repeating their visible heading as a second label. Existing warm
palette, text scaling, Read Aloud priority and original-image comparison remain.

Executed validation:

- `test-medicine-sections`: 14/14 (single vs multiple medicines, dosage vs
  strength, confidence caps, numeric boundaries, zero/negative/maximum controls,
  ambiguous schedules, preserved exceptions and saved-result reconstruction).
- `test-field-localization`: 14/14, including new section labels/values and
  preservation of unsupported conditional English directions.
- `test-ui`: 92/92, using actual PP-OCRv5, including section values, switching
  language without OCR or extraction changes, medical refusals, responsive and
  accessibility checks, offline scanning and existing regression paths.
- `test-medicine-ui`: eight real-rendered language/viewport captures (320, 390,
  768, 1440px) plus zero violations in the checked Chinese WCAG A/AA rules.
  The input is the explicitly synthetic single-medicine label through real OCR;
  it is not a clinical prescription accuracy claim.
- Safari parser: 27 scripts, zero blockers. Precache: 54 required entries,
  zero missing. The inherited size-budget limitation remains.
- Signed iOS build succeeded. Six relevant bundle files match current source
  by SHA-256. The update was installed on the connected physical iPhone.

Artifacts/screenshots: `artifacts/medicine-sections/`; full regression log:
`artifacts/redesign/medicine-sections-ui.log`. Physical-phone scans of the user's
own label and clinical interpretation remain unverified. Nothing was pushed.
