"""
SightLine ML — Document corpus for the sentence-transformer classifier.

Design goals (in priority order):
  1. HONEST EVALUATION. Every template is unique. Templates are split
     train/val BEFORE any augmentation, so no phrase can appear on both
     sides. The old corpus silently duplicated ~15 templates, which is why
     the previous run reported n_val=15 and a 6.7-point swing per sample.
  2. REALISTIC DIFFICULTY. Real OCR output is noisy: dropped characters,
     digit/letter confusions (0/O, 1/l, 5/S), broken words, missing
     punctuation, Tesseract's favourite errors. Templates below include
     those artifacts deliberately.
  3. HARD CATEGORIES. "banking" and "general" overlap in real life (an
     invoice mentions money; a bank letter mentions dates). Templates
     include genuine distractors so the model cannot cheat on a single
     keyword.

`make_corpus()` returns (texts, labels). `augment()` applies OCR-style
corruption ONLY to the training split.
"""

import random
import re

CATEGORIES = ["banking", "medical", "legal", "general"]

# ── Banking ─────────────────────────────────────────────────────────────
BANKING = [
    "HSBC UK Bank plc Statement. Account Holder: J SMITH. Account Number: 40218877. Sort Code: 40-11-04.",
    "Statement of Account for the period 01/03 to 31/03. Current account. Opening balance GBP 1,204.33.",
    "Closing balance 2,544.06. Available balance 2,100.00. Arranged overdraft limit 500.00.",
    "Direct Debit 89.99 BRITISH GAS DD ref 4471 taken from account 98765432 on 14/02.",
    "CASH WITHDRAWAL 200.00 at ATM 8812 on 02/02. Card ending 4521. Balance 1,234.56.",
    "STANDING ORDER rent 1,450.00 monthly to LANDLORD ACCOUNT 55443322. Next due 01/03.",
    "Credit Card statement. Card ending 8812. Minimum payment due 45.00 by 28th. Statement balance 1,204.88.",
    "Mortgage statement. Principal outstanding 248,900.12. Interest rate fixed at 3.49% until 2031.",
    "IBAN GB29 NWBK 6016 1331 9268 19. BIC NWBKGB2L. SEPA transfer of 500.00 GBP executed.",
    "Sort code 20-44-19, account number 77889901. Branch: Manchester Piccadilly. Opening balance 88.22.",
    "Transaction history: SALARY CREDIT 2,900.00 on 28th. RENT DD 1,100.00 on 01st. Balance 4,102.11.",
    "Overdraft interest charged at 0.5% above base rate for this quarter. Total interest 4.12.",
    "Savings account statement. Interest earned 12.34. Closing balance 8,901.55. Goal balance 10,000.",
    "Faster Payment sent to Ms A Recipient. Reference INVOICE-9931. Amount 240.00 GBP.",
    "Cheque number 000231 deposited. Funds available after 3 working days. Amount 175.00.",
    "Foreign transaction fee 2.75 applied to purchase of 55.40 EUR. Total charge 1.52.",
    "Loan repayment schedule: 12 monthly instalments of 210.50 commencing 05/2024.",
    "Business current account. Annual service charge 96.00. Fee waived this quarter.",
    "Alert: your card ending 7719 was used at a merchant in another country on 19/01.",
    "Joint account statement for Mr and Mrs Smith. Sort code 40-52-08. Both holders must sign.",
    "Fixed rate bond matures June 2027. Early withdrawal penalty applies after 90 days interest.",
    "Your payment of 1,450.00 has been received. Thank you. Reference SO-99120.",
    "Deposit protection: eligible deposits up to 85,000 per person are protected.",
    "Cash withdrawal fee 1.50 applies at non-network ATMs. Fee waived for Premier customers.",
    "Standing Order DD to SAVE 200 monthly. Collected on the 6th of each month.",
    "Interest rate 4.10% variable on current balance above 5,000. Terms apply.",
    "Card payment to TESCO STORES 44.19 approved. Card ending 4521. Available funds 2,100.00.",
    "BACS payment received from ABC LTD, reference INV-2211, amount 3,400.00.",
    "Your monthly statement is now available in the mobile banking app. Log in securely.",
    "Unpresented items total 512.00. Cleared funds 3,088.44. Book balance 3,600.44.",
    "Instalment plan: 24 monthly payments of 88.20 on purchase of a laptop. First payment 01/04.",
    "Cheque paid in for 240.00. Image of cheque available in your documents.",
    "Set up a new direct debit to utility provider. First collection 15/03, amount 76.40.",
    "Account summary: current account 1,204.33, savings account 8,901.55, total funds 10,105.88.",
    "International payment sent. Recipient bank charges borne by sender. Amount USD 1,200.00.",
    "Your overdraft limit increases to 1,000.00 from 01/05. Eligible accounts only.",
    "Previous direct debit of 89.99 to BRITISH GAS FAILED. Reason: insufficient funds.",
    "Statement fee of 5.00 applies to non-economy accounts. Fee waived if balance above 1,000.",
    "Payroll deposit 2,340.55 from HOSPITAL PAYROLL. Net pay, month ending 30/04.",
    "Card ending 4521 has been replaced. New card active, old card destroyed on receipt.",
    "Transfer between your accounts: 500.00 moved to savings on 02/04. Instant, no fee.",
    "Notice: we are changing the interest rate on your savings account to 3.85% from June.",
    "Bank reference for your landlord: sorting number 40-11-04, account 40218877, reference 7712004.",
    "Two failed direct debit attempts on 01/03. A further fee of 6.00 may apply.",
    "Card ending 8812 statement total 1,204.88. Minimum payment 45.00, final payment 1,159.88.",
]

# ── Medical ──────────────────────────────────────────────────────────────
MEDICAL = [
    "PRESCRIPTION. Patient: Jane Doe, DOB 04/02/1979. Rx: Amoxicillin 500mg, one capsule three times daily.",
    "Take one tablet twice daily after meals. Complete the full course of antibiotics even if better.",
    "MetFORMIN hydrochloride 1000mg. Take with food to reduce stomach upset. Twice daily.",
    "Ibuprofen 400mg as required for pain relief. Do not exceed 1200mg in any 24 hour period.",
    "WARNING: this medicine may cause drowsiness. Do not operate machinery or drive after taking.",
    "Lisinopril 10mg once daily in the morning for blood pressure control. Review in 4 weeks.",
    "Inhaler: Salbutamol 100 micrograms. Two puffs when wheezing, maximum four times daily.",
    "Patient: John Smith, date of birth 12/04/1958. Known allergy: PENICILLIN. Allergy status verified.",
    "Apply a thin layer of cream to the affected area twice daily for seven days.",
    "Blood test results: haemoglobin 13.2 g/dL, white cell count 6.1, C-reactive protein normal.",
    "Repeat prescription request approved. Atorvastatin 20mg, quantity 28 tablets.",
    "Paracetamol 500mg every six hours if required. Maximum eight tablets (4g) per day.",
    "Insulin glargine 12 units subcutaneously at bedtime. Rotate injection sites regularly.",
    "Consult your doctor if symptoms persist beyond five days of treatment.",
    "Do not stop taking this medicine without speaking to your pharmacist first.",
    "Warfarin 3mg daily. Weekly INR blood test required. Report any unusual bruising.",
    "Omeprazole 20mg gastro-resistant capsule once daily, thirty minutes before breakfast.",
    "Keep out of reach of children. Store below 25 degrees Celsius in a dry place.",
    "The pharmacist dispensed 28 capsules with a repeat date of next month.",
    "Diagnosis: type 2 diabetes mellitus. HbA1c 7.8%. Medical review in three months.",
    "GP surgery follow-up appointment scheduled for the 15th at 2:30pm with Dr Ahmed.",
    "Adverse reaction reported: mild nausea after the first dose. Continue and monitor.",
    "Eye drops: one drop into each eye every night at bedtime. Avoid contact lenses.",
    "Complete the full seven day course even if you feel better than expected.",
    "Prescription charge paid. Medical exemption certificate held, valid to March 2027.",
    "Take one tablet twice daily after meals. Complete the full course.",
    "MetFORMIN hydrochloride 1000mg. Take with food to reduce stomach upset.",
    "Ibuprofen 400mg as required. Do not exceed 1200mg in 24 hours.",
    "WARNING: may cause drowsiness. Do not operate machinery after taking.",
    "Lisinopril 10mg once daily in the morning for hypertension.",
    "Salbutamol inhaler 100mcg. Two puffs as needed, maximum eight puffs daily.",
    "Patient: A Okafor, DOB 22/07/1965. Prescribed: Levothyroxine 75mcg once daily.",
    "Sertraline 50mg once daily, increased to 100mg after two weeks. Review in six weeks.",
    "Clenil inhaler 200mcg twice daily. Rinse mouth after each use to prevent thrush.",
    "Co-amoxiclav 625mg three times daily for five days. Take with food.",
    "Prednisolone 5mg tablets, tapering dose. Take in the morning with breakfast.",
    "Ferrous sulfate 200mg once daily. Take with orange juice for better absorption.",
    "Furosemide 40mg in the morning. Expect increased urination. Report swelling.",
    "Glyceryl trinitrate 400 micrograms under the tongue. Sit down before taking.",
    "Allergy: severe nut allergy. Adrenaline auto-injector 300 micrograms prescribed.",
    "Clotrimazole cream 1% apply twice daily for two weeks to the affected skin.",
    "Test result: Vitamin D level 14 ng/mL, deficient. Prescribed 50,000 IU weekly.",
    "Referral letter to cardiology. ECG performed. Awaiting consultant review within 6 weeks.",
    "Chemotherapy cycle 2 of 6. Ondansetron 8mg to prevent sickness. Bloods on Monday.",
    "Mental health: CBT sessions weekly with therapist. Review plan after eight sessions.",
    "Physiotherapy referral: lower back pain. Six week course, exercises issued.",
    "Vaccination record: influenza vaccine administered 12/10/2025. Next due 12/10/2026.",
    "Discharge summary: appendicectomy, uncomplicated. Wound healing well. Pain relief as required.",
    "Repeat prescription: Amlodipine 5mg once daily. Blood pressure check in 3 months.",
    "Paediatric: child aged 4, weight 16kg. Amoxicillin 25mg/kg/day divided three times daily.",
]

# ── Legal ───────────────────────────────────────────────────────────────
LEGAL = [
    "LEASE AGREEMENT between the Landlord and the Tenant dated the fifteenth day of January 2024.",
    "This Agreement is binding upon both parties under the laws of England and Wales.",
    "Clause 3.2: the Tenant shall pay rent of 2,500 monthly in advance on the first day.",
    "Clause 5.1: either party may terminate this agreement with two months written notice.",
    "IN THE MATTER OF Case No. 2024-CV-00456 before the District Court of Hong Kong.",
    "The Plaintiff claims damages. The Defendant denies liability in its entirety.",
    "POWER OF ATTORNEY. I appoint the following person to act as my attorney on my behalf.",
    "LAST WILL AND TESTAMENT of John Smith, made this fourth day of June 2023.",
    "Non-Disclosure Agreement. The parties shall not disclose confidential information to third parties.",
    "WHEREAS the parties agree to the terms and conditions set out in this Contract.",
    "EMPLOYMENT CONTRACT between the Company and the Employee. Probationary period six months.",
    "Tenancy deposit of 5,000 held in the Deposit Protection Scheme for the property at 12 Bridge Road.",
    "WITNESSETH: the Landlord lets and the Tenant takes the Property known as 123 Baker Street, London.",
    "JUDGMENT entered for the Claimant. Costs to be assessed by the court officer.",
    "SERVICE of judicial documents may be effected at the registered office of the company.",
    "The Licence is granted subject to the conditions set out in Schedule 2 of this deed.",
    "ARBITRATION CLAUSE: disputes resolved under LCIA Rules, seat London, sole arbitrator.",
    "NOTARISED copy certified true by the solicitor of the Supreme Court of Hong Kong.",
    "TERMS AND CONDITIONS OF BUSINESS. By engaging our services you accept these terms.",
    "SEVERABILITY: if any provision is invalid the remainder of this agreement continues in force.",
    "This DEED is executed on the date first written above by the parties hereto.",
    "Sub-letting is prohibited without the prior written consent of the Landlord.",
    "GOVERNING LAW: this Contract is governed by the law of the Hong Kong SAR.",
    "The Purchaser covenants with the Vendor to observe the restrictive covenants in the lease.",
    "BREACH of any term entitles the innocent party to terminate immediately without notice.",
    "Assignment: the Tenant shall not assign this tenancy without the Landlord's consent.",
    "Schedule 1: the property comprises a three bedroom flat with parking bay number 14.",
    "The parties agree that time shall be of the essence in relation to each obligation.",
    "Indemnity: the Tenant shall indemnify the Landlord against all losses arising from the premises.",
    "Executed as a deed by the parties in the presence of witnesses on 15 January 2024.",
    "Court order: the Defendant shall pay the Plaintiff costs of 4,200 within 14 days.",
    "Shareholders agreement. The Company shall not issue shares without unanimous written consent.",
    "Warranty deed: the Vendor gives full title guarantee, free from encumbrances.",
    "Legal notice: this letter serves as notice of claims under clause 8 of the agreement.",
    "Confidentiality undertaking signed by the employee on commencement of employment.",
    "Settlement agreement. The Claimant withdraws the claim in exchange for 12,000 damages.",
    "The Respondent shall file a defence within 28 days of service of this order.",
    "Right of first refusal over the property expires 60 days after notice is given.",
]

# ── General (non-financial, non-medical, non-legal) ──────────────────────
GENERAL = [
    "Hello, your appointment is confirmed for Tuesday the 15th at 2pm. Please arrive early.",
    "Meeting notes: discussed quarterly targets, actions assigned, next meeting on Friday.",
    "RECIPE: chocolate cake. 200g flour, 150g sugar, 3 eggs. Bake for 30 minutes at 180C.",
    "Dear John, thank you for your email. I will reply by the end of the week.",
    "NOTICE: the office will be closed on Monday for the public holiday.",
    "INVOICE 12345 dated 01/03/2024. Web design services, 12 hours. Total due 500.00.",
    "Your parcel has been dispatched and should arrive within three working days.",
    "WELCOME to the neighbourhood. The residents association meets on the first Tuesday monthly.",
    "REMINDER: library books are due back by the end of the month.",
    "School newsletter: sports day rescheduled to Thursday due to weather.",
    "Your subscription renews automatically on the 15th unless cancelled before then.",
    "Thank you for your order. A receipt has been attached for your records.",
    "Community centre quiz night, Saturday 7pm. Teams of four welcome.",
    "The caretaker has arranged boiler servicing for Wednesday morning.",
    "Please complete the short survey so we can improve our service.",
    "Train service update: the 08:15 to Victoria is cancelled today. Replacement bus service.",
    "We regret to inform you that the event is postponed to next spring.",
    "Happy birthday! The party starts at 3pm at the community hall.",
    "Your application has been received and is under review by the selection panel.",
    "Volunteers are needed for the charity collection this weekend.",
    "Parking restrictions apply from Monday to Friday between 8am and 6pm.",
    "Please note the museum is closed for refurbishment until the spring.",
    "Swimming pool timetable: ladies session 9am to 11am on weekdays.",
    "The choir is looking for new members. Rehearsals every Wednesday at 7pm.",
    "Library books about local history are available on the second floor.",
    "Your dry cleaning is ready for collection from reception until 7pm.",
    "The bus timetable has changed. Services 44 and 45 now run every ten minutes.",
    "Lost property: a blue jacket was handed in at the leisure centre on Tuesday.",
    "Thank you for attending the AGM. Minutes have been emailed to all members.",
    "The allotment society is holding a seed swap on the first Sunday of spring.",
    "Evening class: beginner guitar starts on the 12th. Places are limited.",
    "Your library card expires at the end of the year. Renew online or in branch.",
    "Reminder: the recycling collection moves to Thursday from next week.",
    "The walking group meets at the corner of the square at ten past nine.",
    "Cinema listings: family show at 11am, certificates to follow the screening.",
    "Your photos from the print order are ready. Pick up before the 20th.",
    "Community hall kitchen will be closed for redecorating throughout July.",
    "The running club completed a ten kilometre route along the canal this morning.",
    "Book group selection: this month's novel is available from the lending library.",
    "Please note that the bus stop on High Street has been temporarily relocated.",
    "Aquatic centre: lane swimming available 6:30am to 9am on weekdays.",
]

RAW = {
    "banking": BANKING,
    "medical": MEDICAL,
    "legal": LEGAL,
    "general": GENERAL,
}


# ── OCR corruption (training split only) ────────────────────────────────

_CONFUSIONS = [("0", "O"), ("O", "0"), ("1", "l"), ("l", "1"),
               ("5", "S"), ("S", "5"), ("8", "B"), ("rn", "m")]
_DIGIT_RUN = re.compile(r"\d{2,}")


def ocr_noise(text, rng, p_sub=0.35, p_case=0.2):
    """Simulate Tesseract output artifacts: digit/letter confusion, case damage."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isdigit() and rng.random() < p_sub:
            chars[i] = rng.choice("0O1l5S8")
    if rng.random() < p_sub:
        m = _DIGIT_RUN.search(text)
        if m:
            i = rng.randrange(len(text))
            chars[i] = rng.choice("abcdefgh") if chars[i].isalnum() else chars[i]
    out = "".join(chars)
    if rng.random() < p_case:
        i = rng.randrange(len(out))
        out = out[:i] + out[i].swapcase() + out[i + 1:]
    return out


def augment(texts, labels, seed=123, n_copies=2):
    """Expand the TRAIN split with OCR-realistic corrupted variants."""
    rng = random.Random(seed)
    out_t, out_l = list(texts), list(labels)
    for _ in range(n_copies):
        for t, l in zip(texts, labels):
            out_t.append(ocr_noise(t, rng))
            out_l.append(l)
    return out_t, out_l


def _dedupe(lists):
    seen, out = set(), []
    for t in lists:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out


def make_corpus(seed=123, val_frac=0.30):
    """
    Deduplicated corpus, stratified split.

    Returns (train_texts, train_labels, val_texts, val_labels).
    Val is never augmented. Train is augmented AFTER splitting, so a val
    phrase can never have a corrupted twin in train.
    """
    buckets = {c: _dedupe(RAW[c]) for c in CATEGORIES}
    rng = random.Random(seed)
    tr_t, tr_l, va_t, va_l = [], [], [], []
    for ci, cat in enumerate(CATEGORIES):
        items = buckets[cat][:]
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_frac)))
        tr_t += items[:cut]
        tr_l += [ci] * cut
        va_t += items[cut:]
        va_l += [ci] * (len(items) - cut)
    order = list(range(len(tr_t)))
    rng.shuffle(order)
    tr_t = [tr_t[i] for i in order]
    tr_l = [tr_l[i] for i in order]
    tr_t, tr_l = augment(tr_t, tr_l, seed=seed)
    return tr_t, tr_l, va_t, va_l


def corpus_stats():
    b = {c: len(_dedupe(RAW[c])) for c in CATEGORIES}
    return {"unique_templates": b, "total_unique": sum(b.values())}


if __name__ == "__main__":
    import json
    print(json.dumps(corpus_stats(), indent=2))
    tr_t, tr_l, va_t, va_l = make_corpus()
    print(f"train={len(tr_t)} (after augmentation)  val={len(va_t)} (clean)")
    print("sample augmented:", tr_t[-1])
