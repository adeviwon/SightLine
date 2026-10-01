"""
SightLine — Document classifier.
Classifies scanned documents into categories: banking, medical, legal, general.

Two-tier approach:
  1. ONNX model: A lightweight text classifier (distilled BERT or similar)
     loaded via ONNX Runtime for semantic classification.
  2. TF-IDF fallback: If ONNX model unavailable, use scikit-learn TF-IDF +
     LogisticRegression trained on synthetic document patterns.

The classifier also extracts category-specific fields:
  - Banking: account numbers, sort codes, balances, transaction amounts
  - Medical: drug names, dosages, frequencies, warning labels
  - Legal: case numbers, dates, parties, clause references
  - General: dates, phone numbers, email addresses
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ── Document categories ──
CATEGORIES = ["banking", "medical", "legal", "general"]
CATEGORY_LABELS = {
    "banking": "Banking / Financial Document",
    "medical": "Medical Prescription / Health Record",
    "legal": "Legal Document / Contract",
    "general": "General Document",
}


# ── Synthetic training data for the TF-IDF classifier ──
SYNTHETIC_SAMPLES = [
    # Banking
    ("Account Number: 12345678 Sort Code: 12-34-56 Statement Period: January 2024", "banking"),
    ("Balance: £2,450.00 Deposit: £1,200.00 Withdrawal: £350.00 Bank: HSBC", "banking"),
    ("Dear Customer, Your account ending 4521 has been charged £89.99. Available balance: £1,234.56", "banking"),
    ("IBAN: GB29 NWBK 6016 1331 9268 19 BIC: NWBKGB2L Transaction: £500.00", "banking"),
    ("Mortgage Statement Principal: £250,000 Interest Rate: 3.5% Monthly Payment: £1,200", "banking"),
    ("Credit Card Statement Card ending 4521 Statement balance £1,890.50 Minimum payment £45.00", "banking"),
    ("Account Holder: John Smith Account No: 98765432 Branch: London West End Sort Code 40-12-19", "banking"),
    ("Transaction History Date Description Amount 01/01/2024 Salary Credit £3,200.00", "banking"),

    # Medical
    ("Prescription: Amoxicillin 500mg Take one capsule three times daily for 7 days", "medical"),
    ("Patient: Jane Doe Date of Birth: 15/03/1985 Diagnosis: Hypertension Medication: Lisinopril 10mg", "medical"),
    ("Rx: Metformin 500mg twice daily with meals. Warning: May cause gastrointestinal discomfort", "medical"),
    ("Dosage: 2 tablets every 6 hours. Do not exceed 8 tablets in 24 hours. Consult doctor if symptoms persist", "medical"),
    ("Medication: Atorvastatin 20mg once daily at bedtime. Caution: Avoid grapefruit juice", "medical"),
    ("Patient ID: 1234567 Blood Pressure: 140/90 mmHg Prescription: Amlodipine 5mg once daily", "medical"),
    ("Warning: This medication may cause drowsiness. Do not operate machinery. Keep out of reach of children", "medical"),
    ("Rx: Ibuprofen 400mg every 6-8 hours as needed for pain. Maximum: 1200mg per day", "medical"),

    # Legal
    ("Contract Agreement between Party A and Party B dated 15th January 2024. Clause 3.2: Terms of Service", "legal"),
    ("In the Matter of Case No. 2024-CV-00456 District Court. Plaintiff vs Defendant. Judgment: Dismissed", "legal"),
    ("Non-Disclosure Agreement This contract is binding under the laws of England and Wales", "legal"),
    ("WHEREAS the parties agree to the following terms and conditions: 1. Definitions 2. Obligations 3. Termination", "legal"),
    ("Lease Agreement Property: 123 Baker Street London NW1. Tenant shall pay monthly rent of £2,500", "legal"),
    ("Power of Attorney I hereby appoint the following person as my attorney-in-fact", "legal"),
    ("WILL AND TESTAMENT I, John Smith, being of sound mind, hereby declare this to be my last will", "legal"),
    ("Terms and Conditions By using this service you agree to be bound by the following terms", "legal"),

    # General
    ("Hello, your appointment is confirmed for Tuesday 15th at 2pm. Please arrive 10 minutes early.", "general"),
    ("Meeting notes: Discussed Q1 targets, action items assigned. Next meeting scheduled for Friday.", "general"),
    ("Recipe: Chocolate cake. Ingredients: 200g flour, 150g sugar, 3 eggs. Bake at 180C for 30 minutes.", "general"),
    ("Dear John, Thank you for your email. I will get back to you by end of week. Best regards, Sarah", "general"),
    ("Notice: The office will be closed on Monday for the bank holiday. Normal hours resume Tuesday.", "general"),
    ("Invoice #12345 Date: 01/03/2024 Item: Web design service Total: £500.00 Due: 15/03/2024", "general"),

    # Banking — additional samples
    ("Statement of Account Current Account £1,200.00 deposited on 01/01/2024", "banking"),
    ("Your bank balance is £3,450.75. Transaction fee £2.50. Overdraft limit £500.00", "banking"),
    ("Direct Debit £89.99 to HSBC from account 98765432 sort code 12-34-56", "banking"),
    ("ATM withdrawal £200.00 from account ending 4521. New balance £1,234.56", "banking"),
    ("International transfer £500.00 GBP to IBAN GB29 NWBK 6016 1331 9268 19", "banking"),

    # Medical — additional samples
    ("Prescription for Lisinopril 10mg once daily. Repeat prescription for 28 days", "medical"),
    ("Diagnosis: Type 2 Diabetes. Medication: Metformin 1000mg twice daily with meals", "medical"),
    ("Warfarin 5mg tablets. Take as directed. INR monitoring required. Caution: bleeding risk", "medical"),
    ("Salbutamol inhaler 100mcg. Two puffs as needed for wheezing. Maximum 8 puffs per day", "medical"),
    ("Patient discharged. Medications: Atorvastatin 20mg at bedtime. Follow-up in 2 weeks", "medical"),

    # Legal — additional samples
    ("Employment contract between employer and employee. Terms: salary, notice period, confidentiality", "legal"),
    ("This agreement is hereby terminated in accordance with Clause 7.3 of the original contract", "legal"),
    ("Non-Disclosure Agreement. The parties shall not disclose any confidential information", "legal"),
    ("Tenancy agreement. Landlord and Tenant. Rent payable monthly. Deposit held under DPS", "legal"),
    ("Last Will and Testament. I bequeath all my estate to my heirs. Executor appointed herein", "legal"),

    # General — additional samples
    ("Reminder: Your subscription renews on 15/03/2024. Contact support@email.com for changes", "general"),
    ("Your order #12345 has been shipped. Expected delivery: 3-5 working days. Tracking included", "general"),
    ("Thank you for attending our event. We hope to see you again next year. Please complete the survey", "general"),
    ("Newsletter: This week highlights include new products, special offers, and upcoming events", "general"),
    ("Out of office reply: I will be away until Monday. For urgent matters contact my colleague", "general"),
]


@dataclass
class ExtractedField:
    """A single extracted field with label, value, and confidence."""
    label: str
    value: str
    confidence: float
    source: str  # "regex" or "ml"


@dataclass
class ClassificationResult:
    """Document classification + field extraction result."""
    category: str
    category_label: str
    confidence: float
    engine: str  # "onnx" or "tfidf"
    extracted_fields: list[ExtractedField] = field(default_factory=list)
    summary: str = ""


class DocumentClassifier:
    """
    Document classifier with ONNX model + TF-IDF fallback.
    Trains the TF-IDF classifier on startup using synthetic patterns.
    """

    def __init__(self, onnx_model_path: str | None = None):
        self.onnx_session = None
        self.tfidf_pipeline = self._build_tfidf_classifier()
        self.engine = "tfidf"

        if HAS_ONNX and onnx_model_path:
            try:
                self.onnx_session = ort.InferenceSession(
                    onnx_model_path,
                    providers=["CPUExecutionProvider"],
                )
                self.engine = "onnx"
            except Exception:
                pass  # Fall back to TF-IDF

    def _build_tfidf_classifier(self) -> Pipeline:
        """Build and train a TF-IDF + LogisticRegression classifier on synthetic data."""
        texts = [s[0] for s in SYNTHETIC_SAMPLES]
        labels = [s[1] for s in SYNTHETIC_SAMPLES]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
            )),
        ])
        pipeline.fit(texts, labels)
        return pipeline

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify document text and extract category-specific fields.
        """
        if not text.strip():
            return ClassificationResult(
                category="general",
                category_label=CATEGORY_LABELS["general"],
                confidence=0.0,
                engine=self.engine,
                summary="No text detected.",
            )

        # Classification
        if self.onnx_session is not None:
            category, confidence = self._classify_onnx(text)
        else:
            category, confidence = self._classify_tfidf(text)

        # Field extraction
        fields = self._extract_fields(text, category)

        # Build summary
        summary = self._build_summary(category, fields, text)

        return ClassificationResult(
            category=category,
            category_label=CATEGORY_LABELS.get(category, "Unknown"),
            confidence=confidence,
            engine=self.engine,
            extracted_fields=fields,
            summary=summary,
        )

    def _classify_onnx(self, text: str) -> tuple[str, float]:
        """Classify using ONNX model (tokenized input)."""
        try:
            # Simplified: use character-level features for ONNX input
            # Real implementation would use a proper tokenizer
            features = self._text_to_features(text)
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: features})
            pred_idx = int(outputs[0].argmax())
            confidence = float(outputs[0][0][pred_idx]) if outputs[0].ndim == 2 else float(outputs[0][pred_idx])
            return CATEGORIES[pred_idx], confidence
        except Exception:
            return self._classify_tfidf(text)

    def _classify_tfidf(self, text: str) -> tuple[str, float]:
        """Classify using TF-IDF + LogisticRegression."""
        probas = self.tfidf_pipeline.predict_proba([text])[0]
        best_idx = probas.argmax()
        category = self.tfidf_pipeline.classes_[best_idx]
        confidence = float(probas[best_idx])
        return category, confidence

    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to feature vector for ONNX model."""
        # Simple bag-of-characters as fallback feature encoding
        features = np.zeros((1, 128), dtype=np.float32)
        for char in text.lower():
            if ord(char) < 128:
                features[0][ord(char)] += 1
        # Normalize
        total = features.sum()
        if total > 0:
            features = features / total
        return features

    def _extract_fields(self, text: str, category: str) -> list[ExtractedField]:
        """Extract category-specific fields using regex patterns."""
        fields = []

        if category == "banking":
            fields.extend(self._extract_banking_fields(text))
        elif category == "medical":
            fields.extend(self._extract_medical_fields(text))
        elif category == "legal":
            fields.extend(self._extract_legal_fields(text))
        else:
            fields.extend(self._extract_general_fields(text))

        return fields

    def _extract_banking_fields(self, text: str) -> list[ExtractedField]:
        fields = []
        patterns = {
            "Account Number": r"account\s*(?:no|number)?[:#]?\s*(\d{6,12})",
            "Sort Code": r"sort\s*code[:\s]*(\d{2}[-\s]?\d{2}[-\s]?\d{2})",
            "IBAN": r"(GB\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2})",
            "Balance": r"(?:balance|available)[:\s]*£?([\d,]+\.\d{2})",
            "Amount": r"£([\d,]+\.\d{2})",
            "Card End": r"(?:ending|card)\s*(?:in\s*)?(\d{4})",
        }
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                fields.append(ExtractedField(label=label, value=str(m), confidence=0.92, source="regex"))
        return fields

    def _extract_medical_fields(self, text: str) -> list[ExtractedField]:
        fields = []
        patterns = {
            "Medication": r"(?:Rx\s*:\s*|medication\s*:\s*|prescription\s+for\s+)([A-Z][a-z]+(?:\s\d+mg)?)",
            "Dosage": r"(\d+\s*(?:mg|ml|tablets?|capsules?)\s*(?:every\s*\d+\s*hours?|twice|three times|once daily|at bedtime)?)",
            "Frequency": r"(every\s*\d+\s*(?:hours?|hrs)|\d+\s*times?\s*daily|once\s*daily|twice\s*daily|three\s*times\s*daily)",
            "Warning": r"((?:WARNING|Caution|Do not\s+|Avoid\s+|May cause\s+|Consult\s+|Keep out\s+)[^.]+)",
            "Patient": r"(?:patient|name)[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)",
        }
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                fields.append(ExtractedField(label=label, value=str(m).strip(), confidence=0.88, source="regex"))
        return fields

    def _extract_legal_fields(self, text: str) -> list[ExtractedField]:
        fields = []
        patterns = {
            "Case Number": r"(?:case|matter)\s*(?:no|number)?[:#]?\s*([\w-]+)",
            "Date": r"(\d{1,2}(?:st|nd|rd|th)?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4})",
            "Parties": r"(?:between|vs|plaintiff|defendant)[:\s]*([A-Z][\w\s]+)",
            "Clause": r"(?:clause|section)\s*(\d+\.\d+)",
        }
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                fields.append(ExtractedField(label=label, value=str(m).strip(), confidence=0.85, source="regex"))
        return fields

    def _extract_general_fields(self, text: str) -> list[ExtractedField]:
        fields = []
        patterns = {
            "Date": r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "Phone": r"(\+?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4})",
            "Email": r"([\w.+-]+@[\w-]+\.[\w.-]+)",
            "Time": r"(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)",
        }
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for m in matches:
                fields.append(ExtractedField(label=label, value=str(m).strip(), confidence=0.82, source="regex"))
        return fields

    def _build_summary(self, category: str, fields: list[ExtractedField], text: str) -> str:
        """Build a natural-language summary for TTS narration."""
        if not fields:
            return text[:300].strip()

        # Group fields by label
        medications = [f.value for f in fields if f.label == "Medication"]
        dosages = [f.value for f in fields if f.label == "Dosage"]
        frequencies = [f.value for f in fields if f.label == "Frequency"]
        warnings = [f.value for f in fields if f.label == "Warning"]
        patients = [f.value for f in fields if f.label == "Patient"]

        # Account numbers, balances, amounts for banking
        account_numbers = [f.value for f in fields if f.label == "Account Number"]
        balances = [f.value for f in fields if f.label == "Balance"]
        amounts = [f.value for f in fields if f.label == "Amount"]
        card_ends = [f.value for f in fields if f.label == "Card End"]

        # Case numbers, clauses, dates, parties for legal
        case_numbers = [f.value for f in fields if f.label == "Case Number"]
        clauses = [f.value for f in fields if f.label == "Clause"]
        dates = [f.value for f in fields if f.label == "Date"]
        parties = [f.value for f in fields if f.label == "Parties"]

        # General
        emails = [f.value for f in fields if f.label == "Email"]
        phones = [f.value for f in fields if f.label == "Phone"]
        times = [f.value for f in fields if f.label == "Time"]

        parts = []

        if category == "medical":
            parts.append("This is a medical prescription. ")
            if patients:
                parts.append(f"Patient: {patients[0]}. ")
            if medications:
                parts.append(f"Medications prescribed: {', '.join(medications)}. ")
            if dosages:
                parts.append(f"Dosages: {', '.join(dosages)}. ")
            if frequencies:
                parts.append(f"Frequency: {', '.join(frequencies)}. ")
            if warnings:
                parts.append(f"Warnings: {' '.join(warnings)}")
        elif category == "banking":
            parts.append("This is a banking document. ")
            if account_numbers:
                parts.append(f"Account number: {account_numbers[0]}. ")
            if card_ends:
                parts.append(f"Card ending in {card_ends[0]}. ")
            if balances:
                parts.append(f"Balance: {balances[0]}. ")
            if amounts:
                parts.append(f"Amounts: {', '.join(amounts[:3])}. ")
        elif category == "legal":
            parts.append("This is a legal document. ")
            if case_numbers:
                parts.append(f"Case reference: {case_numbers[0]}. ")
            if dates:
                parts.append(f"Date: {dates[0]}. ")
            if clauses:
                parts.append(f"Clauses referenced: {', '.join(clauses)}. ")
        else:
            parts.append("This is a document. ")
            if dates:
                parts.append(f"Date: {dates[0]}. ")
            if emails:
                parts.append(f"Email: {emails[0]}. ")
            if phones:
                parts.append(f"Phone: {phones[0]}. ")
            if times:
                parts.append(f"Time: {times[0]}. ")

        return "".join(parts)
