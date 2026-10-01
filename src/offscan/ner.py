"""
SightLine — Named Entity Recognition (NER) module.
Uses an ONNX model to extract named entities (medications, amounts, dates,
organizations) from OCR text. Falls back to regex if ONNX model unavailable.

This is the ML layer above the classifier's regex extraction — it catches
entities that patterns miss (drug names not in the pattern list, unusual
date formats, amounts with non-standard currency symbols).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@dataclass
class Entity:
    """A named entity extracted from text."""
    text: str
    label: str  # DRUG, DOSAGE, DATE, MONEY, ORG, PERSON, AMOUNT
    start: int
    end: int
    confidence: float
    source: str  # "onnx" or "regex"


@dataclass
class NERResult:
    """All entities found in a text."""
    entities: list[Entity] = field(default_factory=list)
    engine: str = "regex"


class NERExtractor:
    """
    NER extraction with ONNX model + regex fallback.
    The ONNX model is a token classification model (e.g. distilled BERT NER)
    that labels each token with an entity type.
    """

    # Entity labels we recognize
    LABELS = ["O", "B-DRUG", "I-DRUG", "B-DOSAGE", "I-DOSAGE", "B-DATE", "I-DATE",
              "B-MONEY", "I-MONEY", "B-AMOUNT", "I-AMOUNT", "B-ORG", "I-ORG",
              "B-PERSON", "I-PERSON"]

    def __init__(self, onnx_model_path: str | None = None):
        self.onnx_session = None
        self.engine = "regex"

        if HAS_ONNX and onnx_model_path:
            try:
                self.onnx_session = ort.InferenceSession(
                    onnx_model_path,
                    providers=["CPUExecutionProvider"],
                )
                self.engine = "onnx"
            except Exception:
                pass

    def extract(self, text: str) -> NERResult:
        """Extract named entities from text."""
        if self.onnx_session is not None:
            return self._extract_onnx(text)
        return self._extract_regex(text)

    def _extract_onnx(self, text: str) -> NERResult:
        """ONNX-based NER extraction."""
        try:
            tokens = text.split()
            # Convert tokens to feature vectors (simplified char-level)
            features = self._tokens_to_features(tokens)

            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: features})

            # Decode token-level predictions
            entities = self._decode_predictions(tokens, outputs[0])
            return NERResult(entities=entities, engine="onnx")
        except Exception:
            return self._extract_regex(text)

    def _extract_regex(self, text: str) -> NERResult:
        """Regex-based entity extraction — the reliable fallback."""
        import re
        entities = []

        patterns = [
            # Drugs — common medication patterns
            ("DRUG", r"\b(Amoxicillin|Metformin|Lisinopril|Atorvastatin|Amlodipine|Ibuprofen|Paracetamol|Aspirin|Omeprazole|Simvastatin|Warfarin|Insulin|Prednisolone|Sertraline|Citalopram|Ramipril|Bisoprolol|Salbutamol|Codeine|Tramadol|Diclofenac|Naproxen)\b"),
            # Dosages — number + unit
            ("DOSAGE", r"\b(\d+\s*(?:mg|ml|mcg|micrograms?|tablets?|capsules?|drops?|puffs?))\b"),
            # Frequency
            ("DOSAGE", r"\b(every\s*\d+\s*(?:hours?|hrs)|once\s*daily|twice\s*daily|three\s*times\s*daily|\d+\s*times?\s*(?:a\s*)?day)\b"),
            # Dates — multiple formats
            ("DATE", r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
            ("DATE", r"\b(\d{1,2}(?:st|nd|rd|th)?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4})\b"),
            # Money — GBP, USD, EUR
            ("MONEY", r"(?:£|\$|€)([\d,]+\.\d{2})"),
            ("MONEY", r"\b([\d,]+\.\d{2})\s*(?:GBP|USD|EUR|pounds?)\b"),
            # Amounts
            ("AMOUNT", r"\b(\d+%)\b"),
            # Organizations — bank names, NHS, etc.
            ("ORG", r"\b(HSBC|Barclays|Lloyds|NatWest|Santander|NHS|Halifax|TSB|Metro Bank|Revolut|Monzo)\b"),
            # People — Patient/Account Holder names
            ("PERSON", r"(?:Patient|Name|Account Holder|Holder)[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)"),
        ]

        for label, pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                # For patterns with capture groups, use group(1); else group(0)
                value = m.group(1) if m.lastindex else m.group(0)
                entities.append(Entity(
                    text=value.strip(),
                    label=label,
                    start=m.start(1 if m.lastindex else 0),
                    end=m.end(1 if m.lastindex else 0),
                    confidence=0.90,
                    source="regex",
                ))

        return NERResult(entities=entities, engine="regex")

    def _tokens_to_features(self, tokens: list[str]) -> np.ndarray:
        """Convert tokens to feature vectors for ONNX model."""
        max_len = 128
        features = np.zeros((1, max_len, 64), dtype=np.float32)

        for i, token in enumerate(tokens[:max_len]):
            # Simple character-level encoding
            for j, char in enumerate(token[:64]):
                if ord(char) < 64:
                    features[0][i][j] = ord(char) / 64.0

        return features

    def _decode_predictions(self, tokens: list[str], predictions: np.ndarray) -> list[Entity]:
        """Decode ONNX token-level predictions into entity spans."""
        entities = []
        current_entity = None
        current_tokens = []

        for i, token in enumerate(tokens[:len(predictions)]):
            pred = predictions[i] if i < len(predictions) else 0
            label = self.LABELS[pred] if pred < len(self.LABELS) else "O"
            conf = 0.85  # Default confidence for ONNX predictions

            if label.startswith("B-"):
                if current_entity:
                    entities.append(Entity(
                        text=" ".join(current_tokens),
                        label=current_entity,
                        start=i - len(current_tokens),
                        end=i,
                        confidence=conf,
                        source="onnx",
                    ))
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith("I-") and current_entity:
                current_tokens.append(token)
            else:
                if current_entity:
                    entities.append(Entity(
                        text=" ".join(current_tokens),
                        label=current_entity,
                        start=i - len(current_tokens),
                        end=i,
                        confidence=conf,
                        source="onnx",
                    ))
                    current_entity = None
                    current_tokens = []

        if current_entity:
            entities.append(Entity(
                text=" ".join(current_tokens),
                label=current_entity,
                start=len(tokens) - len(current_tokens),
                end=len(tokens),
                confidence=conf,
                source="onnx",
            ))

        return entities
