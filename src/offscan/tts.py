"""
SightLine — Text-to-Speech module.
Fully offline TTS using espeak-ng (preferred) or pyttsx3 (fallback).

Features:
  - Voice selection (male/female)
  - Speed control (words per minute)
  - Category-aware narration (reads document summaries intelligently)
  - Audio output to speaker or file
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

try:
    import pyttsx3
    HAS_PYTTSX = True
except ImportError:
    HAS_PYTTSX = False


class TextToSpeech:
    """Offline TTS engine — espeak-ng preferred, pyttsx3 fallback."""

    def __init__(
        self,
        voice: str = "default",
        rate: int = 175,
        volume: float = 1.0,
        output_file: str | None = None,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.output_file = output_file
        self.engine = self._detect_engine()
        self._pyttsx_engine = None

    def _detect_engine(self) -> str:
        """Detect available TTS engine."""
        # Check for espeak-ng
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True, check=True, timeout=5,
            )
            return "espeak-ng"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Check for espeak (older)
        try:
            subprocess.run(
                ["espeak", "--version"],
                capture_output=True, check=True, timeout=5,
            )
            return "espeak"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Fallback to pyttsx3
        if HAS_PYTTSX:
            return "pyttsx3"

        return "none"

    def speak(self, text: str) -> str:
        """
        Speak text aloud (or write to file if output_file is set).
        Returns the engine used.
        """
        if not text.strip():
            return "none"

        if self.engine == "espeak-ng" or self.engine == "espeak":
            return self._speak_espeak(text)
        elif self.engine == "pyttsx3":
            return self._speak_pyttsx3(text)
        else:
            # Last resort: write to stdout
            print(f"[TTS unavailable] {text}")
            return "stdout"

    def _speak_espeak(self, text: str) -> str:
        """Use espeak/espeak-ng for speech synthesis."""
        cmd = [self.engine]
        # Speed: espeak uses words-per-minute, default ~175
        cmd.extend(["-s", str(self.rate)])
        # Volume: 0-200, default 200
        cmd.extend(["-a", str(int(self.volume * 200))])
        # Voice selection
        if self.voice == "female":
            cmd.extend(["-v", "en+f3"])
        elif self.voice == "male":
            cmd.extend(["-v", "en+m3"])
        # Punctuation for better readability
        cmd.extend(["--punct", "<"])

        if self.output_file:
            # Output to WAV file
            cmd.extend(["-w", self.output_file])
            cmd.append(text)
            subprocess.run(cmd, capture_output=True, timeout=30)
            return f"espeak->{self.output_file}"
        else:
            cmd.append(text)
            subprocess.run(cmd, capture_output=True, timeout=30)
            return self.engine

    def _speak_pyttsx3(self, text: str) -> str:
        """Use pyttsx3 for speech synthesis."""
        if self._pyttsx_engine is None:
            self._pyttsx_engine = pyttsx3.init()
            self._pyttsx_engine.setProperty("rate", self.rate)
            self._pyttsx_engine.setProperty("volume", self.volume)

        if self.output_file:
            self._pyttsx_engine.save_to_file(text, self.output_file)
            self._pyttsx_engine.runAndWait()
            return f"pyttsx3->{self.output_file}"
        else:
            self._pyttsx_engine.say(text)
            self._pyttsx_engine.runAndWait()
            return "pyttsx3"

    def speak_summary(self, category: str, summary: str, full_text: str,
                      read_full: bool = False) -> str:
        """
        Narrate a document intelligently:
        1. Announce document type
        2. Read summary with key fields
        3. Optionally read full text
        """
        parts = []
        parts.append(summary)

        if read_full:
            parts.append(f" Now reading the full document. {full_text}")

        return self.speak(" ".join(parts))

    def speak_prompt(self, message: str) -> str:
        """Speak a short UI prompt (e.g. camera guidance)."""
        return self.speak(message)


def narrate_result(result, read_full: bool = False, voice: str = "default",
                   rate: int = 175, output_file: str | None = None) -> str:
    """
    Convenience function: narrate a pipeline result.
    Args:
        result: ScanResult from the pipeline
        read_full: Whether to read the full OCR text after the summary
        voice, rate, output_file: TTS settings
    Returns:
        Engine used for speech
    """
    tts = TextToSpeech(voice=voice, rate=rate, output_file=output_file)
    return tts.speak_summary(
        category=result.classification.category,
        summary=result.classification.summary,
        full_text=result.ocr_result.text,
        read_full=read_full,
    )
