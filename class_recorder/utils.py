"""Helper utilities for Class Recorder."""

import re
from pathlib import Path
from typing import Iterable


def ensure_directory(path):
    """Ensure directory exists and return Path instance."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


FILLER_PATTERNS: Iterable[str] = (
    "i'm here",
    "can you hear",
    "hello",
    "good luck",
    "test test",
    "mic check",
    "background noise",
)


def prefilter_transcript(text: str) -> str:
    """Remove conversational filler and high-frequency duplicates before LLM usage."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = []
    seen = set()

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        # Skip obvious filler noise or repeated greetings.
        if any(pattern in lowered for pattern in FILLER_PATTERNS):
            continue

        # Drop very short lines unless they contain numbers (for assignments, dates, etc.).
        if len(stripped) < 25 and not any(char.isdigit() for char in stripped):
            continue

        # Avoid sending the same content repeatedly.
        fingerprint = re.sub(r"\W+", "", lowered)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        cleaned.append(stripped)

    return "\n".join(cleaned) if cleaned else text
