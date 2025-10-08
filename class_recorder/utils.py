"""Helper utilities for Class Recorder."""

import re
from pathlib import Path
from typing import Iterable, List


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


def extract_date_fragment(base_name: str) -> str:
    parts = base_name.split('-')[0:3]
    return '-'.join(parts) if len(parts) == 3 else base_name


def build_frontmatter(base_name: str, course: str, duration_seconds: float) -> str:
    date_str = extract_date_fragment(base_name)
    course = course or 'Unknown'
    duration_seconds = duration_seconds or 0

    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    tag_slug = course.lower().replace(' ', '-')

    return (
        "---\n"
        f"date: {date_str}\n"
        f"course: {course}\n"
        f"duration: {duration_str}\n"
        f"tags: [lecture, {tag_slug}]\n"
        "---"
    )


def prefilter_transcript(text: str) -> str:
    """Remove conversational filler and obvious repetition before LLM usage."""
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
        if any(pattern in lowered for pattern in FILLER_PATTERNS):
            continue

        if len(stripped) < 12 and not any(char.isdigit() for char in stripped):
            continue

        fingerprint = re.sub(r"\W+", "", lowered)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        cleaned.append(stripped)

    return "\n".join(cleaned) if cleaned else text


def chunk_text(text: str, max_chars: int, overlap_paragraphs: int = 0) -> List[str]:
    """Split text into paragraph-preserving chunks."""
    if max_chars <= 0:
        return [text]

    paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
    if not paragraphs:
        return [text.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def paragraph_length(paragraph: str) -> int:
        return len(paragraph) + 2  # account for separator

    for paragraph in paragraphs:
        para_len = paragraph_length(paragraph)
        if current and current_len + para_len > max_chars:
            chunks.append('\n\n'.join(current))
            if overlap_paragraphs > 0:
                current = current[-overlap_paragraphs:]
                current_len = sum(paragraph_length(p) for p in current)
            else:
                current = []
                current_len = 0
        current.append(paragraph)
        current_len += para_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def clean_transcript_text(text: str, max_repeat_lines: int = 3) -> str:
    """Condense obvious noise while preserving lecture intent."""
    if not text:
        return text

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        # Preserve intentional blank lines sparingly
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            i += 1
            continue

        # Count consecutive duplicate lines
        count = 1
        while i + count < len(lines) and lines[i + count].strip() == line:
            count += 1

        if count > max_repeat_lines and len(line.split()) <= 10:
            # Keep a few repetitions, then summarize the rest
            cleaned_lines.extend([line] * max_repeat_lines)
            remaining = count - max_repeat_lines
            snippet = (line[:60] + '…') if len(line) > 60 else line
            cleaned_lines.append(f"[repeated {remaining} more times: '{snippet}']")
        else:
            cleaned_lines.extend([line] * count)

        i += count

    cleaned_text = "\n".join(cleaned_lines)

    # Collapse word-level stutters (e.g., "hi hi hi hi")
    repeated_word_pattern = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)

    def _shrink_repetition(match):
        word = match.group(1)
        return f"{word} {word} {word}"

    cleaned_text = repeated_word_pattern.sub(_shrink_repetition, cleaned_text)

    # Limit excess blank lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()
