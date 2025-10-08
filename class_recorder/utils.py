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


def _chunk_by_units(units: List[str], max_chars: int, overlap_units: int, join_with: str = '\n\n') -> List[str]:
    """
    Generic chunking for any text units (paragraphs or sentences).
    
    Args:
        units: List of text units to chunk (paragraphs, sentences, etc.)
        max_chars: Maximum characters per chunk
        overlap_units: Number of units to overlap between chunks
        join_with: String to join units with ('\n\n' for paragraphs, ' ' for sentences)
    
    Returns:
        List of text chunks with overlap
    """
    if not units:
        return []
    
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit) + len(join_with)  # Account for separator
        
        # If adding this unit exceeds max_chars and we have content, finalize chunk
        if current and current_len + unit_len > max_chars:
            chunks.append(join_with.join(current))
            
            # Keep last N units for overlap
            if overlap_units > 0 and len(current) > overlap_units:
                current = current[-overlap_units:]
                current_len = sum(len(p) + len(join_with) for p in current)
            else:
                current = []
                current_len = 0
        
        current.append(unit)
        current_len += unit_len

    # Add remaining content
    if current:
        chunks.append(join_with.join(current))

    return chunks


def chunk_text(text: str, max_chars: int, overlap_paragraphs: int = 0) -> List[str]:
    """
    Split text into chunks with intelligent handling.
    
    Strategy:
    1. First try splitting by paragraphs (\n\n)
    2. If only 1 long paragraph exists, split by sentences instead
    3. Maintain overlap between chunks for context preservation
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap_paragraphs: Number of units (paragraphs or sentences) to overlap
    
    Returns:
        List of text chunks
    """
    if max_chars <= 0 or not text:
        return [text]

    # Try paragraph splitting first
    paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
    
    # Case 1: No paragraphs found (empty text)
    if not paragraphs:
        return [text.strip()]
    
    # Case 2: Single long paragraph - split by sentences
    if len(paragraphs) == 1 and len(paragraphs[0]) > max_chars:
        print(f"📝 Single long paragraph detected ({len(paragraphs[0])} chars), splitting by sentences...")
        
        # Split by sentence endings (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        print(f"   Found {len(sentences)} sentences, creating chunks with {overlap_paragraphs} sentence overlap")
        return _chunk_by_units(sentences, max_chars, overlap_paragraphs, join_with=' ')
    
    # Case 3: Multiple paragraphs - use paragraph chunking
    print(f"📝 Found {len(paragraphs)} paragraphs, creating chunks with {overlap_paragraphs} paragraph overlap")
    return _chunk_by_units(paragraphs, max_chars, overlap_paragraphs, join_with='\n\n')


def preview_chunks(text: str, max_chars: int, overlap_units: int = 5):
    """
    Preview how text would be chunked without actually processing.
    Useful for debugging and verification.
    
    Args:
        text: Text to preview
        max_chars: Max characters per chunk
        overlap_units: Number of overlap units
    """
    chunks = chunk_text(text, max_chars, overlap_units)
    
    print(f"\n{'='*60}")
    print(f"📊 CHUNK PREVIEW")
    print(f"{'='*60}")
    print(f"Total text length: {len(text):,} characters")
    print(f"Max chunk size: {max_chars:,} characters")
    print(f"Overlap units: {overlap_units}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")
    
    for i, chunk in enumerate(chunks, 1):
        # Get first and last 150 chars for preview
        start_preview = chunk[:150].replace('\n', ' ').strip()
        end_preview = chunk[-150:].replace('\n', ' ').strip()
        
        print(f"--- Chunk {i}/{len(chunks)} ({len(chunk):,} chars) ---")
        print(f"Starts: {start_preview}...")
        print(f"Ends:   ...{end_preview}")
        print()


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