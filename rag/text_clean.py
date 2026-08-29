"""Text cleaning and chunking, tuned for Bengali agricultural documents.

Two separate concerns kept in separate functions on purpose: cleaning
operates per-page (needs the full set of a document's pages to detect
repeated headers/footers), chunking operates on the cleaned, concatenated
document text.
"""

from __future__ import annotations

import re
from collections import Counter

from rag.config import (
    BANGLA_SENTENCE_ENDERS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    LOW_BANGLA_LINE_RATIO_THRESHOLD,
    MIN_CHUNK_CHARS,
    MIN_LINE_LEN_FOR_GARBAGE_CHECK,
)

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BANGLA_DIGIT_LINE_RE = re.compile(r"^[\s০-৯0-9|.\-–]{1,10}$")

# Duplicated (not imported) from pdf_extract.py deliberately — cleaning
# stays decoupled from the extraction module (single responsibility;
# extraction decides direct-vs-OCR, cleaning only judges the text it's
# handed).
_BANGLA_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_ALPHA_LIKE_RE = re.compile(r"[^\s\d.,;:()\-|/]")


def _line_bangla_ratio(line: str) -> float:
    """Fraction of non-trivial chars in a single line that are Bangla-Unicode."""
    alpha_like = len(_ALPHA_LIKE_RE.findall(line))
    if not alpha_like:
        return 0.0
    return len(_BANGLA_CHAR_RE.findall(line)) / alpha_like


def find_repeated_lines(page_texts: list[str], min_occurrences: int = 3) -> set[str]:
    """Detect header/footer boilerplate repeated across many pages of one doc.

    A line that appears near-identically on >= min_occurrences pages
    (e.g. "কৃষি প্রযুক্তি হাতবই ৬ষ্ঠ সংস্করণ" on every page) is noise,
    not content, and is stripped in `clean_page_text`.
    """
    if len(page_texts) < min_occurrences:
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = {ln.strip() for ln in text.split("\n") if ln.strip()}
        counts.update(lines)
    return {line for line, c in counts.items() if c >= min_occurrences and len(line) < 80}


def clean_page_text(text: str, boilerplate_lines: set[str]) -> str:
    """Strip repeated headers/footers, standalone page-number lines,
    OCR glyph-soup lines, and normalize whitespace. Preserves paragraph
    structure (blank lines)."""
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line in boilerplate_lines:
            continue
        if _BANGLA_DIGIT_LINE_RE.match(line):
            continue  # bare page number, "- 12 -", etc.
        # OCR forcing Bengali-script decoding on Latin text (scientific/
        # binomial names, stray non-Bangla content) produces symbol-soup
        # like "(0/1610171011:/711 [177৫071171/170711/71!" — drop lines
        # long enough to be real content but too non-Bangla to be real
        # Bangla text. Short lines are exempt so legitimate short English
        # terms (NPK, pH, cultivar codes) survive.
        if (
            len(line) >= MIN_LINE_LEN_FOR_GARBAGE_CHECK
            and _line_bangla_ratio(line) < LOW_BANGLA_LINE_RATIO_THRESHOLD
        ):
            continue
        lines.append(line)

    joined = "\n".join(lines)
    joined = _WHITESPACE_RE.sub(" ", joined)
    joined = _MULTI_NEWLINE_RE.sub("\n\n", joined)
    return joined.strip()


def _find_sentence_boundary(text: str, target: int, window: int = 120) -> int:
    """Search backward from `target` for the nearest sentence-ending
    punctuation, within `window` chars, to avoid cutting mid-sentence.
    Falls back to the nearest word boundary (not a hard mid-word cut) if
    no sentence-ender is found nearby — common on OCR'd pages where the
    "।" glyph itself was misread or dropped."""
    lo = max(0, target - window)
    for i in range(target, lo, -1):
        if i < len(text) and text[i - 1] in BANGLA_SENTENCE_ENDERS:
            return i
    for i in range(target, lo, -1):
        if i < len(text) and text[i - 1].isspace():
            return i
    return target


def _snap_start_to_boundary(text: str, target: int, window: int = 60) -> int:
    """Move `target` forward to the next whitespace (word boundary),
    within `window` chars, so overlap-shifted chunk starts don't begin
    mid-word (e.g. 'ুমে', 'িকাংশ'). Falls back to `target` if none found."""
    n = len(text)
    hi = min(n, target + window)
    for i in range(target, hi):
        if text[i].isspace():
            return i + 1
    return target


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split cleaned text into overlapping chunks, preferring to break
    at Bangla sentence boundaries (।, ?, !, .) rather than mid-word.

    Both chunk ends AND chunk starts are boundary-snapped — snapping only
    the end still lets overlap-shifted starts land mid-word.

    Duplicate guard: snap-forward(start) combined with snap-backward(end)
    can otherwise converge on nearly the same [start, end) window twice in
    a row, producing duplicate or near-duplicate consecutive chunks
    (observed as identical chunks in dhan chash.pdf p79 after the start-snap
    fix was added). `min_progress` enforces a minimum distance between
    consecutive chunk starts to prevent this.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    prev_start = -1
    # Minimum required distance between consecutive chunk starts. Derived
    # from chunk_size - overlap (the "intended" step size) minus the
    # start-snap window, so the snap can never fully cancel out the step.
    min_progress = max(1, chunk_size - overlap - 60)  # 60 = start-snap window

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_sentence_boundary(text, end)

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS and start >= prev_start + min_progress:
            chunks.append(chunk)
            prev_start = start

        if end >= n:
            break

        next_start = max(end - overlap, start + 1)  # guarantee forward progress
        next_start = max(next_start, prev_start + min_progress)  # avoid re-converging
        start = _snap_start_to_boundary(text, next_start)

    return chunks


def table_to_text(table: list[list[str]]) -> str:
    """Render an extracted table as pipe-delimited rows so it embeds
    and reads reasonably as a retrieval chunk (dosage/schedule tables
    are exactly the content Section 11 flags as needing preservation)."""
    return "\n".join(" | ".join(cell for cell in row if cell) for row in table)
