"""Per-page PDF text extraction with automatic OCR fallback.

Why this exists (not just PyMuPDF + Tesseract-for-scanned-pages, as
originally planned): inspecting the actual 20-document corpus showed a
third failure mode beyond "digital" vs "scanned" — several government
PDFs have a text layer that *looks* present but uses a non-Unicode
legacy Bangla font (glyph codes with no ToUnicode CMap), so
`page.get_text()` returns ASCII/Latin glyph soup instead of real Bangla.
One document even partially maps (some glyphs correct, others silently
dropped to blank space), which is worse than fully garbled because it
looks complete. See rag/data/sources.md for the full breakdown.

Rather than trying to reverse-engineer a font-specific glyph mapping
(fragile, unverifiable without a ground-truth table), every page is
independently checked: if its direct-extracted text isn't
majority-Bangla-Unicode, the page is rendered to an image and OCR'd with
Tesseract's Bengali model instead. This makes the pipeline robust to any
future PDF regardless of which legacy font it was authored in.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

from rag.config import (
    ALWAYS_OCR,
    BANGLA_RATIO_OCR_THRESHOLD,
    MIN_CHARS_FOR_DIRECT_TEXT,
    OCR_DPI,
    OCR_LANG,
)

logger = logging.getLogger(__name__)

_BANGLA_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_ALPHA_LIKE_RE = re.compile(r"[^\s\d.,;:()\-|/]")


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed
    text: str
    method: str  # "direct" | "ocr"
    bangla_ratio: float
    tables: list[list[list[str]]] = field(default_factory=list)


def _bangla_ratio(text: str) -> float:
    """Fraction of non-trivial characters that are Bangla-Unicode.

    Used to distinguish real Bangla text from legacy-font glyph soup,
    which is mostly Latin/symbol characters even though non-empty.
    """
    if not text.strip():
        return 0.0
    bangla = len(_BANGLA_CHAR_RE.findall(text))
    alpha_like = len(_ALPHA_LIKE_RE.findall(text))
    return bangla / alpha_like if alpha_like else 0.0


def _ocr_page(page: fitz.Page) -> str:
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except pytesseract.TesseractError as exc:
        logger.warning("OCR failed on page: %s", exc)
        return ""


def _extract_tables_for_page(pdf_path: str, page_index: int) -> list[list[list[str]]]:
    """Best-effort table extraction via pdfplumber for a single page.

    Only attempted for direct-text pages — pdfplumber's table detector
    relies on the same text/line layout that's meaningless on scanned
    or legacy-font pages, so calling it there would be wasted work.
    """
    tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return tables
            for table in pdf.pages[page_index].extract_tables():
                cleaned = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                    if any(cell and cell.strip() for cell in row)
                ]
                if cleaned:
                    tables.append(cleaned)
    except Exception as exc:  # pdfplumber can raise on malformed PDFs
        logger.warning("Table extraction failed on %s page %d: %s", pdf_path, page_index, exc)
    return tables


def extract_pdf(
    pdf_path: str,
    page_range: tuple[int, int] | None = None,
) -> list[PageExtraction]:
    """Extract text (with automatic OCR fallback) and tables per page.

    Args:
        pdf_path: path to the PDF file.
        page_range: optional inclusive (start, end) 1-indexed page range
            to restrict extraction to (used for scoping the 650-page
            BARI handbook to its relevant chapters — see sources.json).

    Returns:
        One PageExtraction per processed page, in page order.
    """
    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    if page_range:
        start, end = page_range
        start = max(1, start)
        end = min(n_pages, end)
        page_indices = range(start - 1, end)
    else:
        page_indices = range(n_pages)

    results: list[PageExtraction] = []
    for idx in page_indices:
        page = doc[idx]
        direct_text = page.get_text()
        direct_ratio = _bangla_ratio(direct_text)
        char_count = len(direct_text.strip())

        ratio_check_passes = (
            char_count >= MIN_CHARS_FOR_DIRECT_TEXT
            and direct_ratio >= BANGLA_RATIO_OCR_THRESHOLD
        )
        # See config.ALWAYS_OCR docstring: the ratio check alone is not
        # sufficient evidence for this corpus (silent word-level glyph
        # corruption slips past it), so OCR is used regardless unless
        # explicitly disabled.
        use_direct = ratio_check_passes and not ALWAYS_OCR

        if use_direct:
            text = direct_text
            method = "direct"
            final_ratio = direct_ratio
            tables = _extract_tables_for_page(pdf_path, idx)
        else:
            text = _ocr_page(page)
            method = "ocr"
            # BUG FIX (was computing ratio from the discarded direct_text
            # here, so every OCR'd page recorded bangla_ratio ~0.0
            # regardless of how clean the actual stored OCR text was —
            # since ALWAYS_OCR=True, that meant essentially the whole
            # corpus). Recompute on the text that is actually kept.
            final_ratio = _bangla_ratio(text)
            tables = []  # OCR path: table structure isn't recoverable this way

        results.append(
            PageExtraction(
                page_number=idx + 1,
                text=text,
                method=method,
                bangla_ratio=final_ratio,
                tables=tables,
            )
        )

    doc.close()
    return results
