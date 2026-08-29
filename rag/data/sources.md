# KrishokBot RAG Corpus — Source Log

Finalized: 2026-08-16. 20 PDFs, crop scope = rice, potato, tomato only
(per `krishokbot_rag_sources.md` policy). All Bengali-language, per the
locked language policy (Section 1 of that file).

> **Machine-readable version:** `sources.json` in this folder is what
> `ingest.py` actually reads (crop tags, page-range scoping for the
> handbook). This file is the human-readable summary — keep both in sync
> if a document is added/removed.

## Extraction reality check (important finding)

PyMuPDF text-layer inspection revealed the corpus is **not uniformly
"digital" or "scanned"** — there's a third, less obvious category:

| Category | Meaning | Count |
|---|---|---|
| `unicode_digital` | Real Unicode Bangla text layer, extract directly | 6 |
| `legacy_font_ocr` | Text layer exists but uses a non-Unicode legacy Bangla font (e.g. SutonnyMJ-style) — PyMuPDF returns ASCII/glyph soup, not real text | 7 |
| `scanned_ocr` | No text layer at all (pure scanned image) | 6 |
| `partial_font_ocr` | Text layer partially maps correctly, silently drops other glyphs as blanks — worse than fully garbled because it *looks* complete | 1 |

**Decision:** `legacy_font_ocr`, `scanned_ocr`, and `partial_font_ocr` are
all routed to the same fallback: render the page to an image (300 DPI) and
run Tesseract Bengali OCR on it. This was chosen over trying to reverse a
legacy-font glyph mapping because the mapping is font-specific and
unverifiable without a ground-truth table, whereas OCR-on-render is
font-agnostic and was spot-checked to produce genuinely readable Bangla
(verified on `dhan chash.pdf` and `Alu.pdf` — see ingest run log).
`ingest.py` doesn't trust these hints blindly — it re-checks each page's
Bangla-Unicode character ratio itself and OCRs automatically if the
direct-extraction ratio is low, so a wrong hint here can't corrupt output.

## Documents

| # | File | Crop | Pages | Category | Source org |
|---|---|---|---|---|---|
| 1 | কৃষি প্রযুক্তি হাতবই (krishiProjuktiHatboi_10.pdf) | rice, potato, tomato | 650 (scoped to ~193 relevant) | legacy_font_ocr | BARI |
| 2 | dhan chash.pdf (আধুনিক ধানের চাষ) | rice | 132 | legacy_font_ocr | BRRI |
| 3 | dhan chasher shomossa.pdf | rice | 82 | legacy_font_ocr | BRRI |
| 4 | Dhan 2.pdf | rice | 2 | scanned_ocr | DAE |
| 5 | Dhan 3.pdf | rice | 6 | scanned_ocr | DAE |
| 6 | ধানের ব্লাস্ট রোগ ও দমন ব্যবস্থাপনা (dhaner blast rog.pdf) | rice | 1 | scanned_ocr | DAE |
| 7 | ধানের রোগ ও প্রতিকার (dhaner rog o protikar.pdf) | rice | 4 | scanned_ocr | DAE/AIS |
| 8 | Alu.pdf | potato | 1 | scanned_ocr | DAE |
| 9 | Alu 2.pdf | potato | 8 | unicode_digital | DAE/AIS |
| 10 | alu utpadon.pdf | potato | 13 | unicode_digital | DAE/AIS |
| 11 | alur_rog_protirod_guide_bangla.pdf | potato | 2 | legacy_font_ocr | unconfirmed |
| 12 | আলুর মড়ক রোগ ও তার প্রতিকার (alur morok o protikar.pdf) | potato | 2 | scanned_ocr | DAE/AIS |
| 13 | টমেটো (tomato.pdf) | tomato | 3 | unicode_digital | e-Krishi/AIS |
| 14 | টমেটোর রোগ ও প্রতিকার (tomato rog o protikar.pdf) | tomato | 4 | unicode_digital | DAE/AIS |
| 15 | tomato 1.pdf | tomato | 1 | unicode_digital | DAE |
| 16 | tomato 2.pdf | tomato | 1 | partial_font_ocr | DAE |
| 17 | tomato 4.pdf | tomato | 1 | legacy_font_ocr | DAE |
| 18 | টমেটোর রোগ (Colletotrichum coccodes) (tomato 3.pdf) | tomato | 1 | legacy_font_ocr | DAE |
| 19 | tomato utpadon.pdf | tomato | 1 | unicode_digital | DAE/AIS |
| 20 | Summer tomato production technology.pdf | tomato | 7 | legacy_font_ocr | BARI (probable) |

## Filename update (2026-08-17)

The 5 PDFs that originally had Bangla-script filenames were renamed to
ASCII/English filenames by Ashik (source content unchanged — Windows was
rendering the Bangla filenames as garbled symbols on extraction, a zip
filename-encoding issue, not a content issue). This file and `sources.json`
were updated to match:

- ধানের ব্লাস্ট রোগ ও দমন ব্যবস্থাপনা.pdf → `dhaner blast rog.pdf` (trailing
  space before `.pdf` also removed for path-safety)
- ধানের রোগ ও প্রতিকার (মূল পেজ).pdf → `dhaner rog o protikar.pdf`
- আলুর মড়ক রোগ ও তার প্রতিকার (মূল পেজ).pdf → `alur morok o protikar.pdf`
- টমেটো.pdf → `tomato.pdf`
- টমেটোর রোগ ও প্রতিকার.pdf → `tomato rog o protikar.pdf`
- toamto 3pdf.pdf (typo: doubled "pdf", misspelled "tomato") → `tomato 3.pdf`

The Bangla `title` field in `sources.json` is unchanged — only `filename`
(the on-disk path `ingest.py` reads) was updated.

## Open items — need Ashik's confirmation before README claims a source

Most district/DAE leaflet files (`Dhan 2.pdf`, `Dhan 3.pdf`, `Alu.pdf`,
`tomato 1/2/4.pdf`, `toamto 3pdf.pdf`, etc.) don't have an original
source URL or download date recorded on disk. I've marked these
`"not recorded — needs confirmation from user"` in `sources.json` rather
than inventing a plausible-looking DAE/AIS URL. If you have the original
links, send them and I'll fill these in — this matters for README
citation honesty (Section 31 of the plan: "the source shown in the UI
must correspond to the actual retrieved document").

## Handbook page-range scoping

`krishiProjuktiHatboi_10.pdf` is 650 pages covering ~15 crop categories;
only 4 chapters are in scope (rice/potato/tomato + general pathology).
Full-book OCR at the measured ~2.8s/page would cost ~30 min and mostly
process irrelevant chapters (fruit, flower, spice crops, farm machinery,
etc.). Scoped via the PDF's own bookmark/TOC (readable — it's in
transliterated English, unaffected by the legacy-font issue) to:

- Kondal Fosal (tuber crops incl. potato): pages 26–93
- Sobji Fosal (vegetables incl. tomato): pages 164–260
- Dana Fosal (grain crops incl. rice): pages 485–500
- Pathology (general disease management): pages 564–575

~193 pages total (~9 min OCR) instead of 650. Full TOC preserved in
`sources.json` notes if a later phase needs other chapters (e.g. IPM,
irrigation).
