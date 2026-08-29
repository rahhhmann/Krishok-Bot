# RAG Ingestion — Validation Report

Ran: 2026-08-16, in the development sandbox, against the real 20-document
corpus in `rag/data/`. This documents what was actually executed and
measured — not projected.

## What was fully tested end-to-end
PDF → per-page extraction (with auto OCR fallback) → header/footer
cleaning → sentence-aware chunking. Every one of the 20 documents was
run through this real code path (large documents in page-range batches
to fit sandbox execution-time limits — batching is a sandbox artifact
only, `rag/ingest.py` processes a whole document in one call in a normal
environment).

## Key finding that changed the pipeline design
Initial plan (per `krishokbot_rag_sources.md`) was "digital PDFs via
PyMuPDF, scanned PDFs via OCR." Actual inspection found a third,
more dangerous failure mode: several PDFs have a text layer that reads
as valid Bangla Unicode (passes any ratio-based check) but is
**word-corrupted** — individual conjunct/matra glyphs mapped to the
wrong-but-still-valid codepoint (`চাষের` → `ত্ষের`, `গুরুত্বপূর্ণ` →
`গুরুত্বপূর্ব`). Direct A/B test on `Alu 2.pdf` page 1:

- Direct extraction: `আলু ত্ষের অন্যিম গুরুত্বপূর্ব খাদ্য শস্য` (wrong)
- OCR (300 DPI render + Tesseract `ben`): `আলু বিশ্বের অন্যতম গুরুত্বপূর্ণ খাদ্য শস্য` (correct)

**Decision:** OCR is now used for every page in this corpus regardless
of what the direct-text-layer looks like (`config.ALWAYS_OCR = True`,
with full reasoning in a comment there). The ratio-based heuristic is
still computed and stored in each chunk's metadata for transparency, it
just doesn't gate the extraction decision anymore.

## Measured totals

| Document | Pages processed | Chunks | Extraction method |
|---|---:|---:|---|
| 17 short/medium leaflets (all except the two below + handbook) | ~55 | 184 | OCR |
| dhan chash.pdf (BRRI, আধুনিক ধানের চাষ) | 132 | 392 | OCR |
| dhan chasher shomossa.pdf (BRRI) | 82 | 156 | OCR |
| Handbook — Kondal Fosal (potato) | 68 | 244 | OCR |
| Handbook — Sobji Fosal (veg/tomato) | 97 | 376 | OCR |
| Handbook — Dana Fosal (rice) | 16 | 61 | OCR |
| Handbook — Pathology | 12 | 41 | OCR |
| **Total** | **~462** | **~1,454** | — |

Table extraction (pdfplumber) found 0 usable tables in any batch. This
is expected and not a bug: pdfplumber's table detector needs a real
text/line layout, which none of these pages have (they're all routed
through OCR precisely because their native layer isn't trustworthy).
**Consequence:** if any of these documents contain dosage/schedule
tables, that structured data is currently only present as OCR'd running
text, not as a distinguished table chunk. Worth a manual check of the
higher-value docs (dhan chash.pdf, the handbook) if dosage precision
matters for the README's RAG-quality claims.

## OCR quality — honest caveats
Spot-checking output across batches, OCR is clearly readable Bangla in
the large majority of chunks, but not perfect:
- Occasional misreads of italic Latin scientific/pathogen names as
  digit strings (e.g. a *Ralstonia* species name OCR'd as
  `73906607191`). These don't corrupt the Bangla prose around them but
  do mean pathogen binomial names are unreliable in the corpus as
  currently ingested.
- Minor character-level noise consistent with normal OCR error rates
  (a few percent of characters), not reviewed exhaustively page-by-page.

This should be stated plainly in the README's RAG-limitations section,
per the project's scientific-honesty requirement — this is real OCR
output, not manually verified transcription.

## What could NOT be tested here (genuine sandbox blocker)
The embedding step (`sentence-transformers`) requires downloading model
weights from `huggingface.co`, which returned `403 Forbidden` — this
sandbox's network is allowlisted to package registries only (pypi, npm,
github, etc.), not huggingface.co. Both the primary
(`l3cube-pune/bengali-sentence-similarity-sbert`) and fallback
(`paraphrase-multilingual-mpnet-base-v2`) load attempts were exercised
and failed with the same, correctly-caught error — confirming the
fallback logic itself works, just not the actual model download.

**To run the full pipeline (including embedding + ChromaDB persist) in
an environment with normal internet access:**

```bash
pip install -r rag/requirements.txt
apt-get install tesseract-ocr tesseract-ocr-ben   # if not already present
python -m rag.ingest
```

This will download both embedding models on first run, embed all
~1,454 chunks, and persist them to `rag/chroma_db/`. Expect the
Bengali-specific model download to be a few hundred MB.

## Not yet done (next steps, not this session's scope)
- `rag/retriever.py` (runtime query module) — not started yet.
- Benchmarking `l3cube-pune/bengali-sentence-similarity-sbert` vs the
  multilingual fallback on a small retrieval sample, per
  `krishokbot_rag_sources.md` Section 1 ("keep both as options,
  benchmark... before committing").
- Filling in the "not recorded" source URLs for leaflets without a
  confirmed origin (see `sources.md`, "Open items").
