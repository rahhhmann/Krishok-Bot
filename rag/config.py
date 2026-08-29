"""Centralized configuration for the KrishokBot RAG pipeline.

All paths/thresholds live here instead of being scattered across
ingest.py / retriever.py, per project engineering rules. Values can be
overridden via environment variables so the same code runs locally,
on Kaggle, and in the Streamlit Cloud deployment without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("KRISHOKBOT_RAG_DATA_DIR", RAG_DIR / "data"))
CHROMA_PERSIST_DIR = Path(
    os.getenv("KRISHOKBOT_CHROMA_DIR", RAG_DIR / "chroma_db")
)
SOURCES_JSON = DATA_DIR / "sources.json"
MANIFEST_PATH = CHROMA_PERSIST_DIR / "ingest_manifest.json"

# --- Embedding model -----------------------------------------------------
# Bengali-specific model preferred (corpus is Bengali-only, see
# krishokbot_rag_sources.md Section 1). Falls back to a general
# multilingual model if the Bengali one fails to load (e.g. offline
# sandbox, or underperforms in retrieval benchmarking).
EMBEDDING_MODEL_PRIMARY = os.getenv(
    "KRISHOKBOT_EMBEDDING_MODEL",
    "l3cube-pune/bengali-sentence-similarity-sbert",
)
EMBEDDING_MODEL_FALLBACK = os.getenv(
    "KRISHOKBOT_EMBEDDING_MODEL_FALLBACK",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

COLLECTION_NAME = os.getenv("KRISHOKBOT_COLLECTION", "krishokbot_agri_docs")

# --- OCR ------------------------------------------------------------------
OCR_LANG = "ben"
OCR_DPI = 300
# Below this ratio of Bangla-Unicode chars among non-whitespace/punct
# chars, a page's direct text layer is treated as untrustworthy
# (legacy-font garbage or partial glyph mapping) and OCR is used instead.
BANGLA_RATIO_OCR_THRESHOLD = 0.5
# Below this raw character count, treat the page as having no usable
# text layer at all (pure scanned image) regardless of ratio.
MIN_CHARS_FOR_DIRECT_TEXT = 30

# IMPORTANT — read before changing this to False:
# Manual A/B verification on this corpus (see rag/data/sources.md,
# "Extraction reality check") found that even pages passing the
# Bangla-ratio check above can be silently WORD-CORRUPTED — individual
# conjunct/matra glyphs mapped to the wrong-but-still-valid Bangla
# Unicode codepoint (e.g. "চাষের" extracted as "ত্ষের", "গুরুত্বপূর্ণ"
# as "গুরুত্বপূর্ব"). The Bangla-ratio heuristic cannot detect this
# because the output IS majority Bangla-Unicode — it's just wrong.
# Direct comparison against OCR on the same page (Alu 2.pdf, page 1)
# showed OCR producing correct, readable text where direct extraction
# did not. Given this, OCR-render is used for every page in this corpus
# by default, regardless of what the ratio check would say — the ratio
# check is kept (and still logged) for transparency/debugging and as a
# reusable building block for a future, cleaner corpus, but it does not
# gate the extraction decision on its own.
ALWAYS_OCR = True

# --- Chunking ---------------------------------------------------------
# Character-based (not token-based) since Bengali tokenization for the
# chosen embedding model isn't whitespace-delimited the way English is;
# character counts are a simpler, more predictable proxy here.
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40  # drop near-empty chunks (OCR noise, stray headers)

# Bangla sentence-ending punctuation used to prefer chunk boundaries at
# sentence edges rather than mid-sentence.
BANGLA_SENTENCE_ENDERS = ("।", "?", "!", ".")

# --- Line-level garbage filtering (post-OCR cleanup) --------------------
# Tesseract with lang="ben" forces Bengali-script decoding even on Latin
# text (scientific/binomial names, stray English), producing symbol-soup
# lines like "(0/1610171011:/711 [177৫071171/170711/71!" rather than
# readable text. Lines this garbled are dropped in clean_page_text.
# Short lines are exempt so legitimate short English terms (NPK, pH, EC)
# survive.
LOW_BANGLA_LINE_RATIO_THRESHOLD = 0.3
MIN_LINE_LEN_FOR_GARBAGE_CHECK = 15

# --- Hybrid retrieval (vector + keyword) ---------------------------------
# WHY THIS EXISTS: pure vector search on this corpus was empirically found
# to under-rank short, keyword-dense diagnostic chunks. Test case: query
# "ধান ব্লাস্ট রোগ প্রতিকার" against the real 955-chunk corpus did not
# surface ANY chunk from "dhaner blast rog.pdf" (the dedicated, most
# authoritative blast-disease document) even at top_k=50 with pure vector
# search, despite 3 of its 4 chunks containing the exact word "ব্লাস্ট"
# and matching diagnostic content. Root cause: the embedding model
# (l3cube-pune/bengali-sentence-similarity-sbert) captures overall topic
# similarity well but under-weights exact rare/technical keyword matches
# on this corpus's short OCR'd chunks.
#
# FIX: combine vector search with BM25 keyword search via Reciprocal Rank
# Fusion (RRF) — a rank-based fusion method that needs no score
# normalization between the two very different scoring scales (cosine
# similarity vs. BM25 term-frequency scores). Verified on this exact
# corpus/query: BM25 alone ranks 2 of the 3 blast chunks in its own top 3;
# after RRF fusion with the real vector ranking, the blast document lands
# at fused rank 5 (from being entirely absent in top 50 under vector-only
# search). See rag/retriever.py Retriever._hybrid_retrieve.
HYBRID_SEARCH_ENABLED = True
# Reciprocal Rank Fusion constant. Standard default from the original RRF
# paper (Cormack et al., 2009) — de-emphasizes rank-1 dominance while
# still rewarding top ranks; not corpus-tuned, safe default.
HYBRID_RRF_K = 60
# How many candidates each retriever (vector, BM25) contributes to the
# fusion pool before top_k final results are cut. Must be >= the largest
# top_k callers will realistically request; 30 comfortably covers the
# agent's usage (top_k typically 5-10) with headroom.
HYBRID_POOL_SIZE = 30
# Bangla/Latin/digit word tokenizer for BM25. Kept intentionally simple
# (no stemming/normalization) — Bengali morphological stemming is a
# separate, unsolved problem for this project scope; a plain word-boundary
# tokenizer is enough for BM25 to catch exact/near-exact keyword overlap,
# which is precisely the gap vector search leaves.
BM25_TOKEN_PATTERN = r"[\u0980-\u09FFa-zA-Z0-9]+"
