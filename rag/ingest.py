"""RAG corpus ingestion pipeline.

Usage:
    python -m rag.ingest                 # ingest new/changed docs only
    python -m rag.ingest --force          # re-ingest everything
    python -m rag.ingest --only "Alu.pdf" # ingest a single document

Pipeline per document (see rag/data/sources.md for the corpus and the
extraction-strategy findings that shaped this):
    1. Load per-file metadata (crop, optional page-range scoping) from
       rag/data/sources.json.
    2. Extract text page-by-page via rag.pdf_extract (auto-detects
       digital vs OCR-needed pages, pulls tables where available).
    3. Clean text: strip repeated headers/footers, normalize whitespace.
    4. Chunk into overlapping, sentence-boundary-aware pieces.
    5. Embed with a Bengali sentence-transformer (fallback to a
       multilingual model if that fails to load).
    6. Upsert into a persistent ChromaDB collection with metadata
       (source file, crop, page, extraction method, table flag) so the
       UI can show real source citations (Section 31 of the plan).

Idempotency: a manifest (content hash per source file) is kept so
re-running only re-ingests new or changed files, unless --force is
passed. This makes the pipeline safe to re-run after adding new PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from rag.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_FALLBACK,
    EMBEDDING_MODEL_PRIMARY,
    MANIFEST_PATH,
    SOURCES_JSON,
)
from rag.pdf_extract import extract_pdf
from rag.text_clean import chunk_text, clean_page_text, find_repeated_lines, table_to_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.ingest")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_sources() -> list[dict]:
    if not SOURCES_JSON.exists():
        raise FileNotFoundError(
            f"{SOURCES_JSON} not found. Every ingested PDF must be listed "
            "there with at least a filename and crop tag — see "
            "rag/data/sources.md for the corpus and format."
        )
    data = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    return data["documents"]


def _load_embedder():
    """Load the Bengali sentence-transformer, falling back to a
    multilingual model if it can't be loaded (e.g. no internet access
    to huggingface.co, or the model repo changes/is unavailable).

    This function is the one part of the pipeline that could not be
    exercised end-to-end in the development sandbox — the sandbox's
    network allowlist doesn't include huggingface.co. Everything above
    this point (PDF extraction, OCR, cleaning, chunking) was verified
    with the real 20-document corpus.
    """
    from sentence_transformers import SentenceTransformer

    for model_name in (EMBEDDING_MODEL_PRIMARY, EMBEDDING_MODEL_FALLBACK):
        try:
            logger.info("Loading embedding model: %s", model_name)
            model = SentenceTransformer(model_name)
            logger.info("Loaded embedding model: %s", model_name)
            return model, model_name
        except Exception as exc:
            logger.warning("Failed to load %s: %s", model_name, exc)
    raise RuntimeError(
        "Could not load either the primary or fallback embedding model. "
        "Check internet access to huggingface.co, or set "
        "KRISHOKBOT_EMBEDDING_MODEL to a locally cached model."
    )


def process_document(doc_meta: dict) -> list[dict]:
    """Extract, clean, and chunk a single document into ingest-ready records.

    Each record: {text, source_file, title, crop, source_org, source_url,
    page, extraction_method, bangla_ratio, is_table}
    """
    filename = doc_meta["filename"]
    path = DATA_DIR / filename
    if not path.exists():
        logger.error("Listed in sources.json but missing on disk: %s", filename)
        return []

    page_ranges = doc_meta.get("page_ranges")
    records: list[dict] = []

    if page_ranges:
        # Scoped extraction (e.g. the 650-page handbook) — process each
        # relevant chapter range separately, tagging which chapter a
        # chunk came from.
        for chapter, (start, end) in page_ranges.items():
            pages = extract_pdf(str(path), page_range=(start, end))
            records.extend(_pages_to_records(pages, doc_meta, chapter_tag=chapter))
    else:
        pages = extract_pdf(str(path))
        records.extend(_pages_to_records(pages, doc_meta, chapter_tag=None))

    return records


def _pages_to_records(pages, doc_meta: dict, chapter_tag: str | None) -> list[dict]:
    page_texts = [p.text for p in pages]
    boilerplate = find_repeated_lines(page_texts)

    records = []
    for page in pages:
        cleaned = clean_page_text(page.text, boilerplate)

        for table in page.tables:
            table_text = table_to_text(table)
            if len(table_text.strip()) >= 20:
                records.append(
                    _make_record(table_text, doc_meta, page, chapter_tag, is_table=True)
                )

        for chunk in chunk_text(cleaned):
            records.append(
                _make_record(chunk, doc_meta, page, chapter_tag, is_table=False)
            )
    return records


def _make_record(text: str, doc_meta: dict, page, chapter_tag, is_table: bool) -> dict:
    return {
        "text": text,
        "source_file": doc_meta["filename"],
        "title": doc_meta.get("title", doc_meta["filename"]),
        "crop": ",".join(doc_meta.get("crop", [])),
        "source_org": doc_meta.get("source_org", "unspecified"),
        "source_url": doc_meta.get("source_url", "not recorded"),
        "page": page.page_number,
        "chapter": chapter_tag or "",
        "extraction_method": page.method,
        "bangla_ratio": round(page.bangla_ratio, 3),
        "is_table": is_table,
    }


def run(force: bool = False, only: str | None = None) -> dict:
    """Run the full ingestion pipeline. Returns a summary dict."""
    t0 = time.time()
    sources = _load_sources()
    if only:
        sources = [s for s in sources if s["filename"] == only]
        if not sources:
            raise ValueError(f"No document named {only!r} in sources.json")

    manifest = _load_manifest()
    to_process = []
    skipped = []
    for doc_meta in sources:
        path = DATA_DIR / doc_meta["filename"]
        if not path.exists():
            logger.error("Missing file, skipping: %s", doc_meta["filename"])
            continue
        file_hash = _file_hash(path)
        if not force and manifest.get(doc_meta["filename"]) == file_hash:
            skipped.append(doc_meta["filename"])
            continue
        to_process.append((doc_meta, file_hash))

    logger.info(
        "%d document(s) to process, %d unchanged (skipped)",
        len(to_process),
        len(skipped),
    )

    all_records: list[dict] = []
    per_doc_counts: dict[str, int] = {}
    for doc_meta, file_hash in to_process:
        logger.info("Processing: %s", doc_meta["filename"])
        t_doc = time.time()
        records = process_document(doc_meta)
        per_doc_counts[doc_meta["filename"]] = len(records)
        all_records.extend(records)
        manifest[doc_meta["filename"]] = file_hash
        logger.info(
            "  -> %d chunks in %.1fs", len(records), time.time() - t_doc
        )

    if not all_records:
        logger.info("Nothing new to embed/persist.")
        _save_manifest(manifest)
        return {
            "processed": [d["filename"] for d, _ in to_process],
            "skipped": skipped,
            "chunks_created": 0,
            "per_document_chunks": per_doc_counts,
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    _embed_and_persist(all_records)
    _save_manifest(manifest)

    summary = {
        "processed": [d["filename"] for d, _ in to_process],
        "skipped": skipped,
        "chunks_created": len(all_records),
        "per_document_chunks": per_doc_counts,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    logger.info("Ingestion summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def _embed_and_persist(records: list[dict]) -> None:
    import chromadb

    model, model_name = _load_embedder()
    texts = [r["text"] for r in records]

    logger.info("Embedding %d chunks with %s ...", len(texts), model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": model_name},
    )

    ids = [
        f"{r['source_file']}::p{r['page']}::{i}" for i, r in enumerate(records)
    ]
    metadatas = [
        {k: v for k, v in r.items() if k != "text"} for r in records
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Persisted %d chunks to collection '%s' at %s", len(ids), COLLECTION_NAME, CHROMA_PERSIST_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest KrishokBot RAG corpus")
    parser.add_argument("--force", action="store_true", help="Re-ingest all documents")
    parser.add_argument("--only", help="Ingest a single document by filename")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract/clean/chunk only — skip embedding + ChromaDB (no model download needed)",
    )
    args = parser.parse_args()

    if args.dry_run:
        sources = _load_sources()
        if args.only:
            sources = [s for s in sources if s["filename"] == args.only]
        total = 0
        for doc_meta in sources:
            records = process_document(doc_meta)
            logger.info("%s -> %d chunks (dry-run, not embedded)", doc_meta["filename"], len(records))
            total += len(records)
        logger.info("Dry-run total: %d chunks across %d documents", total, len(sources))
        return

    try:
        run(force=args.force, only=args.only)
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
