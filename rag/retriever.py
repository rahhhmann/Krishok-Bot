"""
rag/retriever.py
Query -> embed -> ChromaDB top-k retrieval -> structured chunks with metadata.

Assumes rag/ingest.py already persisted a ChromaDB collection using the same
embedding model (see rag/config.py: EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
COLLECTION_NAME). Adjust the import names below if they differ in your config.py.

--- Hybrid retrieval (added) ---------------------------------------------
Pure vector search was empirically found to under-rank short, keyword-dense
diagnostic chunks on this corpus (see config.py, "Hybrid retrieval" section,
for the full test case and numbers). Fix: BM25 keyword search is run
alongside vector search, and the two rankings are merged with Reciprocal
Rank Fusion (RRF) rather than a raw score blend, since cosine similarity
and BM25 scores live on incomparable scales and RRF needs no normalization.

The BM25 index is built once at Retriever init time from every document
already in the ChromaDB collection (cheap at this corpus's scale — under a
second for ~1,000 chunks). If rank_bm25 isn't installed, or
config.HYBRID_SEARCH_ENABLED is False, the retriever transparently falls
back to pure vector search — nothing else in the app needs to change.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from rag import config

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# --- Disease-name synonym expansion -----------------------------------------
# Found via real eval (rag_eval q02): "নাবী ধ্বসা রোগ" (query) scored the
# actually-relevant chunk LOWEST of 5 because the source document says
# "মড়ক" / "লেইট ব্লাইট" instead. Same disease, different common Bengali
# names -- pure vector/BM25 search under-ranks this since neither method
# knows they're synonyms. Fix: append known synonyms to the query text
# before embedding/BM25 so all name variants contribute to the match.
# Small and explicit by design (same philosophy as router.py's keyword
# fallback) -- extend this dict as more mismatches are found via eval.
_DISEASE_SYNONYMS: dict[str, list[str]] = {
    "নাবী ধ্বসা": ["মড়ক", "লেইট ব্লাইট", "late blight"],
    "আগাম ধ্বসা": ["আর্লি ব্লাইট", "early blight"],
    "পাতা ঝলসানো": ["লিফ স্কাল্ড", "leaf scald"],
    "ব্যাকটেরিয়াল ব্লাইট": ["পাতা পোড়া রোগ", "bacterial blight"],
    "টুংরো": ["tungro"],
    "ব্লাস্ট": ["blast"],
    "বাদামী দাগ": ["brown spot"],
}


def _expand_query_synonyms(query: str) -> str:
    """Appends known synonym terms found in the query so embedding/BM25
    matching isn't blocked by which name a source document happens to use.
    Only appends terms whose trigger phrase is actually present -- never
    rewrites or removes anything from the original query."""
    extra_terms = []
    for trigger, synonyms in _DISEASE_SYNONYMS.items():
        if trigger in query:
            extra_terms.extend(synonyms)
    if not extra_terms:
        return query
    return query + " " + " ".join(extra_terms)


@dataclass
class RetrievedChunk:
    text: str
    source: str          # source_file, e.g. "Alu_2.pdf"
    title: str = ""
    page: Optional[int] = None
    chunk_id: Optional[str] = None   # ChromaDB id: "{source_file}::p{page}::{i}"
    chapter: str = ""
    source_org: str = ""
    source_url: str = ""
    is_table: bool = False
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class Retriever:
    """Loads embedding model + ChromaDB collection once; reused across calls."""

    def __init__(
        self,
        persist_dir: str = str(config.CHROMA_PERSIST_DIR),
        collection_name: str = config.COLLECTION_NAME,
        embedding_model: str = config.EMBEDDING_MODEL_PRIMARY,
        embedding_model_fallback: str = config.EMBEDDING_MODEL_FALLBACK,
        min_similarity: float = 0.0,
        hybrid: bool = config.HYBRID_SEARCH_ENABLED,
    ):
        self.min_similarity = min_similarity
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"ChromaDB load failed at '{persist_dir}': {e}")
            raise RuntimeError(
                f"Could not open ChromaDB collection '{collection_name}' at "
                f"'{persist_dir}'. Run rag/ingest.py first."
            ) from e

        # ingest.py stores the model actually used as collection metadata
        # (see _embed_and_persist). Prefer that over guessing, so retriever
        # always matches what the vectors were built with.
        collection_model = (self._collection.metadata or {}).get("embedding_model")
        model_to_load = collection_model or embedding_model
        if collection_model and collection_model != embedding_model:
            logger.info(
                f"Collection was embedded with '{collection_model}' "
                f"(not the configured primary '{embedding_model}') — using that."
            )

        try:
            self._embedder = SentenceTransformer(model_to_load)
            self._embedding_model_used = model_to_load
        except Exception as e:
            logger.warning(f"Failed to load '{model_to_load}': {e}")
            try:
                self._embedder = SentenceTransformer(embedding_model_fallback)
                self._embedding_model_used = embedding_model_fallback
                logger.warning(
                    f"Using fallback '{embedding_model_fallback}'. If this differs "
                    "from the collection's embedding_model, retrieval quality will "
                    "be degraded — re-ingest with a consistent model instead."
                )
            except Exception as e2:
                logger.error(f"Fallback embedding model also failed: {e2}")
                raise RuntimeError("Could not load any embedding model.") from e2

        count = self._collection.count()
        logger.info(f"Retriever ready: {count} chunks in '{collection_name}'.")
        if count == 0:
            logger.warning("Collection is empty — retrieval will return nothing.")

        # --- Hybrid (BM25) index setup ---
        self._hybrid_enabled = hybrid and _BM25_AVAILABLE and count > 0
        if hybrid and not _BM25_AVAILABLE:
            logger.warning(
                "config.HYBRID_SEARCH_ENABLED is True but rank_bm25 isn't "
                "installed — falling back to pure vector search. "
                "Run: pip install rank_bm25"
            )
        self._bm25 = None
        self._bm25_ids: list[str] = []
        if self._hybrid_enabled:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Load every chunk from the collection and build a BM25 index
        over it. Run once at startup — see module docstring for why."""
        try:
            data = self._collection.get(include=["documents"])
        except Exception as e:
            logger.warning(f"Could not load corpus for BM25 index, disabling hybrid search: {e}")
            self._hybrid_enabled = False
            return

        self._bm25_ids = data["ids"]
        docs = data["documents"]
        token_re = re.compile(config.BM25_TOKEN_PATTERN)
        tokenized = [token_re.findall(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(docs)} chunks for hybrid retrieval.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed query (already normalized to Bangla script upstream) and
        fetch top_k chunks. Uses hybrid vector+BM25 fusion when available,
        pure vector search otherwise (see module docstring)."""
        if not query or not query.strip():
            return []

        query = _expand_query_synonyms(query)

        if self._hybrid_enabled:
            return self._hybrid_retrieve(query, top_k)
        return self._vector_retrieve(query, top_k)

    def _vector_query(self, query: str, n_results: int):
        """Runs the raw ChromaDB vector query and returns
        (docs, metas, dists, ids) lists, or ([],[],[],[]) on failure."""
        try:
            query_vec = self._embedder.encode(query, normalize_embeddings=True).tolist()
            n_results = min(n_results, self._collection.count())
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=max(n_results, 1),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Retrieval query failed: {e}")
            return [], [], [], []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return docs, metas, dists, ids

    def _vector_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        docs, metas, dists, ids = self._vector_query(query, top_k)
        chunks = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            similarity = 1.0 - dist  # cosine distance -> similarity
            if similarity < self.min_similarity:
                continue
            chunks.append(self._to_chunk(doc, meta, cid, round(similarity, 4)))

        if not chunks:
            logger.info(f"No chunks retrieved for query: {query[:60]!r}")
        return chunks

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        pool = max(config.HYBRID_POOL_SIZE, top_k)

        # Vector leg
        v_docs, v_metas, v_dists, v_ids = self._vector_query(query, pool)
        vector_rank = {cid: rank for rank, cid in enumerate(v_ids, start=1)}
        doc_lookup = {cid: (doc, meta) for cid, doc, meta in zip(v_ids, v_docs, v_metas)}
        vector_similarity = {cid: 1.0 - dist for cid, dist in zip(v_ids, v_dists)}

        # BM25 leg
        token_re = re.compile(config.BM25_TOKEN_PATTERN)
        query_tokens = token_re.findall(query)
        bm25_rank: dict[str, int] = {}
        if query_tokens and self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
            for rank, idx in enumerate(ranked_idx, start=1):
                cid = self._bm25_ids[idx]
                bm25_rank[cid] = rank
                if cid not in doc_lookup and scores[idx] > 0:
                    # BM25 surfaced a chunk the vector leg didn't return in
                    # its pool — fetch its text/metadata so it can still be
                    # included in the fused result.
                    self._fill_missing_chunk(cid, doc_lookup)

        # Reciprocal Rank Fusion over the union of both candidate pools.
        k = config.HYBRID_RRF_K
        all_ids = set(vector_rank) | set(bm25_rank)
        fused_scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k + bm25_rank[cid])
            fused_scores[cid] = score

        # Tie-break deterministically. Ties are not just theoretical: a
        # symmetric rank swap between two candidates (e.g. vector-rank 1 /
        # bm25-rank 3 vs. vector-rank 3 / bm25-rank 1) produces an EXACT
        # float-equal fused score, and `all_ids` is built from a `set`
        # whose iteration order is not guaranteed — so sorting on
        # fused_scores alone left the winner effectively random between
        # runs (observed: the same input flipped which id ranked first
        # depending on Python's string-hash seed). Break ties by
        # preferring the better (lower) BM25 rank, since promoting strong
        # exact-keyword matches is the entire reason hybrid search exists
        # (see module docstring); fall back to vector rank, then id, so
        # the ordering is fully deterministic even if both ranks tie too.
        ranked_ids = sorted(
            fused_scores,
            key=lambda cid: (
                -fused_scores[cid],
                bm25_rank.get(cid, float("inf")),
                vector_rank.get(cid, float("inf")),
                cid,
            ),
        )[:top_k]

        chunks = []
        for cid in ranked_ids:
            if cid not in doc_lookup:
                continue
            doc, meta = doc_lookup[cid]
            # Report the underlying vector similarity when available (more
            # interpretable to a caller/UI than a raw RRF score); fall back
            # to the RRF score itself for BM25-only hits.
            display_score = round(vector_similarity.get(cid, fused_scores[cid]), 4)
            chunks.append(self._to_chunk(doc, meta, cid, display_score))

        if not chunks:
            logger.info(f"No chunks retrieved for query: {query[:60]!r}")
        return chunks

    def _fill_missing_chunk(self, cid: str, doc_lookup: dict) -> None:
        try:
            res = self._collection.get(ids=[cid], include=["documents", "metadatas"])
            if res["documents"]:
                doc_lookup[cid] = (res["documents"][0], res["metadatas"][0])
        except Exception as e:
            logger.debug(f"Could not fetch BM25-only chunk {cid}: {e}")

    @staticmethod
    def _to_chunk(doc: str, meta: dict, cid: str, score: float) -> RetrievedChunk:
        meta = meta or {}
        return RetrievedChunk(
            text=doc,
            source=meta.get("source_file", "unknown"),
            title=meta.get("title", ""),
            page=meta.get("page"),
            chunk_id=cid,
            chapter=meta.get("chapter", ""),
            source_org=meta.get("source_org", ""),
            source_url=meta.get("source_url", ""),
            is_table=bool(meta.get("is_table", False)),
            score=score,
            metadata=meta,
        )


_retriever_singleton: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Lazy singleton — avoids reloading embedding model/ChromaDB on every call."""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = get_retriever()
    test_queries = [
        ("Bangla", "ধানের বাদামী গাছ ফড়িং দমন"),
        ("Bangla - blast", "ধান ব্লাস্ট রোগ প্রতিকার"),
        ("English", "potato late blight treatment"),
        ("Banglish", "dhaner poka doman korbo kivabe"),  # raw, NOT normalized —
        # demonstrates the script-mismatch problem (see project_plan.md Sec 3)
        # that agent/normalizer.py (not yet built) is meant to fix upstream.
    ]
    for label, q in test_queries:
        print(f"\n--- [{label}] Query: {q} ---")
        chunks = r.retrieve(q, top_k=5)
        if not chunks:
            print("  (no results)")
        for c in chunks:
            print(f"  [{c.score}] {c.source} p{c.page}: {c.text[:80]}...")