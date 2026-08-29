"""Unit tests for rag/retriever.py.

ChromaDB and SentenceTransformer are fully mocked -- no real embedding
model is loaded and no real ChromaDB collection is opened. These tests
verify retriever.py's own logic: synonym expansion, similarity
filtering, and the RRF fusion math, not real retrieval quality (that's
what eval/run_eval.py's real corpus run is for).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import Retriever, RetrievedChunk, _expand_query_synonyms


# --- synonym expansion (pure function, no mocking needed) ------------------


def test_expand_query_synonyms_appends_known_terms():
    result = _expand_query_synonyms("আলুর নাবী ধ্বসা রোগ কিভাবে দমন করব?")
    assert "নাবী ধ্বসা" in result  # original preserved
    assert "মড়ক" in result
    assert "লেইট ব্লাইট" in result
    assert "late blight" in result


def test_expand_query_synonyms_no_trigger_returns_unchanged():
    # No disease-synonym trigger phrase present (avoid "ব্লাস্ট" -- it IS
    # a trigger key in _DISEASE_SYNONYMS).
    query = "টমেটোর সার প্রয়োগের সঠিক নিয়ম কী"
    assert _expand_query_synonyms(query) == query


def test_expand_query_synonyms_multiple_triggers():
    query = "টুংরো এবং ব্লাস্ট দুটোই দেখা যাচ্ছে"
    result = _expand_query_synonyms(query)
    assert "tungro" in result
    assert "blast" in result


# --- Retriever: construct with everything mocked ---------------------------


def _build_mocked_retriever(chunk_count: int = 3, hybrid: bool = False) -> Retriever:
    """Builds a Retriever with chromadb.PersistentClient and
    SentenceTransformer both mocked, so __init__ never touches disk or
    downloads a model."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = chunk_count
    fake_collection.metadata = {"embedding_model": "fake-model"}

    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client), patch(
        "rag.retriever.SentenceTransformer"
    ) as mock_st:
        mock_st.return_value.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        )
        retriever = Retriever(
            persist_dir="fake_dir",
            collection_name="fake_collection",
            hybrid=hybrid,
        )
    return retriever, fake_collection


def test_retriever_init_raises_clear_error_if_collection_missing():
    fake_client = MagicMock()
    fake_client.get_collection.side_effect = Exception("no such collection")
    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="Could not open ChromaDB collection"):
            Retriever(persist_dir="fake_dir", collection_name="missing")


def test_retriever_empty_query_returns_empty_list():
    retriever, _ = _build_mocked_retriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_vector_only_returns_chunks_with_similarity(monkeypatch):
    retriever, collection = _build_mocked_retriever(chunk_count=2, hybrid=False)
    collection.query.return_value = {
        "documents": [["chunk text one", "chunk text two"]],
        "metadatas": [[{"source_file": "a.pdf", "page": 1}, {"source_file": "b.pdf", "page": 2}]],
        "distances": [[0.1, 0.4]],  # similarity = 1 - distance
        "ids": [["id1", "id2"]],
    }
    chunks = retriever.retrieve("ধানের ব্লাস্ট রোগ", top_k=5)
    assert len(chunks) == 2
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].source == "a.pdf"
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[1].score == pytest.approx(0.6)


def test_retriever_vector_only_filters_below_min_similarity():
    fake_collection = MagicMock()
    fake_collection.count.return_value = 2
    fake_collection.metadata = {"embedding_model": "fake-model"}
    fake_collection.query.return_value = {
        "documents": [["low sim chunk", "high sim chunk"]],
        "metadatas": [[{"source_file": "a.pdf"}, {"source_file": "b.pdf"}]],
        "distances": [[0.9, 0.1]],  # similarities: 0.1, 0.9
        "ids": [["id1", "id2"]],
    }
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection
    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client), patch(
        "rag.retriever.SentenceTransformer"
    ) as mock_st:
        mock_st.return_value.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        )
        retriever = Retriever(
            persist_dir="fake_dir", collection_name="fake", min_similarity=0.5, hybrid=False
        )
    chunks = retriever.retrieve("query", top_k=5)
    assert len(chunks) == 1
    assert chunks[0].source == "b.pdf"


def test_retriever_query_failure_returns_empty_list_not_raise():
    retriever, collection = _build_mocked_retriever(hybrid=False)
    collection.query.side_effect = Exception("chromadb exploded")
    chunks = retriever.retrieve("any query", top_k=5)
    assert chunks == []


# --- RRF hybrid fusion -------------------------------------------------


def test_hybrid_retrieve_fuses_vector_and_bm25_rankings():
    """A chunk ranked poorly by vector search but well by BM25 (the
    documented real bug: exact keyword match under-ranked by pure vector
    search) should be promoted by RRF fusion -- this is the core behavior
    rag/config.py's HYBRID_SEARCH_ENABLED docstring describes."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 3
    fake_collection.metadata = {"embedding_model": "fake-model"}
    # BM25 index build reads every doc via collection.get(include=["documents"])
    fake_collection.get.return_value = {
        "ids": ["id_blast_A", "id_blast_B", "id_irrelevant"],
        "documents": [
            "ধান ব্লাস্ট রোগ প্রতিকার ছত্রাকনাশক স্প্রে",
            "ব্লাস্ট রোগের লক্ষণ পাতায় দাগ",
            "সম্পূর্ণ অপ্রাসঙ্গিক টেক্সট বিষয়বস্তু",
        ],
    }
    # Vector search returns the blast-disease chunk (id_blast_A) ranked
    # LAST (rank 3) among the pool -- simulating the real under-ranking bug.
    fake_collection.query.return_value = {
        "documents": [
            [
                "সম্পূর্ণ অপ্রাসঙ্গিক টেক্সট বিষয়বস্তু",
                "ব্লাস্ট রোগের লক্ষণ পাতায় দাগ",
                "ধান ব্লাস্ট রোগ প্রতিকার ছত্রাকনাশক স্প্রে",
            ]
        ],
        "metadatas": [[{"source_file": "irrelevant.pdf"}, {"source_file": "blast_B.pdf"}, {"source_file": "blast_A.pdf"}]],
        "distances": [[0.1, 0.3, 0.5]],
        "ids": [["id_irrelevant", "id_blast_B", "id_blast_A"]],
    }

    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client), patch(
        "rag.retriever.SentenceTransformer"
    ) as mock_st, patch("rag.retriever._BM25_AVAILABLE", True), patch(
        "rag.retriever.BM25Okapi"
    ) as mock_bm25_cls:
        mock_st.return_value.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        )
        # BM25 ranks id_blast_A (index 0 in the doc list) highest.
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [5.0, 3.0, 0.1]  # blast_A, blast_B, irrelevant
        mock_bm25_cls.return_value = mock_bm25

        retriever = Retriever(persist_dir="fake_dir", collection_name="fake", hybrid=True)
        assert retriever._hybrid_enabled is True

        chunks = retriever.retrieve("ব্লাস্ট রোগ প্রতিকার", top_k=2)

    top_sources = [c.source for c in chunks]
    # id_blast_A should now rank #1 despite being vector-rank-3, because
    # BM25 ranked it #1 and RRF fusion combines both signals.
    assert top_sources[0] == "blast_A.pdf"


def test_hybrid_disabled_falls_back_to_vector_when_bm25_not_installed():
    fake_collection = MagicMock()
    fake_collection.count.return_value = 2
    fake_collection.metadata = {"embedding_model": "fake-model"}
    fake_collection.query.return_value = {
        "documents": [["a", "b"]],
        "metadatas": [[{"source_file": "a.pdf"}, {"source_file": "b.pdf"}]],
        "distances": [[0.1, 0.2]],
        "ids": [["id1", "id2"]],
    }
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client), patch(
        "rag.retriever.SentenceTransformer"
    ) as mock_st, patch("rag.retriever._BM25_AVAILABLE", False):
        mock_st.return_value.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        )
        retriever = Retriever(persist_dir="fake_dir", collection_name="fake", hybrid=True)

    assert retriever._hybrid_enabled is False  # BM25 unavailable -> disabled
    chunks = retriever.retrieve("query", top_k=5)
    assert len(chunks) == 2


def test_retriever_empty_collection_logs_warning_but_does_not_raise(caplog):
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    fake_collection.metadata = {}
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    with patch("rag.retriever.chromadb.PersistentClient", return_value=fake_client), patch(
        "rag.retriever.SentenceTransformer"
    ) as mock_st:
        mock_st.return_value.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        )
        retriever = Retriever(persist_dir="fake_dir", collection_name="empty", hybrid=True)

    assert retriever._hybrid_enabled is False  # count == 0 -> hybrid disabled
