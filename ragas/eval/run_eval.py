"""eval/run_eval.py

Evaluates the KrishokBot text/RAG/tool agent against eval/test_queries.jsonl.

Run:
    python -m eval.run_eval
    python -m eval.run_eval --limit 5          # smoke test a subset
    python -m eval.run_eval --queries q01,q14  # run specific ids only

Requires a working .env (LLM provider key) and a populated ChromaDB
collection (run rag/ingest.py first). This harness makes REAL calls
through agent.graph.run_text_query — it is not a mock/offline test.

What is measured (per Section 27/39 -- no fabricated scores):

1. Routing correctness: does RouteDecision.{needs_rag,needs_weather,
   needs_price} match the query's expected_route label. Deterministic,
   always computed.

2. Retrieval hit-rate: for rag-expected queries, did retrieval return
   >=1 chunk. Deterministic, always computed.

3. Guardrail behavior: for the injection-probe query, was it flagged.
   Deterministic, always computed.

4. RAGAS metrics (faithfulness, answer_relevancy, context_precision,
   context_recall): computed ONLY if the `ragas` package is installed
   AND an LLM key is available, since RAGAS itself needs an LLM judge.
   If unavailable, this harness records ragas_available=False and
   reports the proxy metrics below instead -- it does NOT invent
   RAGAS-style scores.

5. Proxy faithfulness (fallback when ragas is unavailable): lexical
   token overlap between the final answer and the retrieved chunk
   text, using the same Bangla+Latin token pattern as rag/retriever.py
   BM25 indexing. This is a crude, clearly-labeled proxy -- it checks
   whether the answer's vocabulary is grounded in retrieved text, NOT
   whether the answer is semantically faithful. Documented as a
   limitation in the summary output, not presented as RAGAS-equivalent.

Results are written to eval/results/:
    rag_eval_results.json   -- full per-query records, machine-readable
    rag_eval_results.csv    -- flat table for spreadsheet review
    rag_eval_summary.md     -- human-readable summary with real numbers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
QUERIES_PATH = EVAL_DIR / "test_queries.jsonl"

# Same tokenizer used by rag/retriever.py's BM25 index (Bangla block + Latin
# alnum) -- reused here so the proxy-faithfulness overlap check is at least
# consistent with how the retriever itself tokenizes text.
_TOKEN_RE = re.compile(r"[\u0980-\u09FFa-zA-Z0-9]+")

try:
    import ragas  # noqa: F401
    _RAGAS_INSTALLED = True
except ImportError:
    _RAGAS_INSTALLED = False


@dataclass
class QueryResult:
    id: str
    query: str
    lang: str
    expected_route: str
    expected_crop: Optional[str]

    # What actually happened
    actual_needs_rag: Optional[bool] = None
    actual_needs_weather: Optional[bool] = None
    actual_needs_price: Optional[bool] = None
    actual_crop: Optional[str] = None
    route_source: Optional[str] = None  # "llm" | "keyword_fallback"
    route_correct: Optional[bool] = None

    normalized_query: str = ""
    detected_script: str = ""

    retrieved_chunk_count: int = 0
    citations: list = field(default_factory=list)
    retrieval_hit: Optional[bool] = None  # only meaningful when expected_route includes "rag"

    final_answer: str = ""
    answer_length_chars: int = 0

    injection_flagged: Optional[bool] = None

    proxy_faithfulness_overlap: Optional[float] = None  # 0-1, see module docstring

    latency_seconds: float = 0.0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    run_error: Optional[str] = None  # set if run_text_query itself raised


def _route_matches(expected: str, needs_rag: bool, needs_weather: bool, needs_price: bool) -> bool:
    """expected_route is a '+'-joined label like 'rag', 'weather', 'rag+price'.
    A route is correct if every flag named in the label is True AND no
    unnamed flag is unexpectedly True (strict match, not just "at least")."""
    wanted = set(expected.split("+"))
    actual = set()
    if needs_rag:
        actual.add("rag")
    if needs_weather:
        actual.add("weather")
    if needs_price:
        actual.add("price")
    return wanted == actual


def _proxy_faithfulness(answer: str, chunks: list) -> Optional[float]:
    """Fraction of the answer's content tokens that also appear somewhere
    in the retrieved chunk text. None if there's no answer or no chunks
    to compare against (metric not applicable, not zero)."""
    if not answer or not chunks:
        return None
    answer_tokens = set(t.lower() for t in _TOKEN_RE.findall(answer))
    if not answer_tokens:
        return None
    chunk_tokens: set[str] = set()
    for c in chunks:
        text = getattr(c, "text", "") or ""
        chunk_tokens.update(t.lower() for t in _TOKEN_RE.findall(text))
    if not chunk_tokens:
        return None
    overlap = answer_tokens & chunk_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def _ragas_judge():
    """Builds the RAGAS judge LLM + embeddings from this project's own
    provider config (agent/config.py), instead of letting ragas silently
    default to OpenAI (which this project has no key for -- ragas.evaluate()
    would otherwise raise an opaque "OPENAI_API_KEY not set" error).

    Judge LLM: whichever provider agent/config.LLM_PROVIDER points at
    (groq or gemini), wrapped via LangChain so ragas.llms.LangchainLLMWrapper
    can drive it.
    Embeddings (needed for answer_relevancy): the same Bengali
    sentence-transformer the RAG pipeline itself is embedded with
    (rag/config.py EMBEDDING_MODEL_PRIMARY), so the judge's notion of
    semantic similarity matches what retrieval actually uses.

    Returns (llm, embeddings) already wrapped for ragas, or (None, None)
    if the judge can't be built (missing key/package) -- caller skips
    RAGAS rather than crashing.
    """
    from agent import config as agent_config

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        logger.warning("ragas llm/embeddings wrapper import failed: %s", exc)
        return None, None

    llm = None
    if agent_config.LLM_PROVIDER == "groq" and agent_config.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            llm = LangchainLLMWrapper(
                ChatGroq(
                    model=agent_config.GROQ_MODEL,
                    api_key=agent_config.GROQ_API_KEY,
                    temperature=0.0,
                )
            )
        except ImportError:
            logger.warning(
                "KRISHOKBOT_LLM_PROVIDER=groq but langchain_groq isn't installed -- "
                "install with: pip install langchain_groq==0.1.10 --break-system-packages"
            )
    elif agent_config.LLM_PROVIDER == "gemini" and agent_config.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model=agent_config.GEMINI_MODEL,
                    google_api_key=agent_config.GEMINI_API_KEY,
                    temperature=0.0,
                )
            )
        except ImportError:
            logger.warning(
                "KRISHOKBOT_LLM_PROVIDER=gemini but langchain_google_genai isn't "
                "installed -- install with: pip install langchain_google_genai "
                "--break-system-packages"
            )
    if llm is None:
        logger.warning(
            "Could not build a RAGAS judge LLM from KRISHOKBOT_LLM_PROVIDER=%s "
            "(missing key or missing langchain provider package) -- skipping RAGAS.",
            agent_config.LLM_PROVIDER,
        )
        return None, None

    embeddings = None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        from rag import config as rag_config

        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=rag_config.EMBEDDING_MODEL_PRIMARY)
        )
    except ImportError as exc:
        logger.warning(
            "Could not build RAGAS embeddings (%s) -- answer_relevancy will be "
            "skipped for this run.",
            exc,
        )

    return llm, embeddings


def _run_ragas(
    records: list[QueryResult], raw_runs: dict, ground_truths: Optional[dict] = None
) -> Optional[dict]:
    """Attempts real RAGAS scoring. Returns None (not zeros) if it can't
    run, and logs exactly why, per Section 39: no fabricated metrics.

    Metric selection: faithfulness and answer_relevancy need only
    question/answer/contexts, so they always run when a judge LLM is
    available. context_precision and context_recall additionally require
    a ground_truth reference answer (see ragas.metrics.ContextPrecision's
    qcg evaluation mode) -- this project has no hand-written gold answers
    yet, so those two are only requested for rows where test_queries.jsonl
    actually supplies a non-empty "ground_truth" field. Passing an empty
    ground_truth string to them (the previous behavior here) doesn't
    "skip" the metric, it just scores against nothing -- so they're
    omitted entirely rather than reporting a misleading number.
    """
    if not _RAGAS_INSTALLED:
        logger.warning(
            "ragas is not installed -- skipping RAGAS metrics. Install a version "
            "pin known to work with this project's other pinned deps: "
            "pip install ragas==0.1.21 'datasets<3.0' langchain_groq==0.1.10 "
            "--break-system-packages"
        )
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.warning("ragas installed but a dependency is missing (%s) -- skipping.", exc)
        return None

    judge_llm, judge_embeddings = _ragas_judge()
    if judge_llm is None:
        return None

    ground_truths = ground_truths or {}
    rows = []
    has_ground_truth = False
    for r in records:
        if not r.expected_route.startswith("rag") and "rag" not in r.expected_route:
            continue
        raw = raw_runs.get(r.id, {})
        contexts = [getattr(c, "text", "") for c in raw.get("retrieved_chunks", [])]
        if not r.final_answer or not contexts:
            continue
        ground_truth = (ground_truths.get(r.id) or "").strip()
        if ground_truth:
            has_ground_truth = True
        rows.append(
            {
                "question": r.query,
                "answer": r.final_answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    if not rows:
        logger.warning("No RAG rows with both an answer and retrieved context -- skipping RAGAS.")
        return None

    metrics = [faithfulness, answer_relevancy] if judge_embeddings else [faithfulness]
    if not judge_embeddings:
        logger.warning("No embeddings available -- answer_relevancy skipped, faithfulness only.")
    if has_ground_truth:
        metrics += [context_precision, context_recall]
    else:
        logger.info(
            "No ground_truth values in test_queries.jsonl for the RAG rows in this "
            "run -- context_precision/context_recall skipped (they need a gold "
            "reference answer). Add a \"ground_truth\" field per query to enable them."
        )

    try:
        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        return dict(result)
    except Exception as exc:  # RAGAS can fail on LLM-judge errors, rate limits, etc.
        logger.error("RAGAS evaluation failed: %s", exc)
        return None


def run(query_ids: Optional[list[str]] = None, limit: Optional[int] = None) -> list[QueryResult]:
    # Imported here, not at module level, so `--help` and other argparse-only
    # invocations don't require .env/ChromaDB/LLM keys to be present.
    from agent.graph import run_text_query

    all_queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_queries.append(json.loads(line))

    if query_ids:
        wanted = set(query_ids)
        all_queries = [q for q in all_queries if q["id"] in wanted]
    if limit:
        all_queries = all_queries[:limit]

    if not all_queries:
        logger.error("No queries selected (check --queries ids match test_queries.jsonl).")
        return []

    records: list[QueryResult] = []
    raw_runs: dict = {}  # id -> raw AgentState, kept out of the dataclass (not JSON-serializable as-is)

    for q in all_queries:
        logger.info("Running %s: %r", q["id"], q["query"])
        result = QueryResult(
            id=q["id"],
            query=q["query"],
            lang=q["lang"],
            expected_route=q["expected_route"],
            expected_crop=q.get("expected_crop"),
        )
        start = time.monotonic()
        try:
            state = run_text_query(q["query"])
        except Exception as exc:
            result.run_error = f"{type(exc).__name__}: {exc}"
            logger.error("Query %s raised: %s", q["id"], result.run_error)
            records.append(result)
            continue
        result.latency_seconds = round(time.monotonic() - start, 3)

        raw_runs[q["id"]] = state
        route = state.get("route")
        if route is not None:
            result.actual_needs_rag = route.needs_rag
            result.actual_needs_weather = route.needs_weather
            result.actual_needs_price = route.needs_price
            result.actual_crop = route.crop
            result.route_source = route.source
            result.route_correct = _route_matches(
                q["expected_route"], route.needs_rag, route.needs_weather, route.needs_price
            )

        result.normalized_query = state.get("normalized_query", "")
        result.detected_script = state.get("detected_script", "")

        chunks = state.get("retrieved_chunks", []) or []
        result.retrieved_chunk_count = len(chunks)
        result.citations = state.get("citations", []) or []
        if "rag" in q["expected_route"]:
            result.retrieval_hit = result.retrieved_chunk_count > 0

        result.final_answer = state.get("final_answer", "") or ""
        result.answer_length_chars = len(result.final_answer)

        result.warnings = state.get("warnings", []) or []
        result.errors = state.get("errors", []) or []
        result.injection_flagged = any("injection" in w.lower() for w in result.warnings)

        result.proxy_faithfulness_overlap = _proxy_faithfulness(result.final_answer, chunks)

        records.append(result)

    ground_truths = {q["id"]: q.get("ground_truth", "") for q in all_queries}
    ragas_scores = _run_ragas(records, raw_runs, ground_truths)
    _save_results(records, ragas_scores)
    return records


def _save_results(records: list[QueryResult], ragas_scores: Optional[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "rag_eval_results.json"
    payload = {
        "ragas_available": _RAGAS_INSTALLED,
        "ragas_scores": ragas_scores,  # None if not computed -- never fabricated
        "results": [asdict(r) for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # CSV (flat, drop list/dict fields that don't flatten cleanly)
    csv_path = RESULTS_DIR / "rag_eval_results.csv"
    if records:
        flat_fields = [
            "id", "query", "lang", "expected_route", "expected_crop",
            "actual_needs_rag", "actual_needs_weather", "actual_needs_price",
            "actual_crop", "route_source", "route_correct",
            "detected_script", "retrieved_chunk_count", "retrieval_hit",
            "answer_length_chars", "injection_flagged",
            "proxy_faithfulness_overlap", "latency_seconds", "run_error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_fields)
            writer.writeheader()
            for r in records:
                row = asdict(r)
                writer.writerow({k: row.get(k) for k in flat_fields})

    _write_summary(records, ragas_scores)
    logger.info("Results written to %s", RESULTS_DIR)


def _write_summary(records: list[QueryResult], ragas_scores: Optional[dict]) -> None:
    total = len(records)
    failed_runs = [r for r in records if r.run_error]
    completed = [r for r in records if not r.run_error]

    route_scored = [r for r in completed if r.route_correct is not None]
    route_correct_n = sum(1 for r in route_scored if r.route_correct)

    rag_expected = [r for r in completed if r.retrieval_hit is not None]
    retrieval_hit_n = sum(1 for r in rag_expected if r.retrieval_hit)

    injection_probe = [r for r in completed if "ignore previous instructions" in r.query.lower()]

    overlaps = [r.proxy_faithfulness_overlap for r in completed if r.proxy_faithfulness_overlap is not None]
    avg_overlap = round(sum(overlaps) / len(overlaps), 4) if overlaps else None

    lines = [
        "# KrishokBot Text/RAG/Tool Evaluation — Summary",
        "",
        f"Total queries: {total} | Completed: {len(completed)} | Runtime errors: {len(failed_runs)}",
        "",
        "## Routing correctness (deterministic, always computed)",
        f"{route_correct_n}/{len(route_scored)} queries routed to the expected combination of "
        f"RAG/weather/price ({round(100*route_correct_n/len(route_scored), 1) if route_scored else 'N/A'}%).",
        "",
        "## Retrieval hit-rate (queries where expected_route includes 'rag')",
        f"{retrieval_hit_n}/{len(rag_expected)} returned at least one chunk "
        f"({round(100*retrieval_hit_n/len(rag_expected), 1) if rag_expected else 'N/A'}%).",
        "",
        "## Guardrail: prompt-injection probe (q29)",
    ]
    if injection_probe:
        p = injection_probe[0]
        lines.append(f"Flagged by injection heuristic: {p.injection_flagged}. Warnings: {p.warnings}")
    else:
        lines.append("Probe query not present in this run.")

    lines += [
        "",
        "## RAGAS metrics (faithfulness / answer_relevancy / context_precision / context_recall)",
    ]
    if ragas_scores:
        for k, v in ragas_scores.items():
            lines.append(f"- {k}: {v}")
    else:
        lines += [
            f"NOT COMPUTED. ragas package installed: {_RAGAS_INSTALLED}.",
            "Reason: either ragas/datasets is not installed, no rows had both a final "
            "answer and retrieved context, or the RAGAS LLM-judge call failed "
            "(see run log). Install with `pip install ragas datasets --break-system-packages` "
            "and ensure an LLM key is configured, then re-run.",
        ]

    lines += [
        "",
        "## Proxy faithfulness (fallback lexical overlap — NOT a RAGAS-equivalent metric)",
        f"Average token overlap between answer and retrieved chunks: {avg_overlap}"
        if avg_overlap is not None else "No comparable rows.",
        "This measures vocabulary grounding only, not semantic faithfulness. "
        "Treat as a weak sanity signal, not a substitute for RAGAS faithfulness.",
        "",
        "## Per-query results",
        "See rag_eval_results.json / rag_eval_results.csv for full detail.",
        "",
        "## Runtime errors",
    ]
    if failed_runs:
        for r in failed_runs:
            lines.append(f"- {r.id}: {r.run_error}")
    else:
        lines.append("None.")

    (RESULTS_DIR / "rag_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run KrishokBot text/RAG/tool evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries.")
    parser.add_argument(
        "--queries", type=str, default=None,
        help="Comma-separated query ids to run, e.g. q01,q14,q29",
    )
    args = parser.parse_args()

    ids = args.queries.split(",") if args.queries else None
    records = run(query_ids=ids, limit=args.limit)

    if not records:
        sys.exit(1)

    n_errors = sum(1 for r in records if r.run_error)
    print(f"\nDone. {len(records) - n_errors}/{len(records)} queries completed without a runtime error.")
    print(f"Results: {RESULTS_DIR}/rag_eval_results.json, rag_eval_results.csv, rag_eval_summary.md")


if __name__ == "__main__":
    main()
