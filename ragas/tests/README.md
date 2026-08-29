# KrishokBot unit tests

121 tests, all pure/mocked — no network calls, no real LLM calls, no real
ChromaDB/YOLO/embedding model loaded. Runtime: well under 1 second.

## Run

```powershell
pip install pytest --break-system-packages
python -m pytest
```

From the project root (`E:\krishokbot`). `pytest.ini` already points
`testpaths` at `tests/`.

Run one file: `python -m pytest tests/test_router.py -v`
Run one test: `python -m pytest tests/test_router.py::test_route_valid_llm_json_is_used_directly -v`

## Coverage by file

| File | Covers | Real deps needed to even import? |
|---|---|---|
| `test_normalizer.py` | script detection, LLM transliteration fallback logic | none (LLM mocked) |
| `test_router.py` | LLM routing, JSON parse failure → keyword fallback, crop/weather/price keyword detection | none (LLM mocked) |
| `test_guardrails.py` | prompt-injection detection, garbled-OCR stripping, uncertainty-language enforcement | none |
| `test_tools.py` | text/digit/crop/division/unit normalization, `get_weather` (HTTP mocked) | `requests`, `beautifulsoup4` |
| `test_detector.py` | confidence-tier thresholds, detection parsing/sorting, pending-review logging | none — `ultralytics` is stubbed by `conftest.py` if not installed |
| `test_retriever.py` | synonym expansion, vector similarity filtering, **RRF hybrid fusion math** (the documented blast-disease under-ranking bug) | none — `chromadb`/`sentence_transformers`/`rank_bm25` are stubbed by `conftest.py` if not installed |
| `test_graph.py` | every LangGraph node function, `route_selector` fan-out, `extract_citations` dedup/no-raw-text-leak, `_build_context_block` formatting | `langgraph` (real, lightweight — not stubbed since `build_graph()` actually compiles a real graph) |

`conftest.py` stubs `ultralytics`, `chromadb`, `sentence_transformers`, and
`rank_bm25` with empty placeholder modules **only if they aren't already
installed** — so on your actual dev machine (where all of these are real
project dependencies) the tests run against the real import machinery,
they just still mock the actual model/DB *calls* within each test via
`unittest.mock.patch`.

## What's intentionally NOT covered

- `agent/tools.py`'s DAM PDF-scraping internals (`_get_division_pdf_price`,
  `_extract_pdf_tables`, OCR correction tables, etc.) — these need a
  realistic saved DAM HTML+PDF fixture to test meaningfully. Mocking them
  down to unit-testable pieces would mostly test the mocks, not the
  scraping logic. If this becomes a priority, the right next step is
  saving one real DAM PDF as a fixture and writing an integration-style
  test against it, not more mocking.
- Real end-to-end `run_text_query()` / `graph.invoke()` — that's what
  `eval/run_eval.py`'s real corpus run against the actual ChromaDB
  collection + live LLM already does, and duplicating it here with
  mocks would just test whether the mocks agree with each other.
- RAG retrieval *quality* (is the right chunk actually top-ranked for a
  real query) — that's an eval question, not a unit-test question;
  `test_retriever.py` only proves the RRF fusion arithmetic is correct.

## RAGAS

`eval/run_eval.py`'s `_run_ragas()` was patched (see project root diff)
to build its judge LLM from **your own** `agent/config.py` provider
(Groq/Gemini) instead of silently defaulting to OpenAI. To actually run
it:

```powershell
pip install ragas==0.1.21 "datasets<3.0" langchain_groq==0.1.10 --break-system-packages
python -m eval.run_eval
```

That exact version pin was verified in an isolated environment — newer
`ragas` (0.4.x) has a broken `langchain_community` import chain as of
this writing; don't `pip install ragas` unpinned.

`context_precision`/`context_recall` are only computed for queries where
`test_queries.jsonl` has a non-empty `"ground_truth"` field (optional,
new). Without any ground truth, you'll get `faithfulness` and
`answer_relevancy` only — that's expected, not a bug.
