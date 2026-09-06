"""server.py — FastAPI wrapper around the existing agent/graph.py pipeline.

Replaces the Streamlit UI with a real HTTP API so a proper frontend can be
built on top of it. Two endpoints:

    POST /api/chat      text query -> SSE stream of node-by-node trace events,
                         ending with the final synthesized answer
    POST /api/diagnose   image upload (+ optional text) -> SSE stream of
                         vision -> rag-enrichment -> synthesis steps

Both use Server-Sent Events so the frontend can render an "agent trace"
panel showing normalize -> route -> rag/weather/price -> synthesis as it
actually happens, instead of faking a spinner.

Run locally:
    uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger("krishokbot.server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="KrishokBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_READY = False
BACKEND_ERROR = ""
try:
    from agent.graph import get_graph, extract_citations, vision_node, synthesis_node
    from agent.guardrails import filter_injected_chunks

    BACKEND_READY = True
except Exception as exc:  # noqa: BLE001
    BACKEND_ERROR = f"{type(exc).__name__}: {exc}"
    logger.error("Agent backend failed to import: %s", BACKEND_ERROR)


# --- serialization helpers ---------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of dataclasses / pydantic models / Chunk objects
    coming out of the graph state into plain JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):  # pydantic BaseModel (RouteDecision)
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    # Chunk objects from rag.retriever: expose only what the UI needs, never
    # raw chunk text (mirrors extract_citations' rationale in agent/graph.py)
    if hasattr(value, "source") and hasattr(value, "page"):
        return {"source": getattr(value, "source", None), "page": getattr(value, "page", None)}
    return str(value)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


NODE_LABELS = {
    "normalize": "Normalizing input (script detection)",
    "router": "Routing query (RAG / weather / price)",
    "rag": "Retrieving from agriculture knowledge base",
    "weather": "Fetching live weather",
    "price": "Fetching DAM market prices",
    "vision": "Running crop-disease detector (YOLO)",
    "synthesis": "Generating grounded answer",
}


# --- routes -------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ready": BACKEND_READY, "error": BACKEND_ERROR or None}


@app.post("/api/chat")
async def chat(message: str = Form(...), location: str = Form(default="")):
    def stream():
        if not BACKEND_READY:
            yield _sse("error", {"message": f"Backend not ready: {BACKEND_ERROR}"})
            return

        graph = get_graph()
        initial = {
            "original_input": message,
            "input_type": "text",
            "user_location": location or None,
            "warnings": [],
            "errors": [],
        }
        final_state: dict = {}
        try:
            for update in graph.stream(initial, stream_mode="updates"):
                for node_name, node_output in update.items():
                    final_state.update(node_output or {})
                    yield _sse(
                        "step",
                        {
                            "node": node_name,
                            "label": NODE_LABELS.get(node_name, node_name),
                            "output": _to_jsonable(node_output),
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield _sse("error", {"message": str(exc)})
            return

        yield _sse(
            "final",
            {
                "answer": final_state.get("final_answer", ""),
                "citations": _to_jsonable(final_state.get("citations", [])),
                "weather_result": _to_jsonable(final_state.get("weather_result")),
                "price_result": _to_jsonable(final_state.get("price_result")),
                "market_result": _to_jsonable(final_state.get("market_result")),
                "warnings": final_state.get("warnings", []),
            },
        )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/diagnose")
async def diagnose(image: UploadFile = File(...), message: str = Form(default="")):
    # Read the upload's bytes now, in the async request handler. The
    # StreamingResponse generator below runs later in a worker thread, by
    # which point Starlette has already closed the UploadFile's spooled
    # file — reading it lazily inside the generator raises
    # "I/O operation on closed file".
    image_bytes = await image.read()
    filename = image.filename or "upload.jpg"

    def stream():
        if not BACKEND_READY:
            yield _sse("error", {"message": f"Backend not ready: {BACKEND_ERROR}"})
            return

        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        state: dict = {
            "original_input": message,
            "input_type": "image",
            "image_path": tmp_path,
            "normalized_query": message or "এই ছবিতে কী সমস্যা আছে ব্যাখ্যা করো",
            "warnings": [],
            "errors": [],
        }
        try:
            vision_out = vision_node(state)
            state.update(vision_out)
            yield _sse(
                "step",
                {"node": "vision", "label": NODE_LABELS["vision"], "output": _to_jsonable(vision_out)},
            )

            rag_query = (state.get("vision_result") or {}).get("rag_query")
            if rag_query:
                from rag.retriever import get_retriever

                chunks = filter_injected_chunks(get_retriever().retrieve(rag_query, top_k=8))
                citations = extract_citations(chunks)
                state.update({"retrieved_chunks": chunks, "citations": citations})
                yield _sse(
                    "step",
                    {
                        "node": "rag",
                        "label": "Enriching diagnosis with agriculture docs",
                        "output": {"citations": citations},
                    },
                )

            synth_out = synthesis_node(state)
            state.update(synth_out)
            yield _sse(
                "step",
                {"node": "synthesis", "label": NODE_LABELS["synthesis"], "output": _to_jsonable(synth_out)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("diagnose stream failed")
            yield _sse("error", {"message": str(exc)})
            return
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        yield _sse(
            "final",
            {
                "answer": state.get("final_answer", ""),
                "vision_result": _to_jsonable(state.get("vision_result")),
                "citations": _to_jsonable(state.get("citations", [])),
                "warnings": state.get("warnings", []),
            },
        )

    return StreamingResponse(stream(), media_type="text/event-stream")
