# KrishokBot — new UI setup

Replaces `app.py` (Streamlit) with a real API + a standalone web frontend.

## 1. Place the files

```
krishokbot_project/
├── server.py              <- put here (project root, next to app.py)
├── requirements-server.txt
└── frontend/
    └── index.html          <- new folder
```

## 2. Install

```bash
pip install -r requirements.txt          # existing deps (unchanged)
pip install -r requirements-server.txt   # fastapi, uvicorn, python-multipart
```

Your `.env` stays exactly where it already is (project root) — `server.py`
imports `agent.graph` the same way `pages/chat.py` did, so it picks up the
same environment.

## 3. Run (two terminals)

```bash
# terminal 1 — backend
uvicorn server:app --reload --port 8000

# terminal 2 — frontend (no build step, plain static file)
cd frontend && python -m http.server 5173
```

Open `http://localhost:5173`.

## What it does differently from Streamlit

- `server.py` calls the **same** `agent/graph.py` you already have
  (`get_graph()`, `vision_node`, `synthesis_node`) — no logic was rewritten.
- `/api/chat` streams `graph.stream(..., stream_mode="updates")` over SSE,
  so the frontend's right-hand **trace panel** shows each LangGraph node
  (normalize → router → rag/weather/price → synthesis) as it actually
  executes, not a fake spinner.
- `/api/diagnose` mirrors `run_image_query`'s vision → RAG-enrich →
  synthesis sequence, streaming the same way, and the frontend draws the
  YOLO bounding box on the uploaded image using the real `bbox`/`tier`
  from `vision/detector.py`.
- `about.py` / `limitations.py` content isn't wired in yet — it's plain
  content, so paste it into a modal or footer in `index.html` if you want
  it in the new UI too.

## Recording the demo video

1. Ask a Bangla question (e.g. "ধানের ব্লাস্ট রোগের প্রতিকার কী") — the trace
   panel fills in live as the router/RAG/synthesis nodes fire.
2. Ask a weather or price question to show the `weather`/`price` branch.
3. Switch to the ছবি tab, drop a leaf photo, show the bbox + confidence
   tier drawn on the image.
4. Screen-record with OBS at 1080p, keep it 60–90s, compress to GIF for
   the README (or upload full video to YouTube unlisted and link it —
   GitHub READMEs don't embed video directly).
