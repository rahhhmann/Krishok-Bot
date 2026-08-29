"""tests/conftest.py

Shared test setup:
1. Puts the project root on sys.path so `import agent.x` / `import rag.x` /
   `import vision.x` work when pytest is run from the project root
   (``pytest`` or ``python -m pytest``).
2. Sets harmless dummy env vars *before* agent/config.py is first imported
   anywhere, so config.validate_config()/module-level reads don't depend on
   a real .env being present. Tests that care about specific config values
   set them explicitly via monkeypatch in the test itself.
3. Stubs the `ultralytics` package if it isn't installed, so
   vision/detector.py (which does `from ultralytics import YOLO` at import
   time) can be imported and unit-tested without pulling in torch/ultralytics
   just to run tests. Real detector *behavior* (predict()) is still tested
   with mocks -- this only avoids a hard import-time dependency for tests
   that don't need the real model.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dummy config values -- never real secrets. Only set if not already present
# so a developer's real .env (if pytest is run with it loaded) still wins.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-groq-key")
os.environ.setdefault("OPENWEATHER_API_KEY", "test-dummy-owm-key")
os.environ.setdefault("KRISHOKBOT_LLM_PROVIDER", "groq")

if "ultralytics" not in sys.modules:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        stub = types.ModuleType("ultralytics")
        stub.YOLO = object  # replaced per-test via monkeypatch/mock.patch
        sys.modules["ultralytics"] = stub

# rag/retriever.py does `import chromadb` and
# `from sentence_transformers import SentenceTransformer` at module level.
# Both are heavy (chromadb pulls in sqlite/onnx deps, sentence-transformers
# pulls in torch) and unit tests always patch these out anyway (see
# tests/test_retriever.py) -- stub them so the import itself doesn't
# require the real packages to be installed just to run tests.
if "chromadb" not in sys.modules:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        stub = types.ModuleType("chromadb")
        stub.PersistentClient = object  # replaced per-test via mock.patch
        sys.modules["chromadb"] = stub

if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = object  # replaced per-test via mock.patch
        sys.modules["sentence_transformers"] = stub

if "rank_bm25" not in sys.modules:
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        stub = types.ModuleType("rank_bm25")
        stub.BM25Okapi = object  # replaced per-test via mock.patch
        sys.modules["rank_bm25"] = stub
