import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.append(str(Path(__file__).resolve().parent.parent))
from styles import inject  # noqa: E402

inject()

st.markdown("<div class='kb-page'>", unsafe_allow_html=True)

st.markdown(
    "<div class='kb-hero'>"
    "<h1>KrishokBot — Agentic Bengali Agriculture Advisory Assistant</h1>"
    "<p>An agentic RAG system combining hybrid document retrieval, live tool-calling, and a "
    "self-trained computer-vision model to advise Bangladeshi farmers on rice, potato, and "
    "tomato cultivation in Bengali, English, or Banglish.</p>"
    "</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("<div class='kb-label'>How it works</div>", unsafe_allow_html=True)
    st.markdown(
        """
Every text query passes through an **input normalizer** that detects script and
transliterates Banglish to Bangla before retrieval, since embedding similarity
otherwise fails against a Bangla-script corpus. A **router node** then decides
whether the query needs RAG, a live tool call, or both. **RAG** runs hybrid
retrieval (BM25 + vector search, fused with Reciprocal Rank Fusion) over a
Bengali-only ChromaDB corpus of BARI/DAE agricultural leaflets. A **weather
tool** calls the OpenWeather current-observation endpoint, and a **market-price
tool** parses the latest live DAM PDF report for the requested division. Image
queries instead run directly through the **YOLOv8m vision node**, self-trained
on rice/potato/tomato disease classes. A single **synthesis node** — the only
node that speaks to the user — combines whatever upstream context exists into
one grounded Bengali answer.
        """
    )

    st.markdown("<div class='kb-label'>Architecture</div>", unsafe_allow_html=True)
    mermaid_code = """
    <div class="mermaid">
    flowchart TD
        U[Text or image input] --> R0{Text or image?}
        R0 -->|text| N[Normalizer\\nBanglish -> Bangla script]
        N --> RT[Router\\nRAG / tool / both]
        R0 -->|image| Y[YOLOv8m Vision Node\\nmAP50 94.6%]
        RT -->|informational| RAG[RAG Node\\nChromaDB, hybrid BM25+vector RRF]
        RT -->|live data| T[Tool Node\\nweather API / DAM price parser]
        RAG --> S[Synthesis Node\\nGroq / Gemini]
        T --> S
        Y --> S
        S --> OUT[Bangla answer + sources]
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.0/mermaid.min.js"></script>
    <script>
        mermaid.initialize({startOnLoad: true, theme: "neutral", flowchart: {curve: "basis"}});
    </script>
    """
    components.html(mermaid_code, height=630, scrolling=False)

with col2:
    st.markdown("<div class='kb-label'>Tech stack</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <table class="kb-metric-table">
        <tr><th>Layer</th><th>Technology</th></tr>
        <tr><td>Orchestration</td><td>LangGraph</td></tr>
        <tr><td>Vector store</td><td>ChromaDB</td></tr>
        <tr><td>Retrieval</td><td>Hybrid BM25 + vector (RRF)</td></tr>
        <tr><td>Embeddings</td><td>Bengali sentence-transformers</td></tr>
        <tr><td>Vision model</td><td>YOLOv8m (Ultralytics)</td></tr>
        <tr><td>LLM</td><td>Groq (Llama 3.3 70B), Gemini fallback</td></tr>
        <tr><td>UI</td><td>Streamlit</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='kb-label' style='margin-top:8px;'>Vision model — measured evaluation</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='kb-card'>"
    "Trained on a merged Dhan-Shomadhan (genuine Bangladeshi field data, rice) + BCDD "
    "(potato/tomato) dataset, 11 classes, evaluated on a held-out test split."
    "<table class='kb-metric-table'>"
    "<tr><th>Metric</th><th>Overall</th></tr>"
    "<tr><td>mAP50</td><td>94.6%</td></tr>"
    "<tr><td>mAP50-95</td><td>94.6%</td></tr>"
    "<tr><td>Precision</td><td>87.1%</td></tr>"
    "<tr><td>Recall</td><td>92.1%</td></tr>"
    "</table>"
    "<table class='kb-metric-table' style='margin-top:10px;'>"
    "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>AP50</th></tr>"
    "<tr><td>rice_brown_spot</td><td>87.2%</td><td>76.0%</td><td>85.4%</td></tr>"
    "<tr><td>rice_leaf_scald</td><td>66.0%</td><td>80.0%</td><td>80.3%</td></tr>"
    "<tr><td>rice_blast</td><td>71.5%</td><td>90.0%</td><td>89.7%</td></tr>"
    "<tr><td>rice_tungro</td><td>91.6%</td><td>90.4%</td><td>95.8%</td></tr>"
    "<tr><td>rice_bacterial_blight</td><td>100.0%</td><td>98.0%</td><td>99.5%</td></tr>"
    "<tr><td>potato_early_blight</td><td>97.1%</td><td>100.0%</td><td>99.5%</td></tr>"
    "<tr><td>potato_late_blight</td><td>81.1%</td><td>97.9%</td><td>97.1%</td></tr>"
    "<tr><td>potato_healthy</td><td>99.0%</td><td>95.8%</td><td>99.1%</td></tr>"
    "<tr><td>tomato_early_blight</td><td>84.9%</td><td>97.9%</td><td>98.8%</td></tr>"
    "<tr><td>tomato_late_blight</td><td>84.1%</td><td>93.8%</td><td>95.8%</td></tr>"
    "<tr><td>tomato_healthy</td><td>95.7%</td><td>93.7%</td><td>99.1%</td></tr>"
    "</table>"
    "<p class='kb-caption' style='margin-top:8px;'>rice_leaf_scald shows the weakest precision (66.0%) "
    "of any class — disclosed rather than hidden behind the aggregate score.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='kb-label'>Evaluation methodology</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='kb-card'>"
    "Text pipeline quality is measured separately with a RAGAS harness (faithfulness, answer "
    "relevancy, context precision/recall) over a hand-written Bangla/English/Banglish gold "
    "query set, run as a background evaluation — not shown to the farmer-facing UI. Vision "
    "quality is measured with standard Ultralytics YOLO evaluation (mAP, precision, recall) "
    "on a held-out test split, per-class rather than aggregate-only."
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "This is a portfolio/research project, not a substitute for a licensed agricultural "
    "expert. See the Limitations page for the full disclosed scope."
)

st.markdown("</div>", unsafe_allow_html=True)
