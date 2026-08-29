import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from styles import inject  # noqa: E402

inject()

st.markdown("<div class='kb-page'>", unsafe_allow_html=True)

st.markdown(
    "<div class='kb-hero'>"
    "<h1>Known Limitations</h1>"
    "<p>Disclosed explicitly, in the same spirit as the author's earlier BD License Plate "
    "Detector project — what the system does not cover is stated directly rather than "
    "implied by omission.</p>"
    "</div>",
    unsafe_allow_html=True,
)

sections = [
    ("Crop and disease coverage",
     "Only rice, paddy, potato, and tomato are covered — 11 disease/health classes total. "
     "No other crop or disease is claimed."),
    ("Dataset composition",
     "Rice images are genuine Bangladeshi field data (Dhan-Shomadhan). Potato and tomato "
     "images are lab-condition PlantVillage-origin (BCDD), not verified against real "
     "Bangladeshi field conditions. Many bounding boxes are full-frame placeholders rather "
     "than precise disease localization, since the source datasets lack fine-grained "
     "annotation."),
    ("Vision confidence is not a diagnosis",
     "YOLO detections are shown with a confidence tier (green \u226585%, amber 60-85%, "
     "red <60%) and are a prediction, not a certified diagnosis. rice_leaf_scald in "
     "particular has the weakest measured precision (66.0%) of any class."),
    ("RAG corpus is Bengali-only",
     "The document corpus (BARI/DAE leaflets) is intentionally Bengali-only. User input in "
     "Banglish or English is still supported via the normalization node, but answer quality "
     "depends entirely on the ingested Bengali source documents, not a broader English corpus."),
    ("DAM market-price text extraction",
     "Some DAM PDF reports use legacy Bengali font mappings that corrupt raw text extraction "
     "even when the page looks correct visually. Only high-confidence corrections are applied; "
     "ambiguous product names are preserved rather than guessed, and a report can therefore "
     "return fewer confirmed items than its total serial-row count."),
    ("Weather is a current observation only",
     "Weather responses reflect the current observation at query time, not an all-day or "
     "multi-day forecast. There is no forecast endpoint in the current implementation."),
    ("Division-scoped pricing",
     "A price query without a division uses the configured default (Dhaka). The tool does "
     "not silently combine prices across multiple divisions."),
    ("Regional dialect input",
     "Heavy regional Bangla dialect input has reduced normalization accuracy relative to "
     "standard Bangla or common Banglish, consistent with findings from the author's "
     "separate dialect-NLP research."),
]

for title, body in sections:
    st.markdown(f"<div class='kb-label'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kb-limit-item'>{body}</div>", unsafe_allow_html=True)

st.info(
    "This assistant does not replace a licensed agricultural expert or a local Department "
    "of Agricultural Extension (DAE) officer. When uncertain, consult one directly."
)

st.markdown("</div>", unsafe_allow_html=True)
