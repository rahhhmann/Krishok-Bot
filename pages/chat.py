import html
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from styles import inject, confidence_badge  # noqa: E402
from vision.class_mapping import get_class_info  # noqa: E402
from streamlit_ui import _market_category  # noqa: E402

inject()

AVATAR_ASSISTANT_PATH = ROOT / "assets" / "farmer_icon.png"
AVATAR_USER_PATH = ROOT / "assets" / "human.png"
AVATAR_ASSISTANT = str(AVATAR_ASSISTANT_PATH) if AVATAR_ASSISTANT_PATH.exists() else None
AVATAR_USER = str(AVATAR_USER_PATH) if AVATAR_USER_PATH.exists() else None

# ---------------------------------------------------------------- backend ---
# Direct import — no adapter layer. If agent/vision aren't importable in this
# environment, the page runs in a clearly-labeled demo mode instead of
# pretending to produce real model output.
BACKEND_READY = False
BACKEND_ERROR = None
try:
    from agent.graph import run_text_query, run_image_query
    from agent.router import _is_greeting, _is_non_rag_input
    BACKEND_READY = True
except Exception as exc:  # noqa: BLE001
    BACKEND_ERROR = f"{type(exc).__name__}: {exc}"
    def _is_greeting(_query):
        return False
    def _is_non_rag_input(_query):
        return False


def _extract_answer(result) -> tuple[str, list, dict | None, object | None, dict]:
    """Normalize run_text_query / run_image_query's return value.

    Accepts a plain string, a dict, or an object exposing the common
    attribute names, without assuming one fixed schema.
    """
    if isinstance(result, str):
        return result, [], None, None, {}
    if isinstance(result, dict):
        text = result.get("final_answer") or result.get("answer") or result.get("text") or ""
        sources = result.get("sources") or result.get("citations") or []
        detection = result.get("vision_result") or result.get("detection")
        market = result.get("market_result") or result.get("market")
        price_results = result.get("price_results") or {}
        return text, sources, detection, market, price_results
    text = getattr(result, "final_answer", None) or getattr(result, "answer", "") or ""
    sources = getattr(result, "sources", []) or []
    detection = getattr(result, "vision_result", None)
    market = getattr(result, "market_result", None) or getattr(result, "market", None)
    price_results = getattr(result, "price_results", {}) or {}
    return text, sources, detection, market, price_results


def _source_title(src) -> str | None:
    """Return a displayable title for one source entry, or None to skip it silently."""
    if not src:
        return None
    if isinstance(src, dict):
        title = src.get("title") or src.get("name") or src.get("source") or src.get("page")
        return str(title) if title else None
    text = str(src).strip()
    return text if text and text.lower() != "none" else None


def _group_sources(sources) -> list[str]:
    """Collapse repeated citation rows into one source card per document."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for src in sources or []:
        if isinstance(src, dict):
            name = src.get("source") or src.get("title") or src.get("name")
            page = src.get("page")
        else:
            name = str(src).strip()
            page = None
        if not name:
            continue
        name = str(name)
        page_text = str(page) if page not in (None, "") else ""
        if page_text and page_text not in grouped[name]:
            grouped[name].append(page_text)
        elif name not in grouped:
            grouped[name] = []
    cards = []
    for name, pages in grouped.items():
        suffix = f" · পৃষ্ঠা {', '.join(pages)}" if pages else ""
        cards.append(f"{name}{suffix}")
    return cards


def _render_sources(sources) -> None:
    """Render one clean, deduplicated citation section below the answer."""
    titles = _group_sources(sources)
    if not titles:
        return
    st.markdown("**তথ্যসূত্র / Sources**")
    for title in titles:
        st.markdown(
            f"<div class='kb-source-card'>{html.escape(title)}</div>",
            unsafe_allow_html=True,
        )


def _render_market_cards(market) -> None:
    """Render the structured DAM report below the concise assistant message."""
    if not market:
        return
    available = getattr(market, "available", None)
    if isinstance(market, dict):
        available = market.get("available", True)
        items = market.get("items") or []
        division = market.get("division", "")
        report_date = market.get("date", "")
        omitted = market.get("omitted_count", 0)
    else:
        items = getattr(market, "items", []) or []
        division = getattr(market, "division", "")
        report_date = getattr(market, "date", "")
        omitted = getattr(market, "omitted_count", 0)
    if not available or not items:
        return

    st.markdown(
        "<div class='kb-market-head'><strong>Market report / বাজারদর</strong></div>",
        unsafe_allow_html=True,
    )

    grouped = defaultdict(list)
    for item in items:
        if isinstance(item, dict):
            product = item.get("product", "")
            price_range = item.get("price_range", "")
            unit = item.get("unit", "")
            category = _market_category(type("MarketItem", (), {"category": item.get("category", ""), "product": product})())
        else:
            product = getattr(item, "product", "")
            price_range = getattr(item, "price_range", "")
            unit = getattr(item, "unit", "")
            category = _market_category(item)
        grouped[str(category)].append((str(product), str(price_range), str(unit)))

    for category, rows in grouped.items():
        with st.expander(f"{category} · {len(rows)} items", expanded=True):
            for product, price_range, unit in rows:
                st.markdown(
                    "<div style='display:flex;justify-content:space-between;gap:16px;"
                    "padding:9px 0;border-bottom:1px solid var(--kb-border);'>"
                    f"<span>{html.escape(product)}</span>"
                    f"<strong style='white-space:nowrap;color:var(--kb-accent-dark);'>"
                                        f"{html.escape(price_range)} {html.escape(unit)}</strong></div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"<div class='kb-market-meta'>রিপোর্টের তারিখ: {html.escape(str(report_date))}<br>"
        "তথ্যসূত্র: DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)</div>",
        unsafe_allow_html=True,
    )
    

def _render_price_cards(price_results) -> None:
    for price in (price_results or {}).values():
        if isinstance(price, dict):
            available = price.get("available", False)
            crop = price.get("crop", "")
            price_range = price.get("price_range") or price.get("price_bdt_per_kg") or ""
            unit = price.get("unit") or "প্রতি কেজি"
            date = price.get("date") or ""
            note = price.get("division_note") or ""
            error = price.get("error") or ""
        else:
            available = getattr(price, "available", False)
            crop = getattr(price, "crop", "")
            price_range = getattr(price, "price_range", "") or getattr(price, "price_bdt_per_kg", "") or ""
            unit = getattr(price, "unit", "") or "প্রতি কেজি"
            date = getattr(price, "date", "") or ""
            note = getattr(price, "division_note", "") or ""
            error = getattr(price, "error", "") or ""
        if not available:
            st.markdown(f"<div class='kb-card kb-note'>{html.escape(str(crop))}: বাজারদর পাওয়া যায়নি — {html.escape(str(error))}</div>", unsafe_allow_html=True)
            continue
        st.markdown(
            f"<div class='kb-card'><h3>DAM market price · {html.escape(str(crop))}</h3>"
            f"<div class='kb-price'>{html.escape(str(price_range))}</div>"
            f"<div class='kb-meta'>একক: {html.escape(str(unit))} · রিপোর্টের তারিখ: {html.escape(str(date))}<br>"
            f"বিভাগ: {html.escape(str(note))}<br>উৎস: DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)</div></div>",
            unsafe_allow_html=True,
        )


def _render_detection(detection) -> None:
    if not detection:
        return
    raw_label = detection.get("class_name") or detection.get("class") or detection.get("label", "")
    conf = float(detection.get("confidence", 0))
    info = get_class_info(raw_label)
    st.markdown(confidence_badge(conf), unsafe_allow_html=True)
    st.caption(f"শনাক্ত: {info.crop_bn} — {info.bn_name}")


def _run_query(query: str, image_bytes: bytes | None):
    """Direct call into the agent graph with safe temporary image-file handling."""
    if not image_bytes and _is_non_rag_input(query):
        if _is_greeting(query):
            reply = "ধন্যবাদ! আপনার কোনো কৃষি-সংক্রান্ত প্রশ্ন থাকলে জানাবেন। শুভ দিন!"
        else:
            reply = "দুঃখিত, আপনার কথাটি বুঝতে পারিনি। ধান, আলু, টমেটো, আবহাওয়া বা বাজারদর সম্পর্কে প্রশ্ন করুন।"
        return reply, [], None, None, {}, False
    if not BACKEND_READY:
        demo_text = (
            "ডেমো মোড: এই পরিবেশে এজেন্ট ব্যাকএন্ড (agent/graph.py) যুক্ত নেই, "
            "তাই এটি একটি নমুনা উত্তর মাত্র। প্রকৃত ব্যাকএন্ড যুক্ত হলে RAG থেকে "
            "প্রকৃত তথ্যসূত্র এবং মডেলের প্রকৃত জবাব এখানে দেখানো হবে।"
        )
        demo_detection = {"class_name": "tomato_late_blight", "confidence": 0.71} if image_bytes else None
        return demo_text, [], demo_detection, None, {}, True

    if image_bytes:
        suffix = ".jpg"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(image_bytes)
                temp_path = handle.name
            result = run_image_query(temp_path, accompanying_text=query)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    else:
        result = run_text_query(query)

    text, sources, detection, market, price_results = _extract_answer(result)
    return text, sources, detection, market, price_results, False


# ---------------------------------------------------------------- session ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div class="kb-brand">
            <div class="kb-brand-logo">🌾</div>
            <div>
                <div class="kb-brand-title">KrishokBot</div>
                <div class="kb-brand-sub">কৃষকের ডিজিটাল সহকারী</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("নতুন চ্যাট", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<div class='kb-caption' style='margin:14px 0 6px 0;'>সহায়তা পাওয়া যায়</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <span class="kb-chip">ধান</span>
        <span class="kb-chip">আলু</span>
        <span class="kb-chip">টমেটো</span>
        <span class="kb-chip">আবহাওয়া</span>
        <span class="kb-chip">বাজারদর</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if not BACKEND_READY:
        with st.expander("বিস্তারিত"):
            st.caption(BACKEND_ERROR or "agent প্যাকেজ পাওয়া যায়নি।")

    # Manually render page nav here (below the branding block) and hide the
    # default auto-injected sidebar nav, which Streamlit otherwise places
    # above all custom sidebar content.
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none;}</style>",
        unsafe_allow_html=True,
    )
    for _label, _candidates in (
        ("Chat", ["chat.py", "pages/chat.py", "pages/1_Chat.py"]),
        ("About", ["about.py", "pages/about.py", "pages/2_About.py"]),
        ("Limitations", ["limitations.py", "pages/limitations.py", "pages/3_Limitations.py"]),
    ):
        for _path in _candidates:
            try:
                st.page_link(_path, label=_label)
                break
            except Exception:  # noqa: BLE001
                continue

    st.divider()
    st.caption("এই বট চূড়ান্ত কৃষি বিশেষজ্ঞের পরামর্শ নয়। নিশ্চিত না হলে স্থানীয় কৃষি সম্প্রসারণ অফিসারের সাথে যোগাযোগ করুন।")

# ---------------------------------------------------------------- header ----
st.markdown(
    "<div class='kb-block' style='padding-top:6px;'>"
    "<span style='font-weight:700;font-size:17px;color:var(--kb-accent-dark);'>🌾 KrishokBot</span>"
    "<span class='kb-caption'> · ধান · আলু · টমেটো পরামর্শ সহায়ক</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- history ---
if not st.session_state.messages:
    st.markdown(
        """
        <div class="kb-empty-state">
            <h2>আজ কীভাবে সাহায্য করতে পারি?</h2>
            <p>ধান, আলু বা টমেটো নিয়ে বাংলা, English বা Banglish-এ প্রশ্ন করুন —
            অথবা নিচের প্রম্পট বাক্সে ফসলের পাতার ছবি আপলোড করে রোগ শনাক্ত করুন।</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    suggestions = [
        "ধানের ব্লাস্ট রোগের প্রতিকার কী?",
        "আজ বরিশালে আবহাওয়া কেমন ?",
        "আজকের আলুর বাজারদর কত ?",
    ]
    for col, text in zip((c1, c2, c3), suggestions):
        if col.button(text, use_container_width=True):
            st.session_state.pending_prompt = text
            st.rerun()

for msg in st.session_state.messages:
    avatar = AVATAR_ASSISTANT if msg["role"] == "assistant" else AVATAR_USER
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image"):
            st.image(msg["image"], width=240)
        st.markdown(msg["content"])
        detection = msg.get("detection")
        if detection:
            _render_detection(detection)
        if not msg.get("is_non_rag_input", False) and not _is_non_rag_input(msg.get("content", "")):
            _render_sources(msg.get("sources", []))
        _render_market_cards(msg.get("market"))
        if msg.get("is_demo"):
            st.markdown("<div class='kb-warn-box'>এটি একটি ডেমো উত্তর — প্রকৃত মডেল নয়।</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- input -----
chat_value = st.chat_input(
    "আপনার প্রশ্ন লিখুন... (বাংলা / English / Banglish)",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
    key="main_chat_input",
)

user_text, user_image_bytes, user_image_display = None, None, None

if st.session_state.pending_prompt:
    user_text = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif chat_value:
    user_text = getattr(chat_value, "text", "") or ""
    files = getattr(chat_value, "files", None) or []
    if files:
        user_image_display = files[0]
        user_image_bytes = files[0].getvalue()
    if not user_text and user_image_bytes:
        user_text = "এই ছবিতে ফসলের কী সমস্যা দেখা যাচ্ছে?"

if user_text or user_image_bytes:
    st.session_state.messages.append({"role": "user", "content": user_text, "image": user_image_display})

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("উত্তর তৈরি হচ্ছে..."):
            text, sources, detection, market, price_results, is_demo = _run_query(user_text, user_image_bytes)
        st.markdown(text or "(কোনো উত্তর তৈরি হয়নি)")
        if detection:
            _render_detection(detection)
        is_non_rag_input = _is_non_rag_input(user_text) if not user_image_bytes else False
        if not is_non_rag_input:
            _render_sources(sources)
        _render_market_cards(market)
        if is_demo:
            st.markdown("<div class='kb-warn-box'>এটি একটি ডেমো উত্তর — প্রকৃত মডেল নয়।</div>", unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "sources": sources,
            "detection": detection,
            "market": market,
            "price_results": price_results,
            "is_non_rag_input": is_non_rag_input,
            "is_demo": is_demo,
        }
    )
    st.rerun()