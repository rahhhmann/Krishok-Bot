"""Shared presentation layer for KrishokBot.

The backend remains unchanged here: this module only manages session state,
visual styling, farmer-facing cards, and safe rendering of dynamic values.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from html import escape
from typing import Any

import streamlit as st


THEMES = {
    "light": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F5F9",
        "text": "#0F172A",
        "text_secondary": "#334155",
        "muted": "#64748B",
        "border": "#CBD5E1",
        "accent": "#2563EB",
        "accent_hover": "#1D4ED8",
        "accent_soft": "rgba(37, 99, 235, 0.10)",
        "success": "#059669",
        "success_soft": "#ECFDF5",
        "danger": "#DC2626",
        "danger_soft": "#FEF2F2",
        "shadow": "0 1px 3px rgba(15, 23, 42, 0.08)",
        "shadow_hover": "0 10px 24px rgba(15, 23, 42, 0.11)",
    },
    "dark": {
        "bg": "#0B0F1A",
        "surface": "#161B2B",
        "surface_alt": "#1E2536",
        "text": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "muted": "#94A3B8",
        "border": "#2D3748",
        "accent": "#3B82F6",
        "accent_hover": "#60A5FA",
        "accent_soft": "rgba(59, 130, 246, 0.14)",
        "success": "#10B981",
        "success_soft": "rgba(16, 185, 129, 0.14)",
        "danger": "#EF4444",
        "danger_soft": "rgba(239, 68, 68, 0.14)",
        "shadow": "0 4px 12px rgba(0, 0, 0, 0.24)",
        "shadow_hover": "0 12px 30px rgba(0, 0, 0, 0.40)",
    },
}


def init_session() -> None:
    defaults = {
        "theme": "light",
        "user_name": "",
        "chat_history": [],
        "last_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_theme() -> None:
    theme = THEMES.get(st.session_state.get("theme", "light"), THEMES["light"])
    st.markdown(
        f"""
        <style>
        :root {{
            --kb-bg: {theme['bg']};
            --kb-surface: {theme['surface']};
            --kb-surface-alt: {theme['surface_alt']};
            --kb-text: {theme['text']};
            --kb-text-secondary: {theme['text_secondary']};
            --kb-muted: {theme['muted']};
            --kb-border: {theme['border']};
            --kb-accent: {theme['accent']};
            --kb-accent-hover: {theme['accent_hover']};
            --kb-accent-soft: {theme['accent_soft']};
            --kb-success: {theme['success']};
            --kb-success-soft: {theme['success_soft']};
            --kb-danger: {theme['danger']};
            --kb-danger-soft: {theme['danger_soft']};
            --kb-shadow: {theme['shadow']};
            --kb-shadow-hover: {theme['shadow_hover']};
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMainViewContainer"] {{
            background: var(--kb-bg) !important;
            color: var(--kb-text) !important;
        }}
        .stApp {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        [data-testid="stSidebar"] {{ background: var(--kb-surface) !important; border-right: 1px solid var(--kb-border); }}
        [data-testid="stSidebar"] * {{ color: var(--kb-text); }}
        [data-testid="stSidebar"] hr {{ border-color: var(--kb-border); }}

        .kb-shell {{ max-width: 1160px; margin: 0 auto; padding: 1.4rem .55rem 4rem; }}
        .kb-brand {{ display: flex; align-items: center; gap: .7rem; margin-bottom: 1.35rem; }}
        .kb-brand-mark {{ width: 34px; height: 34px; border-radius: 10px; background: var(--kb-accent); color: #fff; display: grid; place-items: center; font-weight: 800; font-size: 1.05rem; letter-spacing: -.04em; box-shadow: 0 4px 12px var(--kb-accent-soft); }}
        .kb-brand-name {{ color: var(--kb-text); font-size: 1.05rem; font-weight: 750; letter-spacing: -.02em; }}
        .kb-brand-sub {{ color: var(--kb-muted); font-size: .74rem; margin-top: .1rem; }}
        .kb-kicker {{ color: var(--kb-accent); font-size: .72rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .35rem; }}
        .kb-title {{ color: var(--kb-text); font-size: clamp(1.85rem, 4vw, 2.75rem); font-weight: 760; letter-spacing: -.045em; line-height: 1.12; margin: 0 0 .65rem; }}
        .kb-subtitle {{ color: var(--kb-muted); font-size: .98rem; line-height: 1.7; max-width: 760px; margin-bottom: 1.45rem; }}
        .kb-section-label {{ color: var(--kb-muted); font-size: .72rem; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; margin: 1.4rem 0 .5rem; }}
        .kb-card {{ background: var(--kb-surface); border: 1px solid var(--kb-border); border-radius: 14px; padding: 1.05rem 1.15rem; margin: .7rem 0; box-shadow: var(--kb-shadow); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }}
        .kb-card:hover {{ transform: translateY(-2px); border-color: var(--kb-accent); box-shadow: var(--kb-shadow-hover); }}
        .kb-card h3, .kb-card h4 {{ color: var(--kb-text); margin: 0 0 .42rem; letter-spacing: -.015em; }}
        .kb-meta {{ color: var(--kb-muted); font-size: .86rem; line-height: 1.65; }}
        .kb-price {{ color: var(--kb-accent); font-size: 1.35rem; font-weight: 780; letter-spacing: -.02em; margin: .25rem 0 .2rem; }}
        .kb-answer {{ color: var(--kb-text); line-height: 1.85; font-size: 1rem; }}
        .kb-chat-user {{ background: var(--kb-accent-soft); border: 1px solid color-mix(in srgb, var(--kb-accent) 28%, var(--kb-border)); border-radius: 16px 16px 5px 16px; padding: .82rem 1rem; margin: .65rem 0 .4rem auto; max-width: 86%; color: var(--kb-text); line-height: 1.65; }}
        .kb-chat-bot {{ background: var(--kb-surface); border: 1px solid var(--kb-border); border-radius: 16px 16px 16px 5px; padding: 1rem 1.08rem; margin: .35rem auto .8rem 0; max-width: 94%; color: var(--kb-text); box-shadow: var(--kb-shadow); }}
        .kb-note {{ color: var(--kb-muted); font-size: .84rem; line-height: 1.6; }}
        .kb-status {{ display: inline-flex; align-items: center; gap: .45rem; color: var(--kb-success); font-size: .78rem; font-weight: 650; }}
        .kb-status-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--kb-success); box-shadow: 0 0 0 4px var(--kb-success-soft); }}

        .stButton > button, button[data-testid^="stBaseButton"] {{ background: var(--kb-accent-soft) !important; color: var(--kb-accent) !important; border: 1px solid color-mix(in srgb, var(--kb-accent) 35%, var(--kb-border)) !important; border-radius: 8px !important; font-weight: 600 !important; transition: background .16s ease, color .16s ease, border-color .16s ease, transform .12s ease, box-shadow .16s ease; }}
        .stButton > button:hover, button[data-testid^="stBaseButton"]:hover {{ background: var(--kb-accent) !important; color: #fff !important; border-color: var(--kb-accent) !important; box-shadow: 0 4px 12px var(--kb-accent-soft) !important; }}
        .stButton > button:active, button[data-testid^="stBaseButton"]:active {{ transform: translateY(1px); }}
        .stButton > button:focus-visible, button[data-testid^="stBaseButton"]:focus-visible {{ outline: 2px solid var(--kb-accent) !important; outline-offset: 2px; box-shadow: none !important; }}
        div[data-baseweb="select"] > div {{ background: var(--kb-surface) !important; border: 1px solid var(--kb-border) !important; color: var(--kb-text) !important; border-radius: 8px !important; transition: border-color .16s ease, box-shadow .16s ease; }}
        div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {{ border-color: var(--kb-accent) !important; box-shadow: 0 0 0 3px var(--kb-accent-soft) !important; }}
        div[data-baseweb="popover"] {{ background: var(--kb-surface) !important; border: 1px solid var(--kb-border) !important; }}
        div[data-baseweb="popover"] li:hover {{ background: var(--kb-accent-soft) !important; }}

        [data-testid="stFileUploaderDropzone"] {{ background: var(--kb-surface-alt) !important; border: 1px dashed var(--kb-border) !important; border-radius: 12px !important; transition: border-color .18s ease, background .18s ease, box-shadow .18s ease; }}
        [data-testid="stFileUploaderDropzone"]:hover {{ background: var(--kb-accent-soft) !important; border-color: var(--kb-accent) !important; box-shadow: 0 0 0 3px var(--kb-accent-soft); }}
        [data-testid="stFileUploader"] button {{ background: transparent !important; color: var(--kb-text-secondary) !important; border: 1px solid var(--kb-border) !important; border-radius: 7px !important; }}
        [data-testid="stFileUploader"] button:hover {{ color: var(--kb-accent) !important; border-color: var(--kb-accent) !important; }}

        [data-testid="stExpander"] {{ background: var(--kb-surface) !important; border: 1px solid var(--kb-border) !important; border-radius: 11px !important; overflow: hidden; transition: border-color .16s ease, box-shadow .16s ease; }}
        [data-testid="stExpander"]:hover {{ border-color: var(--kb-accent) !important; box-shadow: var(--kb-shadow); }}
        [data-testid="stExpander"] summary:hover {{ background: var(--kb-accent-soft) !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 1.2rem; border-bottom: 1px solid var(--kb-border); }}
        .stTabs [data-baseweb="tab"] {{ color: var(--kb-muted) !important; font-weight: 650 !important; }}
        .stTabs [aria-selected="true"] {{ color: var(--kb-text) !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: var(--kb-accent) !important; height: 2px !important; }}
        [data-testid="stAlert"] {{ border-radius: 10px; }}
        ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
        ::-webkit-scrollbar-track {{ background: var(--kb-bg); }}
        ::-webkit-scrollbar-thumb {{ background: var(--kb-border); border-radius: 8px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--kb-accent); }}
        header {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        @media (max-width: 640px) {{
            .kb-shell {{ padding: .8rem .1rem 3rem; }}
            .kb-chat-user, .kb-chat-bot {{ max-width: 100%; }}
            .kb-title {{ font-size: 2rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    init_session()
    with st.sidebar:
        st.markdown(
            '<div class="kb-brand"><div class="kb-brand-mark">🌾</div><div><div class="kb-brand-name">KrishokBot</div><div class="kb-brand-sub">Agriculture assistant · কৃষি সহায়তা</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("New chat / নতুন চ্যাট", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_state = None
            st.rerun()
        st.divider()
        st.text_input("Name / নাম (optional)", key="user_name", placeholder="Your name / আপনার নাম")
        st.selectbox(
            "Theme / রঙের ধরন",
            ["light", "dark"],
            key="theme",
            format_func=lambda value: "Light / আলো" if value == "light" else "Dark / ডার্ক",
        )
        st.divider()
        st.caption("Session-only memory · No account or database memory · তথ্যসূত্রভিত্তিক উত্তর")


def page_frame(kicker: str, title: str, subtitle: str = "") -> None:
    init_session()
    inject_theme()
    render_sidebar()
    st.markdown('<div class="kb-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="kb-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kb-title">{escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="kb-subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)


def close_page_frame() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_text_query_runner():
    from agent.graph import run_text_query
    return run_text_query


@st.cache_resource(show_spinner=False)
def get_image_query_runner():
    from agent.graph import run_image_query
    return run_image_query


def friendly_error(state: dict[str, Any] | None) -> str | None:
    if not state:
        return None
    errors = state.get("errors") or []
    if not errors:
        return None
    text = " ".join(str(error).lower() for error in errors)
    market = state.get("market_result")
    if market is not None and getattr(market, "available", False) and any(
        token in text for token in ("synthesis", "llm", "provider")
    ):
        return None
    if "llm" in text or "provider" in text or "synthesis" in text:
        return "Answer service is temporarily unavailable. The available source cards are still shown."
    if "rag" in text or "chrom" in text or "retriev" in text:
        return "Knowledge documents are temporarily unavailable. Market and weather cards are shown separately when available."
    if "dam" in text or "pdf" in text or "market" in text or "tls" in text:
        return "The market source is temporarily unavailable. No price was estimated."
    if "vision" in text or "model" in text:
        return "The image could not be analyzed. Please try a clear image again."
    return "The request could not be completed. Please try again later."


def append_chat(user_text: str, state: dict[str, Any]) -> None:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": state.get("final_answer", ""),
        "state": state,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    st.session_state.last_state = state


def render_chat_history() -> None:
    for message in st.session_state.get("chat_history", []):
        if message.get("role") == "user":
            text = escape(str(message.get("content", "")))
            st.markdown(f'<div class="kb-chat-user">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kb-chat-bot">', unsafe_allow_html=True)
            st.markdown(message.get("content", ""))
            st.markdown("</div>", unsafe_allow_html=True)
            render_state_cards(message.get("state") or {}, compact=True)


def render_weather_card(weather: Any) -> None:
    if weather is None:
        return
    if not weather.available:
        error_text = escape(str(weather.error or "Source unavailable"))
        st.markdown(f'<div class="kb-card kb-note">Weather unavailable · আবহাওয়া পাওয়া যায়নি: {error_text}</div>', unsafe_allow_html=True)
        return
    location = escape(str(weather.location or "Unknown"))
    description = escape(str(weather.description or "No description"))
    rain = "Rain observed now / এই মুহূর্তে বৃষ্টি হচ্ছে" if weather.rain_expected is True else "No rain observed now / এই মুহূর্তে বৃষ্টি নেই" if weather.rain_expected is False else "Current rain status unavailable / তথ্য অনির্দিষ্ট"
    temperature = weather.temperature_c if weather.temperature_c is not None else "—"
    humidity = weather.humidity_percent if weather.humidity_percent is not None else "—"
    st.markdown(
        f'''<div class="kb-card"><h3>Current weather · বর্তমান আবহাওয়া</h3>
        <div class="kb-meta">{location}</div><div class="kb-price">{temperature}°C</div>
        <div class="kb-meta">{description} · Humidity / আর্দ্রতা: {humidity}%<br>{rain}</div>
        <div class="kb-note">Current observation only · এটি forecast নয়।</div></div>''',
        unsafe_allow_html=True,
    )


def render_price_card(price: Any) -> None:
    if price is None:
        return
    if not price.available:
        error_text = escape(str(price.error or "Source unavailable"))
        st.markdown(f'<div class="kb-card kb-note">Market price unavailable · বাজারদর পাওয়া যায়নি: {error_text}</div>', unsafe_allow_html=True)
        return
    crop = escape(str(price.crop or "Product"))
    source = "DAM daily divisional retail market report (PDF) / DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)"
    note = escape(str(price.division_note or "Not specified"))
    st.markdown(
        f'''<div class="kb-card"><h3>DAM market price · {crop}</h3>
        <div class="kb-price">{escape(str(price.price_range or price.price_bdt_per_kg))}</div>
        <div class="kb-meta">Unit / একক: {escape(str(price.unit or "প্রতি কেজি"))} · Average / গড়: {escape(str(price.price_bdt_per_kg))}<br>
        Division / বিভাগ: {note} · Date / তারিখ: {escape(str(price.date or "Not specified"))}<br>
        Source / উৎস: {source}</div></div>''',
        unsafe_allow_html=True,
    )


def _market_category(item: Any) -> str:
    text = f"{getattr(item, 'category', '')} {getattr(item, 'product', '')}".casefold()
    # ``চালকুমড়া`` contains the substring ``চাল`` but is a vegetable, and
    # potato is also shown with vegetables/produce rather than grains here.
    if any(word in text for word in ("চালকুমড়া", "চালকুমড়া", "চালকুমড়ো", "আলু")):
        return "Vegetables & produce / সবজি ও ফল"
    if any(word in text for word in (
        "মাছ", "মোছ", "রুই", "কাতলা", "ইলিশ", "পাঙ্গাস", "পাংগাস", "তেলাপিয়া", "তেলাপিয়া",
        "মাংস", "মোাংস", "গরু", "ছাগল", "খাসি", "মুরগি", "মুরগী", "মোরগ",
    )):
        return "Fish & meat / মাছ ও মাংস"
    if any(word in text for word in ("ডিম", "নডম", "egg")):
        return "Eggs & other / ডিম ও অন্যান্য"
    if any(word in text for word in (
        "আটা", "চাল", "ধান", "ডাল", "মসুর", "মুগ", "মাষকলাই", "কেসারি", "ছোলা", "লবণ", "দুধ", "দুি", "rice", "lentil",
    )):
        return "Grains & pulses / শস্য ও ডাল"
    if any(word in text for word in ("তেল", "সয়াবিন", "সয়াবিন", "পাম", "সরিষার", "চিনি", "soybean")):
        return "Oil & sugar / তেল ও চিনি"
    if any(word in text for word in (
        "পেঁয়াজ", "পেঁয়াজ", "রসুন", "আদা", "মরিচ", "বেগুন", "কুমড়া", "কুমড়া",
        "চিচিঙ্গা", "ধুন্দুল", "কচু", "বরবটি", "লাউ", "শসা", "উচ্ছে", "করল্লা",
        "ঝিঙ্গা", "ঢেঁড়স", "পটল", "ফুলকপি", "বাঁধাকপি", "মূলা", "মুলা", "শিম",
        "পেঁপে", "আপেল", "পেয়ারা", "পেয়ারা", "আম", "কলা", "vegetable", "fruit",
    )):
        return "Vegetables & produce / সবজি ও ফল"
    return "Other / অন্যান্য"


def render_market_card(market: Any, compact: bool = False) -> None:
    if market is None:
        return
    if not market.available:
        error_text = escape(str(market.error or "Source unavailable"))
        st.markdown(f'<div class="kb-card kb-note">Full market report unavailable · পূর্ণ বাজারদর পাওয়া যায়নি: {error_text}</div>', unsafe_allow_html=True)
        return
    division = escape(str(market.division or "Unknown division"))
    date = escape(str(market.date or "Not specified"))
    source = "DAM daily divisional retail market report (PDF) / DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)"
    st.markdown(
        f'<div class="kb-card"><h3>Full market report · পূর্ণ বাজারদর</h3><div class="kb-meta">{division} · Report date / রিপোর্টের তারিখ: {date}<br>Source / উৎস: {source}</div></div>',
        unsafe_allow_html=True,
    )
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in market.items:
        groups[_market_category(item)].append(item)
    order = [
        "Grains & pulses / শস্য ও ডাল",
        "Oil & sugar / তেল ও চিনি",
        "Vegetables & produce / সবজি ও ফল",
        "Fish & meat / মাছ ও মাংস",
        "Eggs & other / ডিম ও অন্যান্য",
        "Other / অন্যান্য",
    ]
    for category in order:
        items = groups.get(category, [])
        if not items:
            continue
        with st.expander(f"{category} · {len(items)} items / পণ্য", expanded=not compact):
            for item in items:
                st.markdown(f"**{escape(str(item.product))}**  \n{escape(str(item.price_range))} · {escape(str(item.unit or 'Unit unavailable / একক উল্লেখ নেই'))}")
def render_citations(state: dict[str, Any]) -> None:
    citations = state.get("citations") or []
    if citations:
        with st.expander("Sources / তথ্যসূত্র", expanded=False):
            for citation in citations:
                page = f", page / পৃষ্ঠা {citation.get('page')}" if citation.get("page") is not None else ""
                st.caption(f"{citation.get('source', 'Unknown source')}{page}")


def render_state_cards(state: dict[str, Any], compact: bool = False) -> None:
    render_weather_card(state.get("weather_result"))
    # Specific price values are already printed once in final_answer. Price
    # cards are reserved for full-market structured reports.
    render_market_card(state.get("market_result"), compact=compact)
    render_citations(state)
