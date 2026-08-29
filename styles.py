"""Central CSS for KrishokBot's Claude-style Streamlit UI.

Import and call inject() once per page. Keeping all CSS in one place
avoids drift between Chat / About / Limitations pages.
"""

import streamlit as st

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --kb-bg: #FAF9F6;
    --kb-surface: #FFFFFF;
    --kb-border: #E8E5DE;
    --kb-text: #262624;
    --kb-text-muted: #6B6862;
    --kb-accent: #2F6F4E;      /* paddy green */
    --kb-accent-dark: #234F38;
    --kb-accent-soft: #E7F2EC;
    --kb-amber: #B8860B;
    --kb-red: #B3401E;
    --kb-user-bubble: #F0EEE6;
    --kb-radius: 18px;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans Bengali', 'Inter', -apple-system, sans-serif;
}

#MainMenu, footer {visibility: hidden;}
.stDeployButton {display: none;}
header[data-testid="stHeader"] {background: transparent;}
header[data-testid="stHeader"] [data-testid="stToolbarActions"] {display: none;}
[data-testid="stSidebarCollapsedControl"] {visibility: visible !important;}

.stApp {
    background: var(--kb-bg);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 100%;
}

section[data-testid="stSidebar"] {
    background: #F5F3EE;
    border-right: 1px solid var(--kb-border);
}

.kb-block {
    max-width: 780px;
    margin: 0 auto;
}

/* ---------- Sidebar brand ---------- */
.kb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 2px 18px 2px;
}
.kb-brand-logo {
    font-size: 26px;
    line-height: 1;
}
.kb-brand-title {
    font-weight: 700;
    font-size: 19px;
    color: var(--kb-accent-dark);
}
.kb-brand-sub {
    font-size: 12px;
    color: var(--kb-text-muted);
    margin-top: -2px;
}

.kb-chip {
    display: inline-block;
    background: var(--kb-accent-soft);
    color: var(--kb-accent-dark);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12.5px;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    border: 1px solid #d6ebe0;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 4px 0;
    max-width: 780px;
    margin: 0 auto;
}

[data-testid="stChatMessageContent"] {
    font-size: 15.5px;
    line-height: 1.65;
    color: var(--kb-text);
}

/* user bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--kb-user-bubble);
    border-radius: var(--kb-radius);
    padding: 12px 18px;
}

.kb-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 6px 0;
}
.kb-badge-green { background: #E4F4EA; color: #1E7A46; }
.kb-badge-amber { background: #FBF0DA; color: #92650B; }
.kb-badge-red   { background: #FBE7E1; color: #A8371A; }

.kb-source-card {
    border: 1px solid var(--kb-border);
    background: var(--kb-surface);
    border-radius: 12px;
    padding: 9px 13px;
    margin: 4px 0;
    font-size: 13px;
    color: var(--kb-text-muted);
}
.kb-source-card b { color: var(--kb-text); }

.kb-warn-box {
    background: #FBF0DA;
    border: 1px solid #EED9A8;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 13.5px;
    color: #6B4E00;
    margin: 6px 0;
}

.kb-empty-state {
    text-align: center;
    padding: 60px 20px 20px 20px;
    color: var(--kb-text-muted);
}
.kb-empty-state h2 {
    color: var(--kb-text);
    font-size: 26px;
    margin-bottom: 6px;
}

/* ---------- Chat input (Claude-style pill) ---------- */
[data-testid="stChatInput"] {
    max-width: 780px;
    margin: 0 auto;
}
[data-testid="stChatInput"] > div {
    background: var(--kb-surface) !important;
    border: 1.5px solid var(--kb-border) !important;
    border-radius: 26px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
}
[data-testid="stChatInputSubmitButton"] button,
[data-testid="stChatInput"] button {
    background: var(--kb-accent) !important;
    border-radius: 50% !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: var(--kb-accent-dark) !important;
}

/* ---------- Stop-generating control ---------- */
.kb-stop-wrap {
    max-width: 780px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    padding-bottom: 6px;
}
div[data-testid="stButton"] button.kb-stop-btn,
.kb-stop-wrap button {
    background: var(--kb-surface);
    border: 1.5px solid var(--kb-border);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 13px;
    color: var(--kb-text);
}

/* ---------- Confidence tier text helper ---------- */
.kb-caption { color: var(--kb-text-muted); font-size: 12.5px; }

/* ---------- Recruiter-facing pages (About / Limitations) ---------- */
.kb-page { max-width: 1080px; margin: 0 auto; padding: 4px 8px; }

.kb-hero {
    padding: 34px 38px;
    border-radius: 12px;
    border: 1px solid var(--kb-border);
    background: var(--kb-surface);
    margin-bottom: 28px;
}
.kb-hero h1 { font-size: 1.9rem; margin: 0 0 10px 0; color: var(--kb-text); font-family: 'Inter', sans-serif; }
.kb-hero p { margin: 0; font-size: 1.05rem; line-height: 1.6; color: var(--kb-text-muted); max-width: 760px; }

.kb-label {
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--kb-accent-dark);
    margin: 26px 0 10px 0;
}

.kb-card {
    border: 1px solid var(--kb-border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    background: var(--kb-surface);
    line-height: 1.7;
}

.kb-status-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.kb-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.kb-dot-green { background: #1E7A46; }
.kb-dot-amber { background: #92650B; }
.kb-dot-red   { background: #A8371A; }
.kb-dot-gray  { background: #8A8780; }

.kb-metric-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.kb-metric-table th, .kb-metric-table td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--kb-border); }
.kb-metric-table th { color: var(--kb-text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
.kb-metric-table td { color: var(--kb-text); }

.kb-limit-item {
    border-left: 3px solid var(--kb-amber);
    background: var(--kb-accent-soft);
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 14px;
    font-size: 14.5px;
    line-height: 1.65;
}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def confidence_badge(confidence: float) -> str:
    """Return HTML for a green/amber/red confidence badge per project thresholds."""
    if confidence >= 0.85:
        tier, cls, label = "high", "kb-badge-green", "উচ্চ নির্ভরযোগ্যতা"
    elif confidence >= 0.60:
        tier, cls, label = "medium", "kb-badge-amber", "মাঝারি নির্ভরযোগ্যতা"
    else:
        tier, cls, label = "low", "kb-badge-red", "কম নির্ভরযোগ্যতা"
    return f'<span class="kb-badge {cls}">● {label} — {confidence*100:.2f}%</span>'

