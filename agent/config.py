"""Centralized configuration for KrishokBot.

All values are read from environment variables, with safe defaults for the
public DAM endpoints and non-secret runtime settings. Put secrets such as
OPENWEATHER_API_KEY and LLM keys in ``.env``; never commit that file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIR: Final[Path] = PROJECT_ROOT.parent
# Support both common layouts: E:\\krishokbot\\.env and
# E:\\krishokbot\\agent\\.env. Existing process environment variables keep
# precedence because python-dotenv defaults to override=False.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = _env_text(name, str(default))
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = _env_text(name, str(default))
    try:
        return max(minimum, float(raw))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_text(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------

LLM_PROVIDER = _env_text("KRISHOKBOT_LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = _env_text("GROQ_API_KEY")
GROQ_MODEL = _env_text("KRISHOKBOT_GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = _env_text("GEMINI_API_KEY")
GEMINI_MODEL = _env_text("KRISHOKBOT_GEMINI_MODEL", "gemini-3.6-flash")
LLM_TIMEOUT_SECONDS = _env_float("KRISHOKBOT_LLM_TIMEOUT", 20.0, minimum=1.0)
LLM_MAX_RETRIES = _env_int("KRISHOKBOT_LLM_MAX_RETRIES", 2, minimum=0)
# Shared generation safety defaults. They are deliberately bounded so a long
# RAG or full-market response cannot consume an unbounded completion budget.
LLM_MAX_OUTPUT_TOKENS = _env_int("KRISHOKBOT_LLM_MAX_OUTPUT_TOKENS", 1200, minimum=128)
LLM_FREQUENCY_PENALTY = _env_float("KRISHOKBOT_LLM_FREQUENCY_PENALTY", 0.15, minimum=0.0)
LLM_PRESENCE_PENALTY = _env_float("KRISHOKBOT_LLM_PRESENCE_PENALTY", 0.0, minimum=0.0)
LLM_DEGENERACY_MIN_WORDS = _env_int("KRISHOKBOT_LLM_DEGENERACY_MIN_WORDS", 30, minimum=10)
LLM_DEGENERACY_REPEAT_COUNT = _env_int("KRISHOKBOT_LLM_DEGENERACY_REPEAT_COUNT", 3, minimum=2)

# ---------------------------------------------------------------------------
# Weather — OpenWeatherMap
# ---------------------------------------------------------------------------

OPENWEATHER_API_KEY = _env_text("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = _env_text(
    "KRISHOKBOT_OPENWEATHER_BASE_URL",
    "https://api.openweathermap.org/data/2.5/weather",
)
OPENWEATHER_GEOCODING_URL = _env_text(
    "KRISHOKBOT_OPENWEATHER_GEOCODING_URL",
    "https://api.openweathermap.org/geo/1.0/direct",
)
WEATHER_TIMEOUT_SECONDS = _env_float("KRISHOKBOT_WEATHER_TIMEOUT", 10.0, minimum=1.0)
WEATHER_MAX_RETRIES = _env_int("KRISHOKBOT_WEATHER_MAX_RETRIES", 2, minimum=0)
WEATHER_COUNTRY_CODE = _env_text("KRISHOKBOT_WEATHER_COUNTRY", "BD").upper()

# ---------------------------------------------------------------------------
# Bangladesh Department of Agricultural Marketing (DAM)
# ---------------------------------------------------------------------------

# These defaults match the user's .env values exactly.
DAM_PRICE_SOURCE_URL = _env_text(
    "KRISHOKBOT_DAM_URL",
    "https://market.dam.gov.bd/",
)
DAM_PDF_PAGE_URL = _env_text(
    "KRISHOKBOT_DAM_PDF_PAGE_URL",
    "https://dam.gov.bd/pages/static-pages/6922e0d1933eb65569e28b21",
)

# The nationwide market ticker is only used when the user did not specify a
# division. Division-specific requests always use the corresponding PDF.
DAM_DEFAULT_DIVISION = _env_text("KRISHOKBOT_DAM_DEFAULT_DIVISION", "ঢাকা")

PRICE_TIMEOUT_SECONDS = _env_float("KRISHOKBOT_PRICE_TIMEOUT", 20.0, minimum=2.0)
PDF_TIMEOUT_SECONDS = _env_float("KRISHOKBOT_PDF_TIMEOUT", 45.0, minimum=2.0)
DAM_MAX_RETRIES = _env_int("KRISHOKBOT_DAM_MAX_RETRIES", 2, minimum=0)
DAM_RETRY_BACKOFF_SECONDS = _env_float(
    "KRISHOKBOT_DAM_RETRY_BACKOFF",
    1.0,
    minimum=0.0,
)
DAM_MAX_PDF_BYTES = _env_int("KRISHOKBOT_DAM_MAX_PDF_BYTES", 20_000_000, minimum=1_000_000)

# TLS verification is enabled by default. If the DAM server presents a broken
# chain in a particular deployment, configure a CA bundle rather than turning
# verification off. The insecure fallback is opt-in and never silent.
DAM_TLS_VERIFY = _env_bool("KRISHOKBOT_DAM_TLS_VERIFY", True)
DAM_CA_BUNDLE = _env_text("KRISHOKBOT_DAM_CA_BUNDLE")
DAM_ALLOW_INSECURE_TLS_FALLBACK = _env_bool(
    "KRISHOKBOT_DAM_ALLOW_INSECURE_TLS_FALLBACK",
    False,
)

# The static page contains the full archive currently exposed by DAM. We select
# the newest report date <= the runtime date, without accepting future dates.
DAM_MAX_REPORT_AGE_DAYS = _env_int(
    "KRISHOKBOT_DAM_MAX_REPORT_AGE_DAYS",
    0,
    minimum=0,
)
# Compatibility names retained for existing agent code.
DAM_PDF_MAX_LOOKBACK_DAYS = DAM_MAX_REPORT_AGE_DAYS
PRICE_LIVE_MAX_LOOKBACK_DAYS = DAM_MAX_REPORT_AGE_DAYS

# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

PRICE_CACHE_PATH = _env_text(
    "KRISHOKBOT_PRICE_CACHE_PATH",
    "eval/results/price_cache.json",
)
PRICE_CACHE_MAX_AGE_HOURS = _env_float(
    "KRISHOKBOT_PRICE_CACHE_MAX_AGE_HOURS",
    1.0,
    minimum=0.0,
)
PRICE_LIVE_BUDGET_SECONDS = _env_float(
    "KRISHOKBOT_PRICE_LIVE_BUDGET",
    35.0,
    minimum=1.0,
)

# ---------------------------------------------------------------------------
# Router / guardrails
# ---------------------------------------------------------------------------

ROUTER_MAX_LLM_ATTEMPTS = _env_int("KRISHOKBOT_ROUTER_ATTEMPTS", 2, minimum=1)
NORMALIZATION_MIN_BANGLA_RATIO = _env_float(
    "KRISHOKBOT_MIN_BANGLA_RATIO",
    0.7,
    minimum=0.0,
)
MAX_QUERY_LENGTH_CHARS = _env_int("KRISHOKBOT_MAX_QUERY_CHARS", 1000, minimum=100)


def validate_config() -> list[str]:
    """Return configuration problems without exposing secret values."""
    problems: list[str] = []

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        problems.append("GROQ_API_KEY is missing.")
    elif LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        problems.append("GEMINI_API_KEY is missing.")
    elif LLM_PROVIDER not in {"groq", "gemini"}:
        problems.append(
            f"Unsupported KRISHOKBOT_LLM_PROVIDER: {LLM_PROVIDER!r}; use 'groq' or 'gemini'."
        )

    if not OPENWEATHER_API_KEY:
        problems.append("OPENWEATHER_API_KEY is missing.")
    if not DAM_PDF_PAGE_URL.startswith(("http://", "https://")):
        problems.append("KRISHOKBOT_DAM_PDF_PAGE_URL must be an HTTP(S) URL.")
    if not DAM_PRICE_SOURCE_URL.startswith(("http://", "https://")):
        problems.append("KRISHOKBOT_DAM_URL must be an HTTP(S) URL.")
    if DAM_ALLOW_INSECURE_TLS_FALLBACK:
        problems.append(
            "DAM insecure TLS fallback is enabled; use a valid CA bundle in production."
        )

    return problems


def config_status() -> dict[str, object]:
    """Return safe diagnostic information; never returns API key contents."""
    return {
        "llm_provider": LLM_PROVIDER,
        "groq_key_loaded": bool(GROQ_API_KEY),
        "groq_key_length": len(GROQ_API_KEY),
        "gemini_key_loaded": bool(GEMINI_API_KEY),
        "gemini_key_length": len(GEMINI_API_KEY),
        "openweather_key_loaded": bool(OPENWEATHER_API_KEY),
        "openweather_key_length": len(OPENWEATHER_API_KEY),
        "weather_country": WEATHER_COUNTRY_CODE,
        "dam_price_source": DAM_PRICE_SOURCE_URL,
        "dam_pdf_page": DAM_PDF_PAGE_URL,
        "dam_default_division": DAM_DEFAULT_DIVISION,
        "dam_tls_verify": DAM_TLS_VERIFY,
        "dam_ca_bundle_configured": bool(DAM_CA_BUNDLE),
        "dam_insecure_fallback": DAM_ALLOW_INSECURE_TLS_FALLBACK,
        "problems": validate_config(),
    }


if __name__ == "__main__":
    print("KrishokBot configuration status")
    print("-" * 40)
    for key, value in config_status().items():
        print(f"{key}: {value}")
