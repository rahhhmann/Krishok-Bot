"""Production-grade external tools for KrishokBot.

The module provides three user-facing capabilities:

* OpenWeatherMap weather lookup for Bangladesh locations.
* DAM nationwide ticker lookup when no division is explicitly requested.
* DAM division-wise PDF lookup for an explicitly requested division, or as
  the fallback when the nationwide ticker does not contain the crop.

The division-wise flow is intentionally strict:

    DAM HTML table -> newest report date <= runtime date -> division cell ->
    PDF -> today's retail column for that report date -> crop row -> price.

It never accepts a future-dated report, never guesses a price from an
unrelated numeric column, and returns an unavailable result when the product
is absent from the selected PDF.

Optional dependency: ``pdfplumber`` is required for PDF extraction.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from agent import config
except ImportError:  # Allows direct local diagnostics from the agent directory.
    import config  # type: ignore[no-redef]

try:
    import pdfplumber
except ImportError:  # The public API returns a clean unavailable result.
    pdfplumber = None


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ---------------------------------------------------------------------------
# Location and name normalization
# ---------------------------------------------------------------------------

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

BD_LOCATIONS: dict[str, tuple[float, float]] = {
    "barishal": (22.7010, 90.3535),
    "barisal": (22.7010, 90.3535),
    "বরিশাল": (22.7010, 90.3535),
    "dhaka": (23.8103, 90.4125),
    "ঢাকা": (23.8103, 90.4125),
    "chattogram": (22.3569, 91.7832),
    "chittagong": (22.3569, 91.7832),
    "চট্টগ্রাম": (22.3569, 91.7832),
    "khulna": (22.8456, 89.5403),
    "খুলনা": (22.8456, 89.5403),
    "rajshahi": (24.3745, 88.6042),
    "রাজশাহী": (24.3745, 88.6042),
    "sylhet": (24.8949, 91.8687),
    "সিলেট": (24.8949, 91.8687),
    "rangpur": (25.7439, 89.2752),
    "রংপুর": (25.7439, 89.2752),
    "mymensingh": (24.7471, 90.4203),
    "ময়মনসিংহ": (24.7471, 90.4203),
    "ময়মনসিংহ": (24.7471, 90.4203),
}

DIVISION_ALIASES = {
    "dhaka": "ঢাকা",
    "ঢাকা": "ঢাকা",
    "chattogram": "চট্টগ্রাম",
    "chittagong": "চট্টগ্রাম",
    "চট্টগ্রাম": "চট্টগ্রাম",
    "barishal": "বরিশাল",
    "barisal": "বরিশাল",
    "বরিশাল": "বরিশাল",
    "khulna": "খুলনা",
    "খুলনা": "খুলনা",
    "rajshahi": "রাজশাহী",
    "রাজশাহী": "রাজশাহী",
    "sylhet": "সিলেট",
    "সিলেট": "সিলেট",
    "rangpur": "রংপুর",
    "রংপুর": "রংপুর",
    "mymensingh": "ময়মনসিংহ",
    "ময়মনসিংহ": "ময়মনসিংহ",
    "ময়মনসিংহ": "ময়মনসিংহ",
}

DIVISIONS = ["ঢাকা", "চট্টগ্রাম", "খুলনা", "রাজশাহী", "বরিশাল", "রংপুর", "সিলেট", "ময়মনসিংহ"]

CROP_ALIASES = {
    "potato": "potato",
    "আলু": "potato",
    "আলু দেশি": "potato",
    "tomato": "tomato",
    "টমেটো": "tomato",
    "onion": "onion",
    "মসুর ডাল": "masur dal",
    "মসুর": "masur dal",
    "masur dal": "masur dal",
    "lentil": "masur dal",
    "পেঁয়াজ": "onion",
    "পেঁয়াজ": "onion",
    "পিয়াজ": "onion",
    "পিয়াজ": "onion",
    "garlic": "garlic",
    "roshun": "garlic",
    "rosun": "garlic",
    "rosuner": "garlic",
    "রসুন": "garlic",
    "রসুনের": "garlic",
    "ginger": "ginger",
    "আদা": "ginger",
    "green chili": "green chili",
    "green chilli": "green chili",
    "কাঁচা মরিচ": "green chili",
    "কাচা মরিচ": "green chili",
    "মরিচ": "green chili",
    "rice": "rice",
    "চাল": "rice",
    "sugar": "sugar",
    "চিনি": "sugar",
    "soybean oil": "soybean oil",
    "soyabean oil": "soybean oil",
    "সয়াবিন তেল": "soybean oil",
    "সয়াবিন তেল": "soybean oil",
    "egg": "egg",
    "ডিম": "egg",
    "beef": "beef",
    "গরুর মাংস": "beef",
    "mutton": "mutton",
    "খাসির মাংস": "mutton",
    "ছাগলের মাংস": "mutton",
    "lau": "bottle gourd",
    "লাউ": "bottle gourd",
    "লাও": "bottle gourd",
    "ash gourd": "ash gourd",
    "চালকুমড়া": "ash gourd",
    "চালকুমড়া": "ash gourd",
    "চালকুমড়ো": "ash gourd",
    "চালকুমড়ো": "ash gourd",
}

DEFAULT_CROPS = [
    "rice", "onion", "garlic", "green chili", "ginger", "sugar",
    "soybean oil", "egg", "beef",     "mutton", "potato", "tomato", "bottle gourd", "ash gourd",
]


TICKER_CROPS = {
    "rice", "onion", "garlic", "green chili", "ginger", "sugar",
    "soybean oil", "egg", "beef", "mutton",
}

# High-confidence OCR corrections observed in DAM Bengali PDFs. The map is
# deliberately conservative: unknown text is preserved instead of guessed.
PRODUCT_EXACT_CORRECTIONS: dict[str, str] = {
    "আটো": "আটা",
    "প্যোরেট-সোেো": "প্যাকেট-সাদা",
    "প্যোরেট-সোদো": "প্যাকেট-সাদা",
    "ক োলো": "খোলা",
    "কখোিো": "খোলা",
    "কেোিো": "খোলা",
    "ক ািা": "খোলা",
    "ডোি": "ডাল",
    "ডোল": "ডাল",
    "মসুর ডোি": "মসুর ডাল",
    "মোশ েলোই": "মাষকলাই ডাল",
    "মোশ েিোই": "মাষকলাই ডাল",
    "ক সোরী": "কেসারি ডাল",
    "ক সোরী ডাল": "কেসারি ডাল",
    "মশুি িাি": "মসুর ডাল",
    "মুগ ডোি": "মুগ ডাল",
    "মুগ িাি": "মুগ ডাল",
    "কখসোরী ডোি": "কেসারি ডাল",
    "কছোিো": "ছোলা",
    "প্যালেট": "প্যাকেট",
    "প্যোরেট": "প্যাকেট",
    "কিি-সয়োনর্ি": "সয়াবিন তেল",
    "কিি-সয়োনর্ি": "সয়াবিন তেল",
    "কিি-পোম": "পাম তেল",
    "কিি": "তেল",
    "সয়ারবন কিি": "সয়াবিন তেল",
    "সয়ারবন ক্িি": "সয়াবিন তেল",
    "পাম কিি": "পাম তেল",
    "পোম": "পাম",
    "নচনি": "চিনি",
    "সোেো": "সাদা",
    "সোদো": "সাদা",
    "সয়োনর্ি": "সয়াবিন",
    "সয়োনর্ি": "সয়াবিন",
    "আমেোিীকৃি": "আমদানিকৃত",
    "আমোনীকৃি": "আমদানিকৃত",
    "কেশী": "দেশী",
    "কদশী": "দেশী",
    "উন্নি": "উন্নত",
    "কমোটো": "মোটা",
    "কমাটা": "মোটা",
    "রসুি": "রসুন",
    "িসুন": "রসুন",
    "রেঁয়াি": "পেঁয়াজ",
    "কেঁয়োি": "পেঁয়াজ",
    "কেঁয়াি": "পেঁয়াজ",
    "আেো": "আদা",
    "আো": "আদা",
    "চোয়িো": "চায়না",
    "েযোি": "ক্যান",
    "1নিিঃ": "১ লি.",
    "5নিিঃ": "৫ লি.",
    "শুেিো মনরচ": "শুকনা মরিচ",
    "োঁচো মনরচ": "কাঁচা মরিচ",
    "োঁচামরিচ": "কাঁচা মরিচ",
    "কর্গুি": "বেগুন",
    "কবগুন": "বেগুন",
    "োঁচালেঁলপ": "কাঁচা পেঁপে",
    "রমরিকুম া": "মিষ্টি কুমড়া",
    "রমরিকুমড়া": "মিষ্টি কুমড়া",
    "োঁচো কেঁরপ": "কাঁচা পেঁপে",
    "নমনি কুমড়ো": "মিষ্টি কুমড়া",
    "নচনচাংগো": "চিচিঙ্গা",
    "চিচিিংগা": "চিচিঙ্গা",
    "পরের িোম": "পণ্যের নাম",
    "পরের প্রেোর": "পণ্যের প্রকার",
    "ধুন্দি": "ধুন্দুল",
    "েচুরিনি": "কচুরলতি",
    "েচুরমুখী": "কচুমুখী",
    "িোউ": "লাউ",
    "চোিকুমড়ো": "চালকুমড়া",
    "িোলকুমড়ো": "চালকুমড়া",
    "চোিকুমড়ো": "চালকুমড়া",
    "ন াংগো": "ঝিঙ্গা",
    "র্রর্টি": "বরবটি",
    "শসো": "শসা",
    "েোিি মোছ": "কাতলা মাছ",
    "রুই মোছ": "রুই মাছ",
    "তেলোনপয়ো মোছ": "তেলাপিয়া মাছ",
    "মোাংস- গরু": "মাংস- গরু",
    "কমোরগ-মুরগি": "মোরগ-মুরগি",
    "কমোরগ": "মোরগ",
    "মোাংস": "মাংস",
    "নডমিঃ": "ডিম:",
    "রুই (": "রুই মাছ (",
    "পোাংগোস মোছ": "পাঙ্গাস মাছ",
    "পোাংগোস": "পাঙ্গাস",
    "পাাংগাস মোছ": "পাঙ্গাস মাছ",
    "কিলোনপয়ো": "তেলাপিয়া",
    "তেলোনপয়ো": "তেলাপিয়া",
    "ইনিশ মোছ": "ইলিশ মাছ",
    "চোরের": "চাষের",
    "ছোলা-েিোই": "ছোলা",
    "কেসোরী": "কেসারি",
    "মুগ ডাি": "মুগ ডাল",
    "বমাটা": "মোটা",
    "ব ািা": "খোলা",
    "হোইনিড": "হাইব্রিড",
    "রসুন বেশী": "রসুন (দেশী)",
    "আদা বেশী": "আদা (দেশী)",
    "রুই েো": "রুই মাছ",
    "আটা (প্যাদেট)": "আটা (প্যাকেট)",
    "আটা (কখািা)": "আটা (খোলা)",
    "সয়াবিন তেল (কখািা)": "সয়াবিন তেল (খোলা)",
    "পাম তেল (কখািা)": "পাম তেল (খোলা)",
    "পেঁয়াজ দেশী": "পেঁয়াজ (দেশী)",
    "পেঁয়াজ (কদনশ)িতুি": "পেঁয়াজ (দেশী)",
    "আদো-আমদোিী": "আদা (আমদানিকৃত)",
    "আদা (আেেোিীকৃি)": "আদা (আমদানিকৃত)",
    "আলু-হল্যোন্ড": "আলু (হল্যান্ড)",
    "তেলোনপয়ো মোছ": "তেলাপিয়া মাছ",
    "রুই মা (চাষকৃি)": "রুই মাছ (চাষের)",
    "মোংস- গরু": "গরু",
    "মোিংস- গরু": "গরু",
    "মাাংস- গরু": "গরু",
    "রসুন (িোয়িো)": "রসুন (চায়না)",
    "রসুন (িায়িা)": "রসুন (চায়না)",
    "রুই মাছ (িোরের)": "রুই মাছ (চাষের)",
    "রুই (িোরের)": "রুই মাছ (চাষের)",
    "নর্নিন্ন": "বিভিন্ন",
    "সনরেোর": "সরিষার",
    "সনরেো": "সরিষার তেল",
    "সয়াবিন (": "সয়াবিন তেল (",
    "সয়াবিন (": "সয়াবিন তেল (",
    "পাম (": "পাম তেল (",
    "িোনি": "লম্বা",
    "ঊরে": "উচ্ছে",
    "েরল্লো": "করল্লা",
    "গ্রোম": "গ্রাম",
    "কেড়স": "ঢেঁড়স",
    "কেড়স": "ঢেঁড়স",
    "পটি": "পটল",
    "টরমরটো": "টমেটো",
    "টমরটো": "টমেটো",
    "মুরনগ": "মুরগি",
    "মুনরগ": "মুরগি",
    "কসোিোলী": "সোনালী",
    "কসোিোিী": "সোনালী",
    "কসোিোলী": "সোনালী",
    "িয়লোর": "ব্রয়লার",
}

# Phrase replacements handle rows containing category/product qualifiers.
# Longer phrases are applied first to avoid partial replacements.
PRODUCT_PHRASE_CORRECTIONS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        PRODUCT_EXACT_CORRECTIONS.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

# Only rows containing at least one high-confidence market term are shown to
# farmers. Unknown rows are omitted and counted instead of being guessed.
# This list is intentionally broad across all Bangladesh division reports.
PRODUCT_CONFIDENCE_TOKENS = (
    "আটা", "মসুর", "মুগ", "ডাল", "কেসারি", "ছোলা", "তেল", "চিনি",
    "পেঁয়াজ", "পেঁয়াজ", "রসুন", "আদা", "মরিচ", "আলু", "বেগুন", "পেঁপে",
    "কুমড়া", "কুমড়া", "চিচিঙ্গা", "ধুন্দুল", "কচু", "বরবটি", "লাউ", "চালকুমড়া",
    "চালকুমড়া", "শসা", "উচ্ছে", "করল্লা", "ঝিঙ্গা", "ঢেঁড়স", "পটল", "ফুলকপি",
    "বাঁধাকপি", "মূলা", "মুলা", "শিম", "পটল", "চাল", "ধান", "লবণ", "দুধ",
    "আপেল", "পেয়ারা", "পেয়ারা", "আম", "কলা", "মাছ", "রুই", "কাতলা", "ইলিশ",
    "পাঙ্গাস", "তেলাপিয়া", "তেলাপিয়া", "মাংস", "গরু", "ছাগল", "মুরগি", "মোরগ",
    "ডিম", "ব্রয়লার", "ব্রয়লার",
)

# Product-specific corrections are applied only after the product text is
# cleaned. This avoids trusting a shifted/OCR-corrupted unit cell for known
# piece-count vegetables.
PRODUCT_UNIT_OVERRIDES: dict[str, str] = {
    "লাউ": "প্রতিটি",
    "চালকুমড়া": "প্রতিটি",
    "চালকুমড়া": "প্রতিটি",
}


CROP_VARIANTS: dict[str, tuple[str, ...]] = {
    "potato": ("আলু", "potato"),
    "tomato": ("টমেটো", "টমরটো", "টরমরটো", "টমেট", "tomato"),
    "onion": ("পেঁয়াজ", "পেঁয়াজ", "পিয়াজ", "পিয়াজ", "রেঁয়াি", "রেঁয়াজ", "কেঁয়াি", "কেঁয়োি", "onion"),
    "masur dal": ("মসুর ডাল", "মসুর ডোি", "মসুর", "masur dal", "lentil"),
    "garlic": ("রসুন", "িসুন", "garlic"),
    "ginger": ("আদা", "আেো", "আো", "ginger"),
    "green chili": ("কাঁচা মরিচ", "কাচা মরিচ", "োঁচো মনরচ", "োঁচামরিচ", "green chili", "green chilli"),
    "rice": ("চাল", "ধান", "rice"),
    "sugar": ("চিনি", "sugar"),
    "soybean oil": ("সয়াবিন তেল", "সয়াবিন তেল", "soybean oil", "soyabean oil"),
    "egg": ("ডিম", "egg"),
    "beef": ("গরুর মাংস", "গরু", "beef"),
    "mutton": ("খাসির মাংস", "ছাগলের মাংস", "mutton"),
    "bottle gourd": ("লাউ", "লাও", "িোউ", "bottle gourd", "lau"),
    "ash gourd": ("চালকুমড়া", "চালকুমড়া", "চালকুমড়ো", "চালকুমড়ো", "চোিকুমড়ো", "ash gourd"),
}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class WeatherResult:
    available: bool
    location: str = ""
    temperature_c: Optional[float] = None
    description: str = ""
    humidity_percent: Optional[int] = None
    rain_expected: Optional[bool] = None
    error: str = ""


@dataclass
class PriceResult:
    available: bool
    crop: str
    price_bdt_per_kg: Optional[float]
    price_range: str
    market: str
    date: str
    source: str
    division_note: str = ""
    error: str = ""
    unit: str = ""


@dataclass(frozen=True)
class DamReport:
    report_date: date
    division: str
    pdf_urls: tuple[str, ...]


@dataclass
class MarketPriceItem:
    """One product row from the selected PDF's current retail column."""

    product: str
    price_range: str
    unit: str = ""
    category: str = ""


@dataclass
class MarketResult:
    """Complete current-retail report for one division and report date."""

    available: bool
    division: str
    date: str
    source: str
    items: list[MarketPriceItem]
    omitted_count: int = 0
    market: str = "DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)"
    division_note: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

_TLS_FALLBACK_NOTICE_EMITTED = False
if getattr(config, "DAM_ALLOW_INSECURE_TLS_FALLBACK", False):
    # The fallback is explicit in .env. Suppress urllib3's duplicate warning;
    # KrishokBot emits one concise, actionable notice instead.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "KrishokBot/1.0 (+https://dam.gov.bd/)",
        "Accept-Language": "bn-BD,bn;q=0.9,en;q=0.8",
        "Accept": "text/html,application/pdf,application/xhtml+xml,*/*;q=0.8",
        "Connection": "keep-alive",
    }
)


def _dam_verify_setting() -> bool | str:
    if config.DAM_CA_BUNDLE:
        return config.DAM_CA_BUNDLE
    return bool(config.DAM_TLS_VERIFY)


def _request_get(
    url: str,
    timeout: float,
    *,
    params: Optional[dict[str, Any]] = None,
    retries: int = 0,
    verify: bool | str = True,
    allow_insecure_tls_fallback: bool = False,
) -> requests.Response:
    """GET with bounded retries and explicit TLS behavior.

    TLS verification is never disabled implicitly. A deployment that needs an
    emergency compatibility fallback must opt in through configuration; the
    fallback is logged as an error so it cannot be mistaken for a secure path.
    """
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = _SESSION.get(url, params=params, timeout=timeout, verify=verify)
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as exc:
            last_error = exc
            if not allow_insecure_tls_fallback or verify is False:
                raise
            global _TLS_FALLBACK_NOTICE_EMITTED
            if config.DAM_ALLOW_INSECURE_TLS_FALLBACK:
                if not _TLS_FALLBACK_NOTICE_EMITTED:
                    LOGGER.warning(
                        "DAM certificate chain is incomplete; using the explicitly "
                        "enabled compatibility fallback. Configure "
                        "KRISHOKBOT_DAM_CA_BUNDLE for verified TLS."
                    )
                    _TLS_FALLBACK_NOTICE_EMITTED = True
                insecure = _SESSION.get(url, params=params, timeout=timeout, verify=False)
                insecure.raise_for_status()
                return insecure
            LOGGER.warning(
                "DAM TLS verification failed for %s; request aborted. Configure "
                "KRISHOKBOT_DAM_CA_BUNDLE or explicitly enable the compatibility fallback.",
                url,
            )
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                raise
            delay = min(float(config.DAM_RETRY_BACKOFF_SECONDS) * (2 ** attempt), 8.0)
            LOGGER.warning(
                "HTTP request failed (%d/%d) for %s: %s; retrying in %.1fs",
                attempt + 1, retries + 1, url, exc, delay,
            )
            if delay:
                time.sleep(delay)
    raise RuntimeError(str(last_error or "HTTP request failed"))


def _weather_get(url: str, timeout: float, *, params: dict[str, Any], retries: int) -> requests.Response:
    """Weather requests use normal certificate verification."""
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = _SESSION.get(url, params=params, timeout=timeout, verify=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(min(2.0 ** attempt, 4.0))
    raise RuntimeError(str(last_error or "Weather request failed"))


# ---------------------------------------------------------------------------
# Weather — preserved behavior with safer error handling
# ---------------------------------------------------------------------------


def _coordinates_for(location: str) -> Optional[tuple[float, float]]:
    return BD_LOCATIONS.get(normalize_text(location).lower())


def _geocode_bangladesh(location: str) -> Optional[tuple[float, float]]:
    response = _weather_get(
        config.OPENWEATHER_GEOCODING_URL,
        config.WEATHER_TIMEOUT_SECONDS,
        params={
            "q": f"{location},{config.WEATHER_COUNTRY_CODE}",
            "limit": 1,
            "appid": config.OPENWEATHER_API_KEY,
        },
        retries=config.WEATHER_MAX_RETRIES,
    )
    items = response.json()
    if not isinstance(items, list) or not items:
        return None
    item = items[0]
    country = str(item.get("country", "")).upper()
    if country and country != config.WEATHER_COUNTRY_CODE:
        return None
    return float(item["lat"]), float(item["lon"])


def get_weather(location: str = "Barishal") -> WeatherResult:
    """Return current weather for a known or geocoded Bangladesh location."""
    location = (location or "").strip()
    if not location:
        return WeatherResult(available=False, error="No location provided.")
    api_key = config.OPENWEATHER_API_KEY.strip()
    if not api_key:
        return WeatherResult(
            available=False,
            location=location,
            error="OPENWEATHER_API_KEY is not configured. Check .env.",
        )

    try:
        coords = _coordinates_for(location) or _geocode_bangladesh(location)
        if coords is None:
            return WeatherResult(
                available=False,
                location=location,
                error=f"Location '{location}' was not found in Bangladesh.",
            )
        response = _weather_get(
            config.OPENWEATHER_BASE_URL,
            config.WEATHER_TIMEOUT_SECONDS,
            params={
                "lat": coords[0],
                "lon": coords[1],
                "appid": api_key,
                "units": "metric",
                "lang": "bn",
            },
            retries=config.WEATHER_MAX_RETRIES,
        )
        data = response.json()
        main = data.get("main") or {}
        weather_items = data.get("weather") or []
        if not main:
            raise ValueError("Weather response does not contain main data.")
        description = str(weather_items[0].get("description", "")).strip() if weather_items else ""
        return WeatherResult(
            available=True,
            location=str(data.get("name") or location),
            temperature_c=float(main["temp"]) if main.get("temp") is not None else None,
            description=description,
            humidity_percent=int(main["humidity"]) if main.get("humidity") is not None else None,
            rain_expected=bool(data.get("rain")),
        )
    except requests.Timeout:
        return WeatherResult(available=False, location=location, error="Weather request timed out.")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            error = "OpenWeatherMap API key is invalid or inactive."
        elif status == 404:
            error = f"Weather location '{location}' was not found."
        elif status == 429:
            error = "OpenWeatherMap API rate limit exceeded."
        else:
            error = f"OpenWeatherMap HTTP error (status {status})."
        LOGGER.error("Weather API error: %s", exc)
        return WeatherResult(available=False, location=location, error=error)
    except requests.RequestException as exc:
        LOGGER.error("Weather network error: %s", exc)
        return WeatherResult(available=False, location=location, error="Weather network error.")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        LOGGER.error("Malformed weather response: %s", exc)
        return WeatherResult(available=False, location=location, error="Unexpected weather response format.")


# ---------------------------------------------------------------------------
# Bengali normalization and price formatting
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", unicodedata.normalize("NFKC", text))


def normalize_digits(text: str) -> str:
    return normalize_unicode(text).translate(BANGLA_DIGITS)


def normalize_text(text: str) -> str:
    value = normalize_digits(text or "")
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = value.replace("।", ".")
    return re.sub(r"\s*\)\s*", ")", value)


def normalize_crop_name(crop: str) -> str:
    value = re.sub(r"\s+", " ", normalize_text(crop).lower()).strip()
    return CROP_ALIASES.get(value, value)


def normalize_division(division: Optional[str]) -> str:
    value = normalize_text(division or "").lower()
    return DIVISION_ALIASES.get(value, value)


def normalize_product_name(product: str) -> str:
    """Clean high-confidence DAM OCR variants without inventing unknown text."""
    value = normalize_text(product)
    if not value:
        return ""
    for source, replacement in PRODUCT_PHRASE_CORRECTIONS:
        value = value.replace(source, replacement)
    value = re.sub(r"^রুই মা(?=\s|$)", "রুই মাছ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"ডাল(?: ডাল)+", "ডাল", value)
    value = re.sub(r"\s*-\s*\(", " (", value)
    value = re.sub(r"\s*\(\s*", " (", value)
    value = re.sub(r"\s*\)", ")", value)
    # The attached DAM PDF visibly prints this row as রসুন (চায়না), while
    # the extracted text may begin the qualifier with a Bengali vowel mark.
    if re.fullmatch(r"রসুন\s*\(ি[^()]*\)", value):
        value = "রসুন (চায়না)"
    # The attached DAM PDF visibly prints the rui row as রুই (চাষের), while
    # pdfplumber may glue the quantity marker to a corrupted text label.
    value = re.sub(
        r"^রুই মাছ \(চাষের\)(?:\s+[0-9০-৯\s\-–—/]+)?$",
        "রুই মাছ (চাষের)",
        value,
    )
    if value.startswith("রুই মাছ (চাষের)"):
        value = "রুই মাছ (চাষের)"
    value = re.sub(r"\s*/\s*", " / ", value)
    value = re.sub(r"(?<=[\u0980-\u09FF)])(?=\d)", " ", value)
    if value.count("(") > value.count(")"):
        value += ")"
    return value.strip(" -–—")


def _canonical_division(division: Optional[str]) -> str:
    value = normalize_division(division)
    if not value:
        return normalize_division(config.DAM_DEFAULT_DIVISION) or "ঢাকা"
    return value


def _format_price(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"


def _make_price_result(
    *,
    crop: str,
    minimum: float,
    maximum: float,
    market: str,
    report_date: str,
    source: str,
    division_note: str = "",
    unit: str = "",
) -> PriceResult:
    low, high = sorted((float(minimum), float(maximum)))
    return PriceResult(
        available=True,
        crop=crop,
        price_bdt_per_kg=round((low + high) / 2.0, 2),
        price_range=f"৳{_format_price(low)} - ৳{_format_price(high)}",
        market=market,
        date=report_date,
        source=source,
        division_note=division_note,
        unit=unit,
    )


RANGE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-]\s*(\d+(?:\.\d+)?)(?![\d.])")
SINGLE_NUMBER_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")


def _parse_price_range(text: str) -> Optional[tuple[float, float]]:
    value = normalize_text(text)
    match = RANGE_RE.search(value)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        if 0 < low <= 1_000_000 and 0 < high <= 1_000_000:
            return tuple(sorted((low, high)))
    numbers = [float(m.group(1)) for m in SINGLE_NUMBER_RE.finditer(value)]
    numbers = [n for n in numbers if 0 < n <= 1_000_000]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    # Some DAM PDFs omit the dash between minimum and maximum, for example
    # ``24.00 30.00``. Accept exactly the first two adjacent values only after
    # the explicit range form failed; a cell containing two full ranges still
    # uses the first range above.
    if len(numbers) >= 2:
        low, high = numbers[0], numbers[1]
        if low > 0 and high > 0:
            return tuple(sorted((low, high)))
    return None


# ---------------------------------------------------------------------------
# DAM nationwide ticker
# ---------------------------------------------------------------------------


def _ticker_candidates() -> dict[str, tuple[str, ...]]:
    return {
        crop: CROP_VARIANTS[crop]
        for crop in TICKER_CROPS
        if crop in CROP_VARIANTS
    }


def _parse_dam_ticker() -> dict[str, tuple[float, float]]:
    """Parse the nationwide ticker; this is not used for explicit divisions."""
    try:
        response = _request_get(
            config.DAM_PRICE_SOURCE_URL,
            config.PRICE_TIMEOUT_SECONDS,
            retries=config.DAM_MAX_RETRIES,
            verify=_dam_verify_setting(),
            allow_insecure_tls_fallback=True,
        )
        text = normalize_text(BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True))
    except Exception as exc:
        LOGGER.warning("DAM ticker request failed: %s", exc)
        return {}

    results: dict[str, tuple[float, float]] = {}
    lower = text.lower()
    for crop, names in _ticker_candidates().items():
        for name in names:
            position = lower.find(normalize_text(name).lower())
            if position < 0:
                continue
            parsed = _parse_price_range(text[position:position + 300])
            if parsed:
                results[crop] = parsed
                break
    return results


# ---------------------------------------------------------------------------
# DAM static-page HTML report discovery
# ---------------------------------------------------------------------------


def _parse_report_date(value: str) -> Optional[date]:
    text = normalize_digits(value)
    # Handles ১৯-০৮-২০২৬, 19/08/2026, 19.08.2026, and mixed OCR separators.
    matches = re.findall(r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(20\d{2})(?!\d)", text)
    for day_text, month_text, year_text in matches:
        try:
            candidate = date(int(year_text), int(month_text), int(day_text))
        except ValueError:
            continue
        if 2000 <= candidate.year <= 2100:
            return candidate
    return None


def _date_is_in_text(value: str, target: date) -> bool:
    text = normalize_digits(value)
    day = str(target.day)
    month = str(target.month)
    year = str(target.year)
    pattern = rf"(?<!\d)0?{re.escape(day)}\s*[./-]\s*0?{re.escape(month)}\s*[./-]\s*{re.escape(year)}(?!\d)"
    return bool(re.search(pattern, text))


def _cell_pdf_urls(cell: Any, base_url: str) -> tuple[str, ...]:
    urls: list[str] = []
    for anchor in cell.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        if not href or href.startswith(("javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if absolute not in urls:
            urls.append(absolute)
    return tuple(urls)


def _find_dam_report_table(soup: BeautifulSoup) -> Optional[Any]:
    for table in soup.find_all("table"):
        header_rows = table.find_all("tr")[:3]
        header_text = normalize_text(" ".join(row.get_text(" ", strip=True) for row in header_rows))
        if "তারিখ" in header_text and any(normalize_text(d).replace("য়", "য়") in header_text.replace("য়", "য়") for d in DIVISIONS):
            return table
    return None


def _get_dam_reports() -> list[DamReport]:
    response = _request_get(
        config.DAM_PDF_PAGE_URL,
        config.PDF_TIMEOUT_SECONDS,
        retries=config.DAM_MAX_RETRIES,
        verify=_dam_verify_setting(),
        allow_insecure_tls_fallback=True,
    )
    soup = BeautifulSoup(response.text, "html.parser")
    table = _find_dam_report_table(soup)
    if table is None:
        raise ValueError("DAM division report table was not found on the configured page.")

    rows = table.find_all("tr")
    if not rows:
        return []

    header_cells = rows[0].find_all(["th", "td"])
    header_names = [normalize_text(cell.get_text(" ", strip=True)) for cell in header_cells]
    header_map: dict[str, int] = {}
    for index, name in enumerate(header_names):
        canonical = normalize_division(name)
        if canonical in DIVISIONS:
            header_map[canonical] = index

    missing = [division for division in DIVISIONS if division not in header_map]
    if missing:
        raise ValueError(f"DAM division columns missing from report table: {', '.join(missing)}")

    reports: list[DamReport] = []
    today = date.today()
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        report_date = _parse_report_date(cells[0].get_text(" ", strip=True))
        if report_date is None or report_date > today:
            continue
        if config.DAM_MAX_REPORT_AGE_DAYS and report_date < today - timedelta(days=config.DAM_MAX_REPORT_AGE_DAYS):
            continue

        for division, column_index in header_map.items():
            if column_index >= len(cells):
                continue
            pdf_urls = _cell_pdf_urls(cells[column_index], config.DAM_PDF_PAGE_URL)
            if pdf_urls:
                reports.append(DamReport(report_date, division, pdf_urls))
    return reports


def _latest_report_for_division(division: str) -> Optional[DamReport]:
    reports = [report for report in _get_dam_reports() if report.division == division]
    if not reports:
        return None
    # The page is a public archive, so this explicit sort prevents any future
    # row or malformed filename from overriding the actual latest valid date.
    reports.sort(key=lambda report: report.report_date, reverse=True)
    return reports[0]


# ---------------------------------------------------------------------------
# Column-aware Bengali PDF extraction
# ---------------------------------------------------------------------------


def _clean_table(table: list[list[Any]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in table:
        values = [normalize_text("" if value is None else str(value)) for value in row]
        if any(values):
            cleaned.append(values)
    return cleaned


def _extract_pdf_tables(pdf_bytes: bytes) -> list[list[list[str]]]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for DAM PDF parsing. Install it with 'pip install pdfplumber'.")
    tables: list[list[list[str]]] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            settings_list = [
                {},
                {"vertical_strategy": "text", "horizontal_strategy": "text", "text_x_tolerance": 2, "text_y_tolerance": 3},
            ]
            for settings in settings_list:
                try:
                    extracted = page.extract_tables(table_settings=settings)
                except Exception as exc:
                    LOGGER.debug("PDF table extraction strategy failed on page %d: %s", page_number, exc)
                    continue
                page_tables: list[list[list[str]]] = []
                for table in extracted:
                    cleaned = _clean_table(table)
                    if len(cleaned) < 2 or max((len(row) for row in cleaned), default=0) < 5:
                        continue
                    page_tables.append(cleaned)
                if not page_tables:
                    continue
                # The default strategy normally preserves the true row/column
                # layout. The text strategy can return a second, malformed
                # coordinate-split version of the same page; do not mix both.
                for cleaned in page_tables:
                    fingerprint = tuple(tuple(row) for row in cleaned)
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        tables.append(cleaned)
                break
    return tables


def _header_rows(table: list[list[str]], limit: int = 8) -> list[list[str]]:
    return table[: min(limit, len(table))]


def _is_current_price_header(text: str, report_date: date) -> bool:
    value = normalize_text(text).lower()
    if not _date_is_in_text(value, report_date):
        return False
    # Comparison and percentage columns also contain the current date; reject
    # those so the actual current-retail-price column remains the first match.
    comparison_markers = ("সাথে", "হ্রাস", "বৃদ্ধি", "%", "গি ম", "গি র")
    if any(marker in value for marker in comparison_markers):
        return False
    current_markers = ("আজ", "আির", "আদর", "র্াজার", "বাজার", "সর্বনিম্ন")
    return any(marker in value for marker in current_markers)


def _find_current_price_column(table: list[list[str]], report_date: date) -> Optional[int]:
    header = _header_rows(table)
    width = max((len(row) for row in header), default=0)
    for column_index in range(width):
        combined = " ".join(row[column_index] for row in header if column_index < len(row))
        if _is_current_price_header(combined, report_date):
            return column_index

    # More tolerant fallback for OCR-heavy PDFs: select the first column whose
    # header contains the report date and has no comparison/percentage marker.
    for column_index in range(width):
        combined = " ".join(row[column_index] for row in header if column_index < len(row))
        value = normalize_text(combined).lower()
        if _date_is_in_text(value, report_date) and not any(
            marker in value for marker in ("সাথে", "হ্রাস", "বৃদ্ধি", "%")
        ):
            return column_index
    return None


def _is_price_like_cell(value: str) -> bool:
    text = normalize_text(value)
    if not text or "%" in text:
        return False
    return bool(re.search(r"\d", text)) and bool(re.search(r"(?:\d[\d,.]*\s*[-–—]\s*\d|\d[\d,.]*\s+\d)", text))


def _infer_continuation_price_column(
    table: list[list[str]],
    preferred_column: Optional[int],
) -> Optional[int]:
    """Reuse or infer the current-price column on a continuation table."""
    if not table:
        return preferred_column
    width = max((len(row) for row in table), default=0)
    if preferred_column is not None and preferred_column < width:
        preferred_score = sum(
            1 for row in table if preferred_column < len(row) and _is_price_like_cell(row[preferred_column])
        )
        if preferred_score >= max(1, min(3, len(table) // 4)):
            return preferred_column

    best_column: Optional[int] = None
    best_score = 0
    for column_index in range(width):
        score = sum(
            1 for row in table if column_index < len(row) and _is_price_like_cell(row[column_index])
        )
        if score > best_score:
            best_score = score
            best_column = column_index
    return best_column if best_score >= max(2, min(4, len(table) // 3)) else None


def _row_contains_crop(row: list[str], crop: str) -> bool:
    variants = [normalize_text(v).lower() for v in CROP_VARIANTS.get(crop, (crop,))]
    # Product name is normally in columns 1–3; limit the search to the left
    # side so a number in a price cell cannot create a false match.
    left_side = " ".join(row[: min(4, len(row))]).lower()
    return any(variant and variant in left_side for variant in variants)


UNIT_MARKERS = (
    "কেজি", "কেতি", "কেনি", "কেজ", "কেলি", "বেরি", "লিটার", "নিটার", "লিট",
    "নিট", "তিটার", "তিটার", "রিটার", "রিটারি", "লি.", "হালি", "হোলি", "গ্রাম",
)


def normalize_unit(unit: str) -> str:
    """Convert DAM/OCR unit variants to stable Bengali display labels."""
    value = normalize_text(unit).lower()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)

    if any(token in value for token in ("লিটার", "নিটার", "লিট", "নিট", "তিটার", "তিটার", "রিটার", "রিটারি", "লি.")):
        return "প্রতি লিটার"
    if any(token in value for token in ("হালি", "হোলি")):
        return "প্রতি হালি"
    if any(token in value for token in ("গ্রাম", "গ্োরোম")):
        return "প্রতি গ্রাম"
    if re.fullmatch(r"(?:\d+\s*)?টি", value) or re.fullmatch(r"(?:প্রতি|প্রনি|প্রতিটি|প্রনিটি)\s*টি", value):
        return "প্রতিটি"
    if any(token in value for token in ("কেজি", "কেতি", "কেনি", "কেজ")):
        return "প্রতি কেজি"
    return normalize_text(unit)


def _unit_for_product(product: str, extracted_unit: str) -> str:
    """Resolve units from both the cell and the product family.

    DAM layouts sometimes shift the unit cell or cause an entire report to
    inherit the wrong unit. Product semantics are used as a safety check, not
    as a price source: oil is per litre, piece-count produce/eggs/bananas are
    each, and other confirmed market products default to per kilogram.
    """
    canonical_product = normalize_product_name(product)
    lowered = canonical_product.lower()
    extracted = normalize_unit(extracted_unit)

    for key, unit in PRODUCT_UNIT_OVERRIDES.items():
        if canonical_product == key or canonical_product.startswith(f"{key} "):
            return unit

    oil_tokens = ("সয়াবিন তেল", "সয়াবিন তেল", "পাম তেল", "সরিষার তেল", "তেল (", "তেল-")
    fish_meat_tokens = (
        "মাছ", "রুই", "কাতলা", "পাঙ্গাস", "তেলাপিয়া", "তেলাপিয়া", "ইলিশ",
        "মুরগি", "মোরগ", "গরু", "ছাগল", "মাংস", "মোংস", "ডিম",
    )
    piece_tokens = ("ডিম", "কলা", "লাউ", "চালকুমড়া", "চালকুমড়া")
    if "ডিম" in lowered:
        return "প্রতিটি"
    if any(token in lowered for token in fish_meat_tokens):
        return "প্রতি কেজি"
    if any(token in lowered for token in oil_tokens):
        return "প্রতি লিটার"
    if any(token in lowered for token in piece_tokens):
        return "প্রতিটি"
    if extracted == "প্রতি হালি":
        return extracted
    if extracted in {"প্রতি লিটার", "প্রতি কেজি", "প্রতি গ্রাম", "প্রতিটি"}:
        # If a non-oil row was falsely labelled litre by a shifted layout,
        # reject that label rather than showing an obviously wrong unit.
        return "প্রতি কেজি" if extracted == "প্রতি লিটার" else extracted
    return "প্রতি কেজি" if canonical_product else extracted


def _extract_row_unit(row: list[str], current_column: int) -> str:
    for value in row[: min(current_column, 5)]:
        normalized = normalize_text(value)
        lower = normalized.lower()
        if normalized and _looks_like_unit(normalized):
            return normalize_unit(normalized)
    return ""


def _parse_pdf_crop_price(pdf_bytes: bytes, crop: str, report_date: date) -> Optional[tuple[float, float, str]]:
    tables = _extract_pdf_tables(pdf_bytes)
    inherited_column: Optional[int] = None
    for table in tables:
        current_column = _find_current_price_column(table, report_date)
        if current_column is None:
            current_column = _infer_continuation_price_column(table, inherited_column)
        if current_column is None:
            continue
        inherited_column = current_column
        inherited_unit = ""
        for row in table:
            if current_column >= len(row) or not _row_contains_crop(row, crop):
                continue
            price = _parse_price_range(row[current_column])
            if price is None:
                continue
            product_name, _ = _product_and_category_from_row(row, current_column)
            extracted_unit = _extract_row_unit(row, current_column)
            if extracted_unit:
                inherited_unit = extracted_unit
            unit = _unit_for_product(product_name, extracted_unit or inherited_unit)
            return price[0], price[1], unit
    return None


def _looks_like_unit(value: str) -> bool:
    normalized = normalize_text(value).lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in UNIT_MARKERS):
        return True
    return bool(
        re.fullmatch(r"(?:\d+\s*)?টি", normalized)
        or re.fullmatch(r"(?:প্রতি|প্রনি|প্রতিটি|প্রনিটি)\s*টি", normalized)
        or re.fullmatch(r"(?:\d+\s*)?(?:লি\.?|লিটার|নিটার|তিটার|তিটার|রিটার|রিটারি|বেরি)", normalized)
    )


DITTO_MARKERS = {"", ",,", '"', "”", "''", "\"\""}
HEADER_PRODUCT_NOISE = {
    "পরের নাম",
    "পণ্যের নাম",
    "পরের িোম",
    "পলেি নাম",
    "পতরমাপ",
    "পরের প্রকার",
    "পণ্যের প্রকার",
    "পনরমাপ",
    "পনরমোপ",
    "পরিমাপ",
}


def _product_and_category_from_row(row: list[str], current_column: int) -> tuple[str, str]:
    """Handle shifted columns, repeated-cell markers, and header rows."""
    left = row[: min(current_column, 4)]
    if not left:
        return "", ""

    product_index = 2 if len(row) > 2 and row[2] and not _looks_like_unit(row[2]) else 1
    if product_index >= len(row):
        return "", ""

    raw_product = normalize_product_name(row[product_index]).strip("-–— ")
    product = raw_product
    if product in DITTO_MARKERS:
        # In Rajshahi-style tables, the previous category cell carries the
        # actual product name when the product column contains `,,`.
        fallback_indices = (1, 0) if product_index == 2 else (0,)
        for fallback_index in fallback_indices:
            if fallback_index >= len(row):
                continue
            candidate = normalize_product_name(row[fallback_index]).strip("-–— ")
            if candidate and candidate not in DITTO_MARKERS and not candidate.isdigit():
                product = candidate
                break
    bengali_letters = sum("\u0980" <= char <= "\u09ff" for char in product)
    if not product or product.isdigit() or _looks_like_unit(product) or bengali_letters < 2:
        return "", ""
    compact_product = re.sub(r"\s+", "", product)
    if product in HEADER_PRODUCT_NOISE or compact_product in {re.sub(r"\s+", "", value) for value in HEADER_PRODUCT_NOISE}:
        return "", ""

    category = ""
    if product_index == 2 and len(row) > 1:
        candidate = normalize_product_name(row[1]).strip("-–— ")
        if (
            candidate
            and candidate not in DITTO_MARKERS
            and candidate not in HEADER_PRODUCT_NOISE
            and not candidate.isdigit()
            and not _looks_like_unit(candidate)
        ):
            category = candidate

    # Some layouts store a broad category in column 1 and a qualifier in
    # column 2. Recombine only known qualifier families; do not concatenate
    # arbitrary OCR fragments.
    qualifier_tokens = (
        "খোলা", "প্যাকেট", "দেশী", "দেশি", "চায়না", "চায়না", "আমদানিকৃত",
        "উন্নত", "মোটা", "সরু", "চাষের", "জীবন্ত", "সোনালী", "ব্রয়লার",
        "লম্বা", "জালি", "সাদা", "ক্যান",
    )
    if category and product:
        category_clean = normalize_product_name(category).strip()
        if category_clean and (category_clean == product or product.startswith(f"{category_clean} ") or product.startswith(f"{category_clean}(")):
            category = ""
        elif category_clean in {"ডাল", "মসুর ডাল", "মুগ ডাল"} and product.startswith(("মসুর", "মুগ", "মাষকলাই", "কেসারি")):
            if "ডাল" not in product:
                product = product.replace(" (", " ডাল (", 1) if " (" in product else f"{product} ডাল"
            category = ""
        elif any(token in product.lower() for token in qualifier_tokens):
            product = f"{category_clean} ({product})"
            category = ""
    product = normalize_product_name(product).strip("-–— ")
    return product, category


def _format_range(price: tuple[float, float]) -> str:
    low, high = sorted(price)
    if low == high:
        return f"৳{_format_price(low)}"
    return f"৳{_format_price(low)} - ৳{_format_price(high)}"


def _looks_like_ambiguous_product(product: str) -> bool:
    """Reject OCR text that is not safely identifiable as a market product."""
    text = normalize_product_name(product).strip()
    if not text:
        return True

    # The user-facing report keeps only the clean domestic garlic row. The
    # attached PDF's China garlic row is too OCR-unstable for reliable display.
    if text in {"রসুন (চায়না)", "রসুন (চায়না)"}:
        return True
    if text.startswith("রসুন") and "দেশী" not in text:
        return True

    # Bengali vowel signs cannot legally begin a normal word. This catches
    # strings such as ``েোেররোি`` without attempting to repair them.
    if text[0] in "ািীুূৃেৈোৌ্ৎংঃঁ়" or text[0].isdigit():
        return True
    # A Bengali vowel mark immediately after a separator or closing bracket
    # is a strong OCR layout artifact, not a reliable product qualifier.
    if re.search(r"(?:\)|/|:)\s*[ািীুূৃেৈোৌ্ৎংঃঁ়]", text):
        return True

    base_letters = sum("\u0980" <= ch <= "\u09ff" and ch not in "ািীুূৃেৈোৌ্ৎংঃঁ়" for ch in text)
    marks = sum(ch in "ািীুূৃেৈোৌ্ৎংঃঁ়" for ch in text)
    if base_letters < 2 or marks > (base_letters * 2 + 3):
        return True

    compact_text = re.sub(r"\s+", "", text)
    if compact_text in {
        "রসুন(িোয়িো)",
        "রসুন(িোয়িো)",
        "রুই(িোরের)",
        "রুইমাছ(িোরের)",
    }:
        return True

    lowered = text.casefold()
    if not any(token in lowered for token in PRODUCT_CONFIDENCE_TOKENS):
        return True

    # Reject numeric layout debris glued after a parenthetical product label,
    # e.g. ``রুই মাছ (িোরের) 1 1 - 4 2``.
    if re.search(r"\)\s*[0-9০-৯][0-9০-৯\s\-–—]*$", text):
        return True

    # Parenthetical qualifiers that begin with a Bengali vowel sign are OCR
    # fragments, not reliable qualifiers. This hides rows such as
    # ``রসুন (িোয়িো)`` without guessing the intended variety.
    for qualifier in re.findall(r"\(([^()]*)\)", text):
        qualifier = qualifier.strip()
        if qualifier and qualifier[0] in "ািীুূৃেৈোৌ্ৎংঃঁ়":
            return True

    # A row with a parenthetical qualifier must have a recognizable product
    # root before the qualifier. This hides OCR fragments such as
    # ``নিনি (আমদোিীকৃি...)`` and ``কিল (সয়াবিন...)`` without guessing a
    # replacement name. Verified roots are intentionally conservative.
    known_roots = (
        "আটা", "মসুর", "মুগ", "মাষকলাই", "কেসারি", "ছোলা", "সয়াবিন", "সয়াবিন",
        "পাম", "সরিষার", "চিনি", "পেঁয়াজ", "পেঁয়াজ", "রসুন", "আদা", "মরিচ",
        "আলু", "বেগুন", "পেঁপে", "কুমড়া", "কুমড়া", "চিচিঙ্গা", "ধুন্দুল", "কচু",
        "বরবটি", "লাউ", "চালকুমড়া", "চালকুমড়া", "শসা", "উচ্ছে", "করল্লা", "ঝিঙ্গা",
        "ঢেঁড়স", "পটল", "চাল", "আপেল", "পেয়ারা", "পেয়ারা", "আম", "কলা", "মাছ",
        "রুই", "কাতলা", "ইলিশ", "পাঙ্গাস", "তেলাপিয়া", "তেলাপিয়া", "মাংস", "গরু",
        "ছাগল", "মুরগি", "মোরগ", "ডিম",
    )
    if "(" in text:
        root = text.split("(", 1)[0].strip(" -–—")
        if not any(root == known or root.startswith(f"{known} ") for known in known_roots):
            return True
        qualifier = text.split("(", 1)[1].rsplit(")", 1)[0].strip()
        allowed_qualifiers = {
            "আটা": {"প্যাকেট-সাদা", "প্যাকেট", "খোলা"},
            "রসুন": {"দেশী", "দেশি"},
        }
        if root in allowed_qualifiers and qualifier not in allowed_qualifiers[root]:
            return True
    return False


def _looks_like_omitted_product_row(row: list[str], current_column: int) -> bool:
    """Return True when a priced row has meaningful but unreadable product text.

    Header rows, unit-only cells, and numeric layout artifacts are excluded so
    the omission count is aimed at farmer-facing OCR ambiguity rather than
    every non-product row encountered by pdfplumber.
    """
    left_side = [normalize_product_name(value).strip("-–— ") for value in row[: min(current_column, 4)]]
    meaningful = [value for value in left_side if value and not value.isdigit() and not _looks_like_unit(value)]
    if not meaningful:
        return False
    joined = " ".join(meaningful)
    compact = re.sub(r"\s+", "", joined)
    if joined in HEADER_PRODUCT_NOISE or compact in {
        re.sub(r"\s+", "", value) for value in HEADER_PRODUCT_NOISE
    }:
        return False
    if _product_and_category_from_row(row, current_column)[0]:
        return False
    return len(compact) >= 2


def _extract_full_market_items_with_omissions(
    pdf_bytes: bytes, report_date: date
) -> tuple[list[MarketPriceItem], int]:
    """Extract product rows and count priced rows omitted for OCR ambiguity."""
    items: list[MarketPriceItem] = []
    omitted_count = 0
    seen: set[str] = set()
    inherited_column: Optional[int] = None
    for table in _extract_pdf_tables(pdf_bytes):
        current_column = _find_current_price_column(table, report_date)
        if current_column is None:
            current_column = _infer_continuation_price_column(table, inherited_column)
        if current_column is None:
            continue
        inherited_column = current_column
        inherited_unit = ""
        for row in table:
            if current_column >= len(row):
                continue
            parsed_price = _parse_price_range(row[current_column])
            if parsed_price is None:
                continue
            product, category = _product_and_category_from_row(row, current_column)
            if not product:
                if _looks_like_omitted_product_row(row, current_column):
                    omitted_count += 1
                continue
            if _looks_like_ambiguous_product(product):
                omitted_count += 1
                continue
            extracted_unit = _extract_row_unit(row, current_column)
            if extracted_unit:
                inherited_unit = extracted_unit
            unit = _unit_for_product(product, extracted_unit or inherited_unit)
            item = MarketPriceItem(
                product=product,
                price_range=_format_range(parsed_price),
                unit=unit,
                category=category,
            )
            identity = normalize_product_name(item.product).casefold()
            if identity not in seen:
                seen.add(identity)
                items.append(item)
    return items, omitted_count


def _extract_full_market_items(pdf_bytes: bytes, report_date: date) -> list[MarketPriceItem]:
    """Backward-compatible wrapper returning only extracted market items."""
    items, _ = _extract_full_market_items_with_omissions(pdf_bytes, report_date)
    return items


# ---------------------------------------------------------------------------
# PDF download and division-specific lookup
# ---------------------------------------------------------------------------


def _download_pdf(pdf_url: str) -> bytes:
    response = _request_get(
        pdf_url,
        config.PDF_TIMEOUT_SECONDS,
        retries=config.DAM_MAX_RETRIES,
        verify=_dam_verify_setting(),
        allow_insecure_tls_fallback=True,
    )
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > config.DAM_MAX_PDF_BYTES:
                raise ValueError("DAM PDF exceeds the configured size limit.")
        except ValueError as exc:
            if "size limit" in str(exc):
                raise
    content = response.content
    if len(content) > config.DAM_MAX_PDF_BYTES:
        raise ValueError("DAM PDF exceeds the configured size limit.")
    if not content.lstrip().startswith(b"%PDF"):
        raise ValueError(f"DAM download did not return a PDF: {pdf_url}")
    return content


def _unavailable(crop: str, source: str, error: str, *, division_note: str = "") -> PriceResult:
    return PriceResult(
        available=False,
        crop=crop,
        price_bdt_per_kg=None,
        price_range="",
        market="",
        date="",
        source=source,
        division_note=division_note,
        error=error,
    )


def _get_division_pdf_price(crop: str, division: Optional[str]) -> PriceResult:
    canonical_crop = normalize_crop_name(crop)
    requested_division = normalize_division(division)
    selected_division = _canonical_division(division)

    if selected_division not in DIVISIONS:
        return _unavailable(
            canonical_crop,
            config.DAM_PDF_PAGE_URL,
            f"অজানা বিভাগ: {division}",
        )

    note = ""
    if not requested_division:
        note = f"বিভাগ উল্লেখ করা হয়নি — {selected_division} বিভাগ default হিসেবে ব্যবহৃত হয়েছে।"
    note = f"{note} বিভাগ: {selected_division}".strip()

    try:
        report = _latest_report_for_division(selected_division)
        if report is None:
            return _unavailable(
                canonical_crop,
                config.DAM_PDF_PAGE_URL,
                f"{selected_division} বিভাগের কোনো বৈধ PDF পাওয়া যায়নি।",
                division_note=note,
            )

        # A malformed page may expose more than one link in the same cell. Try
        # all same-date links, but never silently fall back to an older date.
        last_error = ""
        for pdf_url in report.pdf_urls:
            try:
                pdf_bytes = _download_pdf(pdf_url)
                parsed = _parse_pdf_crop_price(pdf_bytes, canonical_crop, report.report_date)
                if parsed is None:
                    last_error = f"{canonical_crop} row or current-price column not found"
                    continue
                low, high, unit = parsed
                return _make_price_result(
                    crop=canonical_crop,
                    minimum=low,
                    maximum=high,
                    market="DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)",
                    report_date=report.report_date.isoformat(),
                    source=pdf_url,
                    division_note=note,
                    unit=unit,
                )
            except Exception as exc:
                last_error = str(exc)
                LOGGER.warning("DAM PDF candidate failed for %s/%s: %s", selected_division, canonical_crop, exc)

        return _unavailable(
            canonical_crop,
            config.DAM_PDF_PAGE_URL,
            f"{selected_division} বিভাগের সর্বশেষ PDF-এ {canonical_crop} এর দাম পাওয়া যায়নি।",
            division_note=note,
        )
    except requests.exceptions.SSLError:
        return _unavailable(
            canonical_crop,
            config.DAM_PDF_PAGE_URL,
            "DAM সাইটের TLS certificate যাচাই করা যায়নি। trusted CA bundle configure করুন; insecure fallback disabled.",
            division_note=note,
        )
    except requests.RequestException as exc:
        LOGGER.error("DAM division lookup network failure: %s", exc)
        return _unavailable(
            canonical_crop,
            config.DAM_PDF_PAGE_URL,
            "DAM সাইটে যোগাযোগ করা যায়নি।",
            division_note=note,
        )
    except Exception as exc:
        LOGGER.exception("DAM division lookup failed for %s/%s", selected_division, canonical_crop)
        return _unavailable(
            canonical_crop,
            config.DAM_PDF_PAGE_URL,
            f"DAM PDF বিশ্লেষণ করা যায়নি: {exc}",
            division_note=note,
        )


def _get_full_market_price_report(division: Optional[str]) -> MarketResult:
    requested_division = normalize_division(division)
    selected_division = _canonical_division(division)
    note = (
        f"বিভাগ উল্লেখ করা হয়নি — {selected_division} বিভাগ default হিসেবে ব্যবহৃত হয়েছে।"
        if not requested_division
        else ""
    )

    if selected_division not in DIVISIONS:
        return MarketResult(
            available=False,
            division=selected_division,
            date="",
            source=config.DAM_PDF_PAGE_URL,
            items=[],
            division_note=note,
            error=f"অজানা বিভাগ: {division}",
        )

    try:
        report = _latest_report_for_division(selected_division)
        if report is None:
            return MarketResult(
                available=False,
                division=selected_division,
                date="",
                source=config.DAM_PDF_PAGE_URL,
                items=[],
                division_note=note,
                error=f"{selected_division} বিভাগের কোনো বৈধ PDF পাওয়া যায়নি।",
            )

        last_error = ""
        for pdf_url in report.pdf_urls:
            try:
                pdf_bytes = _download_pdf(pdf_url)
                items, omitted_count = _extract_full_market_items_with_omissions(
                    pdf_bytes, report.report_date
                )
                if not items:
                    last_error = "সর্বশেষ PDF-এর current retail column থেকে কোনো product row পাওয়া যায়নি।"
                    continue
                return MarketResult(
                    available=True,
                    division=selected_division,
                    date=report.report_date.isoformat(),
                    source=pdf_url,
                    items=items,
                    omitted_count=omitted_count,
                    division_note=note,
                )
            except Exception as exc:
                last_error = str(exc)
                LOGGER.warning("Full DAM report candidate failed for %s: %s", selected_division, exc)

        return MarketResult(
            available=False,
            division=selected_division,
            date=report.report_date.isoformat(),
            source=config.DAM_PDF_PAGE_URL,
            items=[],
            division_note=note,
            error=last_error or f"{selected_division} বিভাগের বাজারদর পাওয়া যায়নি।",
        )
    except requests.exceptions.SSLError:
        return MarketResult(
            available=False,
            division=selected_division,
            date="",
            source=config.DAM_PDF_PAGE_URL,
            items=[],
            division_note=note,
            error="DAM সাইটের TLS certificate যাচাই করা যায়নি।",
        )
    except requests.RequestException:
        return MarketResult(
            available=False,
            division=selected_division,
            date="",
            source=config.DAM_PDF_PAGE_URL,
            items=[],
            division_note=note,
            error="DAM সাইটে যোগাযোগ করা যায়নি।",
        )
    except Exception as exc:
        LOGGER.exception("Full DAM report lookup failed for %s", selected_division)
        return MarketResult(
            available=False,
            division=selected_division,
            date="",
            source=config.DAM_PDF_PAGE_URL,
            items=[],
            division_note=note,
            error=f"DAM PDF বিশ্লেষণ করা যায়নি: {exc}",
        )


# ---------------------------------------------------------------------------
# Public price APIs
# ---------------------------------------------------------------------------


def get_crop_price(crop: str, division: Optional[str] = None) -> PriceResult:
    """Return a live price.

    If ``division`` is omitted, the configured default ঢাকা division PDF is
    used. If a division is supplied, only that division's PDF is used; the
    nationwide ticker is never used for farmer-facing answers.
    """
    canonical_crop = normalize_crop_name(crop)
    if not canonical_crop:
        return _unavailable("", config.DAM_PDF_PAGE_URL, "পণ্যের নাম দেওয়া হয়নি।")

    # No division means the configured default division (ঢাকা), never the
    # nationwide ticker. This keeps farmer answers geographically consistent.
    if division is None:
        return _get_division_pdf_price(canonical_crop, None)

    return _get_division_pdf_price(canonical_crop, division)


# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------

_CACHE_LOCK = Lock()


def _cache_file() -> Path:
    path = Path(config.PRICE_CACHE_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cache() -> dict[str, Any]:
    try:
        with _cache_file().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def _save_cache(data: dict[str, Any]) -> None:
    path = _cache_file()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
    temporary.replace(path)


def _cache_key(crop: str, division: Optional[str]) -> str:
    # Versioned key invalidates older nationwide-ticker cache entries after the
    # default path was changed to the ঢাকা division PDF.
    return f"v2__{normalize_crop_name(crop)}__{_canonical_division(division)}"


def _get_cached_price(crop: str, division: Optional[str]) -> Optional[PriceResult]:
    item = _load_cache().get(_cache_key(crop, division))
    if not isinstance(item, dict):
        return None
    try:
        saved_at = datetime.fromisoformat(str(item["saved_at"]))
        age_hours = (datetime.now() - saved_at).total_seconds() / 3600.0
        if age_hours < 0 or age_hours > config.PRICE_CACHE_MAX_AGE_HOURS:
            return None
        result_data = item["result"]
        if not isinstance(result_data, dict):
            return None
        result = PriceResult(**result_data)
        result.unit = normalize_unit(result.unit)
        if result.date and _parse_report_date(result.date) and _parse_report_date(result.date) > date.today():
            return None
        return result
    except (KeyError, TypeError, ValueError):
        return None


def _cache_result(crop: str, division: Optional[str], result: PriceResult) -> None:
    if not result.available:
        return
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[_cache_key(crop, division)] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "result": asdict(result),
        }
        _save_cache(cache)


MARKET_CACHE_SCHEMA = "v12"


def _market_cache_key(division: Optional[str]) -> str:
    # Versioning prevents pre-normalization full-market entries from being
    # returned after the OCR/unit schema changes.
    return f"market_{MARKET_CACHE_SCHEMA}__{_canonical_division(division)}"


def _get_cached_market(division: Optional[str]) -> Optional[MarketResult]:
    item = _load_cache().get(_market_cache_key(division))
    if not isinstance(item, dict):
        return None
    try:
        saved_at = datetime.fromisoformat(str(item["saved_at"]))
        age_hours = (datetime.now() - saved_at).total_seconds() / 3600.0
        if age_hours < 0 or age_hours > config.PRICE_CACHE_MAX_AGE_HOURS:
            return None
        result_data = item["result"]
        if not isinstance(result_data, dict):
            return None
        report_date = str(result_data.get("date", ""))
        if report_date:
            parsed_date = date.fromisoformat(report_date)
            if parsed_date > date.today():
                return None
        raw_items = result_data.get("items")
        if not isinstance(raw_items, list):
            return None
        items: list[MarketPriceItem] = []
        for entry in raw_items:
            if not isinstance(entry, dict) or not entry.get("product") or not entry.get("price_range"):
                continue
            product = normalize_product_name(str(entry.get("product", "")))
            if _looks_like_ambiguous_product(product):
                continue
            items.append(
                MarketPriceItem(
                    product=product,
                    price_range=str(entry.get("price_range", "")),
                    unit=_unit_for_product(product, str(entry.get("unit", ""))),
                    category=normalize_product_name(str(entry.get("category", ""))),
                )
            )
        cached_note = str(result_data.get("division_note", "")).strip()
        if not division:
            default_note = f"বিভাগ উল্লেখ করা হয়নি — {_canonical_division(None)} বিভাগ default হিসেবে ব্যবহৃত হয়েছে।"
            if default_note not in cached_note:
                cached_note = f"{default_note} {cached_note}".strip()
        else:
            default_note = f"বিভাগ উল্লেখ করা হয়নি — {_canonical_division(None)} বিভাগ default হিসেবে ব্যবহৃত হয়েছে।"
            cached_note = cached_note.replace(default_note, "").strip()
        return MarketResult(
            available=bool(result_data.get("available")),
            division=str(result_data.get("division", _canonical_division(division))),
            date=report_date,
            source=str(result_data.get("source", config.DAM_PDF_PAGE_URL)),
                items=items,
                omitted_count=int(result_data.get("omitted_count", 0) or 0),
                market=str(result_data.get("market", "DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)")),
                division_note=f"[cache, {report_date}] {cached_note}".strip(),
            error=str(result_data.get("error", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _cache_market(division: Optional[str], result: MarketResult) -> None:
    if not result.available:
        return
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[_market_cache_key(division)] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "result": asdict(result),
        }
        _save_cache(cache)


def get_full_market_prices(
    division: Optional[str] = None, *, force_refresh: bool = False
) -> MarketResult:
    """Return all products from the selected division's latest DAM PDF.

    This is the API to call when the user asks for the whole market report
    without naming one product. No nationwide ticker is used for this flow.
    ``force_refresh=True`` is used for farmer-facing live queries so a cached
    report cannot hide a newer DAM publication.
    """
    if not force_refresh:
        cached = _get_cached_market(division)
        if cached:
            return cached
    result = _get_full_market_price_report(division)
    if not force_refresh:
        _cache_market(division, result)
    return result


# Friendly aliases for routers and integrations that use different naming.
get_market_prices = get_full_market_prices
get_market_price_report = get_full_market_prices


def get_crop_price_smart(crop: str, division: Optional[str] = None) -> PriceResult:
    """Cache-aware user-facing price lookup with strict division semantics."""
    if division is None:
        cached = _get_cached_price(crop, None)
        if cached:
            cached.division_note = f"[cache, {cached.date}] {cached.division_note}".strip()
            return cached
        result = get_crop_price(crop, None)
        _cache_result(crop, None, result)
        return result

    cached = _get_cached_price(crop, division)
    if cached:
        cached.division_note = f"[cache, {cached.date}] {cached.division_note}".strip()
        return cached
    result = get_crop_price(crop, division)
    _cache_result(crop, division, result)
    return result


def get_all_crop_prices(division: Optional[str] = None) -> dict[str, PriceResult]:
    return {crop: get_crop_price_smart(crop, division) for crop in DEFAULT_CROPS}


def run_price_cache_refresh(division: Optional[str] = None) -> dict[str, PriceResult]:
    results: dict[str, PriceResult] = {}
    for crop in ("potato", "tomato"):
        results[crop] = get_crop_price(crop, division or _canonical_division(None))
        _cache_result(crop, division, results[crop])
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _print_weather(title: str, result: WeatherResult) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"available      : {result.available}")
    print(f"location       : {result.location}")
    print(f"temperature °C : {result.temperature_c}")
    print(f"description    : {result.description}")
    print(f"humidity %     : {result.humidity_percent}")
    print(f"rain expected  : {result.rain_expected}")
    print(f"error          : {result.error}")


def _print_price(title: str, result: PriceResult) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"available      : {result.available}")
    print(f"crop           : {result.crop}")
    print(f"price          : {result.price_bdt_per_kg}")
    print(f"range          : {result.price_range}")
    print(f"unit           : {result.unit}")
    print(f"market         : {result.market}")
    print(f"date           : {result.date}")
    print(f"source         : {result.source}")
    print(f"division note  : {result.division_note}")
    print(f"error          : {result.error}")


def _print_market(title: str, result: MarketResult, *, sample_size: int = 5) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"available      : {result.available}")
    print(f"division       : {result.division}")
    print(f"date           : {result.date}")
    print(f"items          : {len(result.items)}")
    print(f"source         : {result.source}")
    print(f"division note  : {result.division_note}")
    print(f"error          : {result.error}")
    for item in result.items[:sample_size]:
        unit = f" {item.unit}" if item.unit else ""
        print(f"  - {item.product}: {item.price_range}{unit}")


def _run_tests() -> None:
    print("=" * 70)
    print("KRISHOKBOT TOOL DIAGNOSTICS")
    print("=" * 70)
    print(f"Weather key loaded : {bool(config.OPENWEATHER_API_KEY)}")
    print(f"Weather key length : {len(config.OPENWEATHER_API_KEY)}")
    print(f"DAM default        : {_canonical_division(None)}")
    _print_weather("Weather / Barishal", get_weather("Barishal"))
    _print_price("Potato / Dhaka default", get_crop_price_smart("আলু"))
    _print_price("Potato / Barishal", get_crop_price_smart("আলু", "বরিশাল"))
    _print_price("Tomato / Sylhet", get_crop_price_smart("টমেটো", "সিলেট"))
    _print_market("Full market / Dhaka default", get_full_market_prices())


if __name__ == "__main__":
    _run_tests()
