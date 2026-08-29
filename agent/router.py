"""agent/router.py

Routes normalized farmer questions to RAG, weather and/or price tools.
The LLM is preferred, but deterministic keyword routing remains safe when
providers are unavailable or quota-limited.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, ValidationError

from agent import config
from agent.llm_provider import LLMError, LLMProvider, get_llm
from agent.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    needs_rag: bool = True
    needs_weather: bool = False
    needs_price: bool = False
    location: Optional[str] = None
    crop: Optional[str] = None
    crops: list[str] = []
    reasoning: str = ""
    source: str = "llm"  # "llm" | "keyword_fallback"


_WEATHER_KEYWORDS = re.compile(
    r"আবহাওয়া|আবহাওয়া|বৃষ্টি|তাপমাত্রা|রোদ|abohawa|brishti|bristi|tapmatra|rod|"
    r"weather|rain|temperature|forecast",
    re.IGNORECASE,
)
_PRICE_KEYWORDS = re.compile(
    r"দাম|মূল্য|বাজারদর|বাজার দর|দর কত|bazar\s*dor|bazardor|bajar\s*dor|"
    r"bajardor|dam|daam|mullo|price|market rate|market price",
    re.IGNORECASE,
)
_RAG_KEYWORDS = re.compile(
    r"রোগ|পোকা|কীট|ব্লাইট|দাগ|পাতা|চাষ|সার|কীটনাশক|প্রতিকার|চিকিৎসা|স্প্রে|লক্ষণ|করণীয়|করনীয়|"
    r"disease|pest|blight|symptom|treatment|fertilizer|spray|cultivation|how to|"
    r"what should|rog|poka|kit|dag|pata|chash|sar|kitnashok|protikar|chikitsa|"
    r"lokkhan|koronio|koroniyo|doman|protirodh|kivabe|ki korbo|debo|dite hobe",
    re.IGNORECASE,
)
_GREETING_ONLY_PATTERN = re.compile(
    r"^(?:(?:হ্যালো|হাই|ধন্যবাদ|থ্যাঙ্কস|dhonnobad|donnobad|বিদায়|বিদায়|বাই|"
    r"আবার দেখা হবে|hello|hi|thanks|thank you|bye|goodbye|ok|okay|ঠিক আছে|শুভ দিন|"
    r"শুভরাত্রি|শুভ রাত্রি)[\s,!.?।!]+)*"
    r"(?:হ্যালো|হাই|ধন্যবাদ|থ্যাঙ্কস|dhonnobad|donnobad|বিদায়|বিদায়|বাই|আবার দেখা হবে|"
    r"hello|hi|thanks|thank you|bye|goodbye|ok|okay|ঠিক আছে|শুভ দিন|শুভরাত্রি|শুভ রাত্রি)"
    r"[\s,!.?।!]*$",
    re.IGNORECASE,
)


def _is_greeting(query: str) -> bool:
    """Return True only for short greeting/thanks/farewell-only messages."""
    text = re.sub(r"\s+", " ", str(query or "").strip())
    return bool(text) and bool(_GREETING_ONLY_PATTERN.fullmatch(text))

_DIVISION_QUERY_ALIASES = {
    "ঢাকা": "ঢাকা",
    "dhaka": "ঢাকা",
    "চট্টগ্রাম": "চট্টগ্রাম",
    "চট্টগ্রামে": "চট্টগ্রাম",
    "chattogram": "চট্টগ্রাম",
    "chittagong": "চট্টগ্রাম",
    "খুলনা": "খুলনা",
    "khulna": "খুলনা",
    "রাজশাহী": "রাজশাহী",
    "rajshahi": "রাজশাহী",
    "বরিশাল": "বরিশাল",
    "barishal": "বরিশাল",
    "barisal": "বরিশাল",
    "সিলেট": "সিলেট",
    "sylhet": "সিলেট",
    "রংপুর": "রংপুর",
    "rangpur": "রংপুর",
    "ময়মনসিংহ": "ময়মনসিংহ",
    "ময়মনসিংহ": "ময়মনসিংহ",
    "mymensingh": "ময়মনসিংহ",
}


def _keyword_location(query: str) -> Optional[str]:
    lowered = query.casefold()
    for alias, canonical in sorted(_DIVISION_QUERY_ALIASES.items(), key=lambda pair: len(pair[0]), reverse=True):
        if alias.casefold() in lowered:
            return canonical
    return None


_CROP_KEYWORDS = {
    "potato": re.compile(r"আলু|potato|alu", re.IGNORECASE),
    "tomato": re.compile(r"টমেটো|tomato|tometo", re.IGNORECASE),
    "rice": re.compile(r"ধান|চাল|rice|dhan|chal", re.IGNORECASE),
    "onion": re.compile(r"পেঁয়াজ|পেঁয়াজ|পিয়াজ|পিয়াজ|onion|peyaj|piyaj", re.IGNORECASE),
    "garlic": re.compile(r"রসুন|garlic|roshun|rosun", re.IGNORECASE),
    "masur dal": re.compile(r"মসুর(?:\s+ডাল)?|masur\s+dal|masur|lentil", re.IGNORECASE),
    "ginger": re.compile(r"আদা|ginger|ada", re.IGNORECASE),
    "green chili": re.compile(r"কাঁচা\s+মরিচ|কাচা\s+মরিচ|green\s+chil(?:i|li)|kacha\s+morich|morich", re.IGNORECASE),
}


def _keyword_fallback_crops(query: str) -> list[str]:
    """Return all explicitly named crops in stable canonical order."""
    return [crop for crop, pattern in _CROP_KEYWORDS.items() if pattern.search(query)]


def _keyword_fallback_crop(query: str) -> Optional[str]:
    crops = _keyword_fallback_crops(query)
    return crops[0] if crops else None


def _is_unclear_input(query: str) -> bool:
    """Return True for short non-agricultural or meaningless input."""
    text = re.sub(r"\s+", " ", str(query or "").strip())
    if not text or _is_greeting(text):
        return False

    # Never classify a real agricultural, weather, price, or crop query as unclear.
    if _WEATHER_KEYWORDS.search(text) or _PRICE_KEYWORDS.search(text):
        return False
    if _RAG_KEYWORDS.search(text):
        return False
    if any(pattern.search(text) for pattern in _CROP_KEYWORDS.values()):
        return False
    if len(text.split()) > 6:
        return False

    # Random English-letter input such as "dbsbjawinc jodi".
    if re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", text):
        return True

    # Short Bangla input with no agricultural signal.
    if re.fullmatch(r"[\u0980-\u09FF\s,!.?।]+", text) and len(text) <= 40:
        return True

    return False


def _is_non_rag_input(query: str) -> bool:
    return _is_greeting(query) or _is_unclear_input(query)


def _keyword_fallback_route(query: str) -> RouteDecision:
    """Route without an LLM while preserving multi-intent questions.

    Price-only and weather-only questions do not need RAG. RAG is enabled for
    explicit disease/pest/cultivation questions, and for otherwise ambiguous
    questions where a knowledge answer is safer than returning no branch.
    """
    needs_weather = bool(_WEATHER_KEYWORDS.search(query))
    needs_price = bool(_PRICE_KEYWORDS.search(query))
    explicit_rag = bool(_RAG_KEYWORDS.search(query))
    location = _keyword_location(query)

    # Do not force RAG for a clean price-only/full-market or weather-only query.
    # Keep RAG for explicit agricultural advice and for unknown intent.
    needs_rag = False if _is_non_rag_input(query) else (explicit_rag or not (needs_weather or needs_price))

    return RouteDecision(
        needs_rag=needs_rag,
        needs_weather=needs_weather,
        needs_price=needs_price,
        location=location,
        crop=(_keyword_fallback_crops(query) or [None])[0] if needs_price else None,
        crops=_keyword_fallback_crops(query) if needs_price else [],
        reasoning="keyword-based fallback (LLM routing unavailable or invalid)",
        source="keyword_fallback",
    )


def _parse_llm_json(raw_text: str) -> dict:
    """Strip common Markdown fences before parsing the router JSON."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def route(query: str, llm: Optional[LLMProvider] = None) -> RouteDecision:
    """Return a validated route and never raise to the caller."""
    if _is_non_rag_input(query):
        return RouteDecision(
            needs_rag=False,
            needs_weather=False,
            needs_price=False,
            reasoning="greeting, farewell, or unclear input",
            source="keyword_fallback",
        )
    llm = llm or get_llm()

    for attempt in range(1, config.ROUTER_MAX_LLM_ATTEMPTS + 1):
        try:
            prompt = ROUTER_USER_TEMPLATE.format(query=query)
            response = llm.generate(
                prompt,
                system=ROUTER_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.0,
                max_output_tokens=220,
            )
            parsed = _parse_llm_json(response.text)
            detected_crops = _keyword_fallback_crops(query)
            detected_location = _keyword_location(query)
            if detected_location:
                # Explicit division names always override an LLM shortcut to
                # the configured Dhaka default.
                parsed["location"] = detected_location
            if detected_crops and bool(_PRICE_KEYWORDS.search(query)):
                # Explicit crop names override an LLM shortcut to full-market.
                # This is what makes “আলু পেঁয়াজ মসুর ডালের দাম” a specific
                # multi-price request rather than a 56-row market report.
                parsed["needs_price"] = True
                parsed["crops"] = detected_crops
                parsed["crop"] = detected_crops[0]
                if not parsed.get("reasoning"):
                    parsed["reasoning"] = "explicit multi-crop price request"
            elif parsed.get("crop") and not parsed.get("crops"):
                parsed["crops"] = [parsed["crop"]]
            return RouteDecision(**parsed, source="llm")
        except (LLMError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "Router LLM attempt %d/%d failed: %s",
                attempt,
                config.ROUTER_MAX_LLM_ATTEMPTS,
                exc,
            )

    logger.warning("Router falling back to keyword-based routing for: %r", query[:60])
    return _keyword_fallback_route(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_queries = [
        "আজ বরিশালের পুরো বাজার দর দিন",
        "আজ বরিশালে আবহাওয়া কেমন থাকবে",
        "আলুর বর্তমান বাজারদর কত",
        "ধানের পোকা হলে কী করব",
        "এই বৃষ্টিতে ধানে কীটনাশক দেব কি",
    ]
    for query in test_queries:
        decision = _keyword_fallback_route(query)
        print(f"{query}\n  {decision}\n")
