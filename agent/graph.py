"""agent/graph.py

The LangGraph StateGraph tying normalizer -> router -> RAG/tools ->
synthesis together (Section 7/8 architecture).

Flow:
    START -> normalize -> route -> [rag?, weather?, price?] (fan-out,
             only branches selected by the router actually run) ->
             synthesis -> END

Image queries bypass normalize/route/rag-tool-selection and go through
`vision_node` -> optional RAG enrichment -> synthesis instead (Section
7's second branch). vision_node imports vision.detector lazily and
degrades to a clear error state if that module/model isn't available,
rather than crashing the whole graph.

NOTE ON TESTING: this file was written and syntax-checked, but NOT
executed end-to-end in the development sandbox — that requires
langgraph + a populated ChromaDB collection + LLM API keys, none of
which are available in this environment. Run `python -m agent.graph`
locally (with .env populated) to exercise it; fix whatever the real
run surfaces before considering this "done" per the project's own
completion rule (Section 3).
"""

from __future__ import annotations

import logging
import re
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.guardrails import (
    check_prompt_injection,
    enforce_uncertainty_language,
    filter_injected_chunks,
    sanitize_input,
    strip_garbled_segments,
    strip_source_dump,
)
from agent.llm_provider import LLMError, get_llm
from agent.normalizer import normalize
from agent.prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE
from agent.router import RouteDecision, route
from agent.tools import (
    MarketResult,
    PriceResult,
    WeatherResult,
    get_crop_price,
    get_full_market_prices,
    get_weather,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCATION = "Dhaka"  # used if router can't extract a location and none is supplied


class AgentState(TypedDict, total=False):
    # input
    original_input: str
    input_type: str  # "text" | "image"
    image_path: Optional[str]
    user_location: Optional[str]  # e.g. from browser geolocation, if available

    # normalization
    normalized_query: str
    detected_script: str

    # routing
    route: RouteDecision

    # retrieved context
    retrieved_chunks: list
    citations: list  # clean {source, page, chapter} dicts for UI display — never raw chunk text (Section 31)
    weather_result: WeatherResult
    price_result: PriceResult
    # Full-market report when the router found no specific crop in the query
    # (e.g. "আজকের বাজারদর কত"). The complete selected-division DAM report is
    # preserved so the UI/evaluation layer can render every extracted row.
    market_result: MarketResult
    # Compatibility field retained for callers/tests that may provide a
    # legacy predefined-crop sweep; the new generic path does not populate it.
    price_results: dict

    vision_result: dict  # {class_name, confidence, tier, bbox, ...} from vision.detector

    # output
    final_answer: str
    warnings: list[str]
    errors: list[str]


# --- Nodes -------------------------------------------------------------------


def normalize_node(state: AgentState) -> dict:
    text = sanitize_input(state["original_input"])
    injection_check = check_prompt_injection(text)
    warnings = []
    if not injection_check.passed:
        warnings.append(f"Input flagged by injection heuristic: {injection_check.reason}")
        logger.warning("Injection heuristic tripped on user input.")

    result = normalize(text)
    if result.warning:
        warnings.append(result.warning)

    return {
        "normalized_query": result.query_bn,
        "detected_script": result.script,
        "warnings": state.get("warnings", []) + warnings,
    }


def route_node(state: AgentState) -> dict:
    decision = route(state["normalized_query"])
    logger.info("Route decision: %s", decision)
    return {"route": decision}


def route_selector(state: AgentState) -> list[str]:
    """Conditional-edge function: fan out to whichever of rag/weather/price
    the router selected. Always includes at least one branch."""
    decision = state["route"]
    dests = []
    if decision.needs_rag:
        dests.append("rag")
    if decision.needs_weather:
        dests.append("weather")
    if decision.needs_price:
        dests.append("price")
    if not dests:
        dests.append("rag")  # safety net — see RouteDecision default
    return dests


def extract_citations(chunks: list) -> list[dict]:
    """Source + page only — deliberately never includes raw chunk text.

    Rationale (from project review): raw OCR chunks can have broken/cut
    sentences at their edges. The synthesis LLM rewrites that raw text
    into a clean answer, so the answer itself is unaffected — but if the
    UI displayed raw chunk text as the "citation", the user WOULD see
    the broken sentence. Keeping citations to source+page (e.g.
    "dhan chash.pdf, পাতা ৭৯") avoids that entirely; a future "বিস্তারিত
    দেখুন" (view details) UI affordance can show raw text on demand if
    ever needed, but it is never the default citation display.
    """
    seen = set()
    citations = []
    for c in chunks:
        key = (c.source, c.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {"source": c.source, "page": c.page, "chapter": getattr(c, "chapter", None)}
        )
    return citations


def rag_node(state: AgentState) -> dict:
    try:
        from rag.retriever import get_retriever  # local import: keeps rag/ optional at import time

        retriever = get_retriever()
        chunks = retriever.retrieve(state["normalized_query"], top_k=5)
        safe_chunks = filter_injected_chunks(chunks)
        return {
            "retrieved_chunks": safe_chunks,
            "citations": extract_citations(safe_chunks),
        }
    except Exception as exc:  # RuntimeError (no collection), ChromaDB errors, etc.
        logger.error("RAG retrieval failed: %s", exc)
        return {
            "retrieved_chunks": [],
            "citations": [],
            "errors": state.get("errors", []) + [f"RAG retrieval failed: {exc}"],
        }


def weather_node(state: AgentState) -> dict:
    decision = state["route"]
    location = decision.location or state.get("user_location") or DEFAULT_LOCATION
    result = get_weather(location)
    return {"weather_result": result}


def price_node(state: AgentState) -> dict:
    decision = state["route"]
    location = decision.location or state.get("user_location")
    crops = list(getattr(decision, "crops", None) or [])
    if not crops and (decision.crop or "").strip():
        crops = [(decision.crop or "").strip()]

    if crops:
        results = {crop: get_crop_price(crop, division=location) for crop in crops}
        first = next(iter(results.values()))
        return {"price_result": first, "price_results": results}

    # No specific crop named -> return the complete selected/default
    # division report. Do not guess a crop or reduce the report to a fixed
    # predefined crop sweep; the DAM PDF contains continuation tables and
    # product rows that must remain available to the UI/evaluation layer.
    result = get_full_market_prices(division=location, force_refresh=True)
    return {"market_result": result}


def vision_node(state: AgentState) -> dict:
    """Runs YOLO detection on state['image_path']. Lazily imports
    vision.detector so the text-only graph path never requires the
    vision dependencies (ultralytics/opencv) to be installed."""
    image_path = state.get("image_path")
    if not image_path:
        return {"errors": state.get("errors", []) + ["vision_node called without image_path"]}

    try:
        from vision.detector import predict  # see vision/detector.py

        detections = predict(image_path)
        top = detections[0] if detections else None
        if top is None:
            return {
                "vision_result": {"class_name": None, "confidence": 0.0, "tier": "red"},
                "warnings": state.get("warnings", []) + ["No disease/crop detected in image"],
            }
        try:
            from vision.class_mapping import get_class_info

            raw_label = top.get("class_name") or top.get("class") or top.get("label")
            info = get_class_info(raw_label)
            top = {
                **top,
                "class_name": raw_label,
                "crop_bn": info.crop_bn,
                "bn_name": info.bn_name,
                "rag_query": info.rag_query,
            }
        except Exception as exc:
            logger.warning("Vision class mapping unavailable: %s", exc)
        return {"vision_result": top}
    except Exception as exc:
        logger.error("Vision inference failed: %s", exc)
        return {
            "vision_result": {"class_name": None, "confidence": 0.0, "tier": "red"},
            "errors": state.get("errors", []) + [f"Vision inference failed: {exc}"],
        }


_IMAGE_CLASS_TERMS = {
    "rice": ("ধান", "চাল", "rice"),
    "potato": ("আলু", "potato"),
    "tomato": ("টমেটো", "tomato"),
}


def _vision_crop_key(vision: dict) -> str:
    class_name = str(vision.get("class_name") or "").casefold()
    for key in _IMAGE_CLASS_TERMS:
        if class_name.startswith(f"{key}_") or key in class_name:
            return key
    crop_bn = str(vision.get("crop_bn") or "")
    return {"ধান": "rice", "আলু": "potato", "টমেটো": "tomato"}.get(crop_bn, "")


def _image_answer_conflicts(answer: str, vision: dict) -> bool:
    """Reject prose that names a different crop than the detector prediction."""
    predicted = _vision_crop_key(vision)
    if not predicted:
        return False
    text = answer.casefold()
    for crop, terms in _IMAGE_CLASS_TERMS.items():
        if crop == predicted:
            continue
        if any(term.casefold() in text for term in terms):
            return True
    return False


def _image_conflict_fallback(vision: dict) -> str:
    crop = vision.get("crop_bn") or "অজানা ফসল"
    disease = vision.get("bn_name") or "নির্দিষ্ট রোগ"
    return (
        f"ছবির detector অনুযায়ী ফসলটি {crop} এবং সম্ভাব্য শনাক্তকরণ {disease}। "
        "নথির তথ্যের সঙ্গে অসঙ্গতি থাকায় বিস্তারিত রোগ-পরামর্শ নিশ্চিতভাবে দেওয়া হয়নি। "
        "আরও পরিষ্কার close-up ছবি দিন এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।"
    )


def _build_context_block(state: AgentState, *, compact: bool = False) -> str:
    parts = []

    chunks = state.get("retrieved_chunks") or []
    if chunks:
        parts.append("[নথি থেকে প্রাপ্ত তথ্য]")
        selected_chunks = chunks[:3] if compact else chunks
        for c in selected_chunks:
            text = re.sub(r"\s+", " ", str(c.text)).strip()
            if compact and len(text) > 850:
                text = text[:850].rsplit(" ", 1)[0] + "…"
            parts.append(f"- ({c.source}, পৃষ্ঠা {c.page}): {text}")

    weather = state.get("weather_result")
    if weather is not None:
        if weather.available:
            if weather.rain_expected is True:
                rain_text = "এই মুহূর্তে বৃষ্টি হচ্ছে"
            elif weather.rain_expected is False:
                rain_text = "এই মুহূর্তে বৃষ্টি হচ্ছে না"
            else:
                rain_text = "বৃষ্টির বর্তমান তথ্য অনির্দিষ্ট"
            parts.append(
                f"[বর্তমান আবহাওয়া তথ্য] {weather.location}: {weather.temperature_c}°C, "
                f"{weather.description}, আর্দ্রতা {weather.humidity_percent}%, {rain_text}. "
                "এটি current weather observation; দিনের বাকি সময়ের forecast নয়।"
            )
        else:
            parts.append(f"[আবহাওয়া তথ্য অনুপলব্ধ] কারণ: {weather.error}")

    price_results = state.get("price_results") or {}
    if len(price_results) > 1:
        parts.append("[নির্দিষ্ট একাধিক DAM বাজারদর]")
        for item_price in price_results.values():
            if item_price.available:
                parts.append(
                    f"- {item_price.crop}: {item_price.price_range or item_price.price_bdt_per_kg}; "
                    f"একক: {getattr(item_price, 'unit', '') or 'প্রতি কেজি'}; "
                    f"তারিখ: {item_price.date}; বিভাগ: {item_price.division_note}"
                )
            else:
                parts.append(f"- {item_price.crop}: তথ্য অনুপলব্ধ; কারণ: {item_price.error}")

    price = state.get("price_result")
    if price is not None and len(price_results) <= 1:
        if price.available:
            unit = getattr(price, "unit", "") or "প্রতি কেজি"
            division_note = getattr(price, "division_note", "") or ""
            parts.append(
                f"[নিশ্চিত DAM বাজারদর] পণ্য: {price.crop}; "
                f"দাম: {getattr(price, 'price_range', '') or price.price_bdt_per_kg}; "
                f"একক: {unit}; গড় দাম: {price.price_bdt_per_kg}; "
                f"রিপোর্টের তারিখ: {price.date}; বাজার: {price.market}; "
                f"বিভাগের তথ্য: {division_note or 'প্রদত্ত নয়'}; উৎস: {price.source}"
            )
        else:
            parts.append(f"[বাজারদর তথ্য অনুপলব্ধ] কারণ: {price.error}")

    market = state.get("market_result")
    if market is not None:
        if market.available:
            parts.append(
                f"[সম্পূর্ণ DAM বাজার রিপোর্ট] বিভাগ: {market.division}; "
                f"রিপোর্টের তারিখ: {market.date}; উৎস: {market.source}"
            )
            for item in market.items:
                unit = item.unit or "একক উল্লেখ নেই"
                category = f" বিভাগ: {item.category};" if item.category else ""
                parts.append(
                    f"- পণ্য: {item.product};{category} দাম: {item.price_range}; "
                    f"একক: {unit}; রিপোর্টের তারিখ: {market.date}; "
                    f"বিভাগ: {market.division}; উৎস: {market.source}"
                )
        else:
            parts.append(
                f"[সম্পূর্ণ বাজার রিপোর্ট অনুপলব্ধ] বিভাগ: {market.division}; "
                f"কারণ: {market.error}"
            )

    vision = state.get("vision_result")
    if vision is not None:
        display_name = vision.get("bn_name") or vision.get("class_name") or "সনাক্ত করা যায়নি"
        crop_name = vision.get("crop_bn") or "অজানা ফসল"
        parts.append(
            f"[ছবি বিশ্লেষণ] ফসল: {crop_name}; সম্ভাব্য শনাক্তকরণ: {display_name}; "
            f"confidence: {vision.get('confidence')}; tier: {vision.get('tier')}"
        )

    if not parts:
        parts.append("(কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি)")

    return "\n".join(parts)


def _compact_market_context(market: MarketResult, preview_limit: int = 12) -> str:
    """Keep the LLM prompt below provider TPM limits.

    The complete structured report stays in ``market_result`` for the UI; the
    model only needs a bounded preview to write a short introductory answer.
    """
    lines = [
        f"[সম্পূর্ণ DAM বাজার রিপোর্ট] বিভাগ: {market.division}; "
        f"তারিখ: {market.date}; মোট পরিষ্কার পণ্য: {len(market.items)}; "
        f"বাদ পড়া অস্পষ্ট row: {getattr(market, 'omitted_count', 0)}; উৎস: {market.source}",
    ]
    for item in market.items[:preview_limit]:
        lines.append(f"- {item.product}: {item.price_range} {item.unit or 'একক উল্লেখ নেই'}")
    if len(market.items) > preview_limit:
        lines.append(
            "নোট: সম্পূর্ণ পরিষ্কার পণ্যতালিকা UI-এর market cards-এ দেখানো হবে। "
            "LLM উত্তরটি শুধু সংক্ষিপ্ত overview হবে; অসম্পূর্ণ table বা missing-row দাবি করা যাবে না।"
        )
    return "\n".join(lines)


def _market_fallback_answer(market: MarketResult) -> str:
    """Return a deterministic farmer-facing answer when LLM is unavailable."""
    omitted = getattr(market, "omitted_count", 0)
    omitted_note = (
        f" Source PDF-এর {omitted}টি অস্পষ্ট row ভুল তথ্য এড়াতে বাদ দেওয়া হয়েছে।"
        if omitted
        else ""
    )
    return (
        f"{market.division} বিভাগের {market.date} তারিখের DAM বাজারদরের রিপোর্ট পাওয়া গেছে। "
        f"পরিষ্কারভাবে পাওয়া {len(market.items)}টি পণ্যের সম্পূর্ণ দাম নিচের তালিকায় দেখানো হয়েছে।"
        f"{omitted_note}"
    )


_PRICE_CROP_LABELS = {
    "potato": "আলু",
    "onion": "পেঁয়াজ",
    "masur dal": "মসুর ডাল",
    "tomato": "টমেটো",
    "garlic": "রসুন",
    "ginger": "আদা",
    "green chili": "কাঁচা মরিচ",
    "rice": "চাল",
}


def _specific_price_answer(state: AgentState) -> str:
    """Render exact named-crop prices once; UI cards are for full-market only."""
    results = dict(state.get("price_results") or {})
    if not results and state.get("price_result") is not None:
        price = state["price_result"]
        results[str(getattr(price, "crop", "পণ্য"))] = price

    lines = ["নির্দিষ্ট বাজারদর:"]
    dates: list[str] = []
    for crop, price in results.items():
        label = _PRICE_CROP_LABELS.get(str(crop), str(crop))
        if not getattr(price, "available", False):
            lines.append(f"{label}: বাজারদর পাওয়া যায়নি।")
            continue
        price_range = getattr(price, "price_range", "") or getattr(price, "price_bdt_per_kg", "") or "—"
        unit = getattr(price, "unit", "") or "প্রতি কেজি"
        lines.append(f"{label}: {price_range} {unit}")
        report_date = str(getattr(price, "date", "") or "").strip()
        if report_date and report_date not in dates:
            dates.append(report_date)

    if dates:
        lines.append("")
        lines.append(f"রিপোর্টের তারিখ: {', '.join(dates)}")
    lines.append("তথ্যসূত্র: DAM দৈনিক বিভাগীয় খুচরা বাজারদর (PDF)")
    return "\n".join(lines)


def _deterministic_synthesis_fallback(state: AgentState) -> str:
    """Return a short complete answer when LLM prose is unavailable."""
    vision = state.get("vision_result") or {}
    if vision:
        display_name = vision.get("bn_name") or vision.get("class_name") or "নিশ্চিতভাবে শনাক্ত করা যায়নি"
        confidence = vision.get("confidence")
        confidence_text = f" confidence {float(confidence) * 100:.1f}%" if confidence is not None else ""
        return (
            f"ছবিতে সম্ভাব্য শনাক্তকরণ: {display_name}{confidence_text}। "
            "এটি automated prediction; চূড়ান্ত রোগ নির্ণয় নয়। বিস্তারিত source-grounded ব্যাখ্যা এখন পাওয়া যায়নি।"
        )
    if state.get("weather_result") is not None or state.get("price_result") is not None or state.get("price_results"):
        return "উপলব্ধ আবহাওয়া ও নির্দিষ্ট বাজারদরের তথ্য দেওয়া হয়েছে।"
    if state.get("retrieved_chunks"):
        return (
            "প্রাসঙ্গিক কৃষি নথি পাওয়া গেছে, কিন্তু রোগ/পোকার ধরন নিশ্চিতভাবে মেলানো যায়নি। "
            "আক্রান্ত পাতা, শীষ বা ফলের পরিষ্কার close-up ছবি দিন এবং নির্দিষ্ট কীটনাশক বা ডোজ "
            "ব্যবহারের আগে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।"
        )
    return "এই মুহূর্তে সম্পূর্ণ উত্তর তৈরি করা যায়নি। কিছুক্ষণ পরে আবার চেষ্টা করুন।"


def synthesis_node(state: AgentState) -> dict:
    market = state.get("market_result")
    market_only = (
        market is not None
        and market.available
        and not state.get("weather_result")
        and not state.get("retrieved_chunks")
        and not state.get("vision_result")
    )

    # Full-market reports are already structured data. Do not ask the LLM to
    # repeat 40–60 rows or create a partial Markdown table. The UI cards are
    # the single source of truth for every exact price, unit and product.
    if market_only:
        return {
            "final_answer": (
                f"{market.division} বিভাগের {market.date} তারিখের সম্পূর্ণ DAM "
                "বাজারদর নিচের তালিকায় দেওয়া হলো।"
            ),
            "warnings": state.get("warnings", []),
        }

    # Specific single-/multi-crop price requests are already exact structured
    # data. Do not generate duplicate price cards or spend an LLM call repeating
    # the same values; the answer itself contains each price once plus date/source.
    specific_prices = state.get("price_results") or {}
    if not specific_prices and state.get("price_result") is not None:
        specific_prices = {"single": state.get("price_result")}
    if specific_prices and not state.get("weather_result") and not state.get("retrieved_chunks") and not state.get("vision_result"):
        return {"final_answer": _specific_price_answer(state)}

    context = (
        _compact_market_context(market)
        if market is not None and market.available
        else _build_context_block(state, compact=True)
    )
    query = state.get("normalized_query") or state["original_input"]

    prompt = SYNTHESIS_USER_TEMPLATE.format(query=query, context=context)

    try:
        llm = get_llm()
        try:
            response = llm.generate(
                prompt,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.25,
                max_output_tokens=1200,
            )
        except LLMError as first_exc:
            # A single shorter retry is useful for long RAG/image answers. Do
            # not loop retries; deterministic fallback below always completes.
            reason = str(first_exc).lower()
            if not any(token in reason for token in ("truncat", "token limit", "max_tokens", "length")):
                raise
            concise_prompt = (
                f"{prompt}\n\nবিশেষ নির্দেশ: compact কিন্তু সম্পূর্ণ উত্তর দাও। সাধারণত ৩–৫টি ছোট bullet বা paragraph যথেষ্ট। "
                "Table সত্যিই দরকার হলে সর্বোচ্চ ২–৪টি column ও ৩–৬টি row রাখো; বড় table নয়। "
                "Repetition নয় এবং উত্তরটি সম্পূর্ণ বাক্যে শেষ করো।"
            )
            response = llm.generate(
                concise_prompt,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=700,
            )
        answer = response.text
    except LLMError as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        diagnostic = f"Synthesis fallback used: {exc}"
        if market is not None and market.available:
            return {
                "final_answer": _market_fallback_answer(market),
                "warnings": state.get("warnings", []) + [diagnostic],
            }
        return {
            "final_answer": _deterministic_synthesis_fallback(state),
            "warnings": state.get("warnings", []) + [diagnostic],
        }

    answer = strip_garbled_segments(answer)
    answer = strip_source_dump(answer)

    vision = state.get("vision_result")
    tier = vision.get("tier") if vision else None
    if vision and _image_answer_conflicts(answer, vision):
        answer = _image_conflict_fallback(vision)
    else:
        answer = enforce_uncertainty_language(answer, tier)

    return {"final_answer": answer}


# --- Graph assembly -----------------------------------------------------------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("normalize", normalize_node)
    workflow.add_node("router", route_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("price", price_node)
    workflow.add_node("vision", vision_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", "router")

    workflow.add_conditional_edges("router", route_selector, ["rag", "weather", "price"])

    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("weather", "synthesis")
    workflow.add_edge("price", "synthesis")
    workflow.add_edge("synthesis", END)

    # Image path is invoked directly (see run_image_query below) rather
    # than wired into the same conditional entry, since input_type is
    # known before the graph even starts (Section 7: input router is the
    # first decision, upstream of this graph).
    workflow.add_edge("vision", "synthesis")

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_text_query(query: str, user_location: Optional[str] = None) -> AgentState:
    graph = get_graph()
    initial: AgentState = {
        "original_input": query,
        "input_type": "text",
        "user_location": user_location,
        "warnings": [],
        "errors": [],
    }
    return graph.invoke(initial)


def run_image_query(image_path: str, accompanying_text: str = "") -> AgentState:
    """Image queries run vision_node directly, then synthesis — they skip
    normalize/route since there's no text to route (Section 7)."""
    state: AgentState = {
        "original_input": accompanying_text,
        "input_type": "image",
        "image_path": image_path,
        "normalized_query": accompanying_text or "এই ছবিতে কী সমস্যা আছে ব্যাখ্যা করো",
        "warnings": [],
        "errors": [],
    }
    state.update(vision_node(state))

    # Enrich image diagnosis with source-grounded RAG advice using the reviewed
    # Bengali mapping, while keeping image queries independent from text routing.
    rag_query = (state.get("vision_result") or {}).get("rag_query")
    if rag_query:
        try:
            from rag.retriever import get_retriever

            chunks = filter_injected_chunks(get_retriever().retrieve(rag_query, top_k=8))
            vision = state.get("vision_result") or {}
            crop_key = _vision_crop_key(vision)
            source_terms = {
                "rice": ("dhan", "dhaner", "rice"),
                "potato": ("alu", "potato"),
                "tomato": ("tomato",),
            }.get(crop_key)
            if source_terms:
                scoped = [
                    chunk for chunk in chunks
                    if any(term in str(getattr(chunk, "source", "")).casefold() for term in source_terms)
                ]
                # If metadata naming is unavailable, retain only chunks whose
                # text explicitly names the detector crop; never mix crops.
                if not scoped:
                    crop_terms = _IMAGE_CLASS_TERMS.get(crop_key, ())
                    scoped = [
                        chunk for chunk in chunks
                        if any(term.casefold() in str(getattr(chunk, "text", "")).casefold() for term in crop_terms)
                    ]
                chunks = scoped[:5]
            state.update(
                {
                    "retrieved_chunks": chunks,
                    "citations": extract_citations(chunks),
                }
            )
        except Exception as exc:
            logger.warning("Image RAG enrichment unavailable: %s", exc)
            state.setdefault("warnings", []).append("Image RAG advice unavailable")

    state.update(synthesis_node(state))
    return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Graph compiled:", get_graph())
    print(
        "\nNo API keys / ChromaDB data available in this environment — "
        "run this locally with .env populated to exercise a real query, e.g.:\n"
        "  python -c \"from agent.graph import run_text_query; "
        "print(run_text_query('ধানের ব্লাস্ট রোগের প্রতিকার কী')['final_answer'])\""
    )