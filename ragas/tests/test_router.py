"""Unit tests for agent/router.py.

Covers: LLM-JSON routing success path, malformed/invalid-JSON fallback,
repeated-failure -> keyword fallback, and the keyword fallback's own
crop/weather/price detection. No real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent import config as agent_config
from agent.llm_provider import LLMError, LLMResponse
from agent.router import RouteDecision, _keyword_fallback_route, route


def _mock_llm(text: str) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(text=text, provider="groq", model="test")
    return llm


# --- happy path: valid LLM JSON ------------------------------------------


def test_route_valid_llm_json_is_used_directly():
    llm = _mock_llm(
        '{"needs_rag": true, "needs_weather": false, "needs_price": false, '
        '"crop": null, "location": null, "reasoning": "factual disease query"}'
    )
    decision = route("ধানের ব্লাস্ট রোগের প্রতিকার কী", llm=llm)
    assert isinstance(decision, RouteDecision)
    assert decision.needs_rag is True
    assert decision.needs_weather is False
    assert decision.source == "llm"
    llm.generate.assert_called_once()


def test_route_strips_markdown_fences_from_llm_json():
    llm = _mock_llm('```json\n{"needs_rag": true, "needs_price": true, "crop": "potato"}\n```')
    decision = route("আলুর বর্তমান বাজারদর কত", llm=llm)
    assert decision.needs_price is True
    assert decision.crop == "potato"
    assert decision.source == "llm"


# --- malformed output -> retries then keyword fallback -------------------


def test_route_invalid_json_retries_then_falls_back():
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(text="not json at all", provider="groq", model="test")
    decision = route("আজ বরিশালে আবহাওয়া কেমন থাকবে", llm=llm)
    assert decision.source == "keyword_fallback"
    assert decision.needs_weather is True  # "আবহাওয়া" keyword matched
    # Retried up to config.ROUTER_MAX_LLM_ATTEMPTS times before falling back
    assert llm.generate.call_count == agent_config.ROUTER_MAX_LLM_ATTEMPTS


def test_route_llm_error_falls_back_to_keyword_routing():
    llm = MagicMock()
    llm.generate.side_effect = LLMError("groq down")
    decision = route("আলুর বর্তমান বাজারদর কত", llm=llm)
    assert decision.source == "keyword_fallback"
    assert decision.needs_price is True
    assert decision.crop == "potato"


def test_route_llm_json_missing_required_shape_falls_back():
    # Valid JSON, but not a dict RouteDecision can validate against
    # (e.g. a JSON array) -- must not crash, must fall back.
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(text="[1, 2, 3]", provider="groq", model="test")
    decision = route("ধানের ব্লাস্ট রোগের প্রতিকার কী", llm=llm)
    assert decision.source == "keyword_fallback"


# --- keyword fallback directly --------------------------------------------


def test_keyword_fallback_detects_weather():
    decision = _keyword_fallback_route("আজ বৃষ্টি হবে কিনা")
    assert decision.needs_weather is True
    assert decision.needs_rag is True  # always True per fallback design
    assert decision.source == "keyword_fallback"


def test_keyword_fallback_detects_price_and_crop():
    decision = _keyword_fallback_route("টমেটোর দাম কত")
    assert decision.needs_price is True
    assert decision.crop == "tomato"


def test_keyword_fallback_no_crop_when_price_not_requested():
    # crop is only extracted when needs_price is True (per implementation) --
    # a plain disease question mentioning "আলু" should not set crop.
    decision = _keyword_fallback_route("আলুর নাবী ধ্বসা রোগের প্রতিকার কী")
    assert decision.needs_price is False
    assert decision.crop is None


def test_keyword_fallback_neither_tool_keyword_still_defaults_rag_true():
    decision = _keyword_fallback_route("এলোমেলো প্রশ্ন যেখানে কোনো keyword নেই")
    assert decision.needs_rag is True
    assert decision.needs_weather is False
    assert decision.needs_price is False


def test_route_decision_defaults():
    d = RouteDecision()
    assert d.needs_rag is True
    assert d.needs_weather is False
    assert d.needs_price is False
    assert d.source == "llm"
