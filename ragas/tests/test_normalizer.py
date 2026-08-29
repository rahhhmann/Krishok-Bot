"""Unit tests for agent/normalizer.py.

All LLM calls are mocked -- these tests never hit Groq/Gemini. That
means they verify normalizer.py's own logic (script detection,
validation, fallback behavior), not real transliteration quality.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.llm_provider import LLMError, LLMResponse
from agent.normalizer import detect_script, normalize


# --- detect_script -----------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ধানের ব্লাস্ট রোগের প্রতিকার কী", "bangla"),
        ("how to treat potato late blight", "latin"),
        ("dhaner poka doman korbo kivabe", "latin"),
        ("", "unknown"),
        ("   ", "unknown"),
        ("123 !!! ...", "unknown"),  # no alpha-like chars at all
    ],
)
def test_detect_script_pure_cases(text, expected):
    assert detect_script(text) == expected


def test_detect_script_mixed_is_neither_bangla_nor_latin():
    # Roughly half Bangla, half Latin letters -> below the 0.7 threshold
    # for both, so classified as "mixed" (see module docstring: this
    # function only answers "what script", not "is it valid Banglish").
    text = "amar dhan গাছে blast disease হয়েছে"
    assert detect_script(text) == "mixed"


# --- normalize: passthrough / unknown -----------------------------------


def test_normalize_bangla_is_passthrough_and_never_calls_llm():
    llm = MagicMock()
    result = normalize("ধানের ব্লাস্ট রোগের প্রতিকার কী", llm=llm)
    assert result.method == "passthrough"
    assert result.script == "bangla"
    assert result.query_bn == "ধানের ব্লাস্ট রোগের প্রতিকার কী"
    assert result.warning is None
    llm.generate.assert_not_called()


def test_normalize_empty_string_is_passthrough():
    llm = MagicMock()
    result = normalize("", llm=llm)
    assert result.method == "passthrough"
    assert result.script == "unknown"
    llm.generate.assert_not_called()


# --- normalize: latin/mixed -> LLM transliteration -----------------------


def test_normalize_latin_success_uses_llm_output():
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(
        text="ধানের পোকা দমন করব কিভাবে", provider="groq", model="test"
    )
    result = normalize("dhaner poka doman korbo kivabe", llm=llm)
    assert result.method == "llm_transliteration"
    assert result.query_bn == "ধানের পোকা দমন করব কিভাবে"
    assert result.warning is None
    llm.generate.assert_called_once()
    # temperature=0.0 for deterministic transliteration
    assert llm.generate.call_args.kwargs["temperature"] == 0.0


def test_normalize_latin_llm_returns_english_falls_back_with_warning():
    """LLM ignored the instruction and answered in English -- the
    Bangla-ratio validation must catch this and fall back rather than
    silently using non-Bangla text as "normalized"."""
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(
        text="how to control rice pests", provider="groq", model="test"
    )
    result = normalize("dhaner poka doman korbo kivabe", llm=llm)
    assert result.method == "fallback_failed"
    assert result.query_bn == "dhaner poka doman korbo kivabe"  # original preserved
    assert result.warning is not None
    assert "could not be verified" in result.warning


def test_normalize_llm_error_falls_back_with_warning():
    llm = MagicMock()
    llm.generate.side_effect = LLMError("all providers down")
    result = normalize("how to treat potato late blight", llm=llm)
    assert result.method == "fallback_failed"
    assert result.query_bn == "how to treat potato late blight"
    assert "LLM unavailable" in result.warning


def test_normalize_strips_surrounding_quotes_from_llm_output():
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(
        text='"ধানের পোকা দমন করব কিভাবে"', provider="groq", model="test"
    )
    result = normalize("dhaner poka doman korbo kivabe", llm=llm)
    assert result.query_bn == "ধানের পোকা দমন করব কিভাবে"
    assert not result.query_bn.startswith('"')
