"""Provider-agnostic LLM abstraction.

Not in the originally requested file list but required by normalizer.py,
router.py, and graph.py's synthesis node — none of them should import
`groq` or `google.generativeai` directly (Section 14: "prefer an
abstraction such as LLMProvider so the application does not become
tightly coupled to one provider").

Usage:
    from agent.llm_provider import get_llm
    llm = get_llm()
    text = llm.generate("prompt here", system="optional system prompt")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from agent import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when no configured provider could produce a response."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


def _is_length_finish_reason(reason: object) -> bool:
    """Recognize provider-specific max-token/truncation finish reasons."""
    if reason is None:
        return False
    value = str(reason).upper()
    return any(marker in value for marker in ("LENGTH", "MAX_TOKENS", "MAXTOKENS", "TOKEN_LIMIT"))


class LLMProvider:
    """Tries the configured primary provider, falls back to the other one.

    Both providers are optional at import time — this class only fails
    (raises LLMError) when .generate() is actually called and neither
    provider is usable (missing key, import error, timeout, API error).
    """

    def __init__(self) -> None:
        self._order = (
            ["groq", "gemini"]
            if config.LLM_PROVIDER == "groq"
            else ["gemini", "groq"]
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate bounded text and fail over if a provider loops.

        The optional ``max_output_tokens`` argument preserves the original API
        while allowing one call to request a smaller completion. All callers
        still receive the configured global bound by default.
        """
        output_limit = max_output_tokens or config.LLM_MAX_OUTPUT_TOKENS
        errors = []
        for provider_name in self._order:
            try:
                if provider_name == "groq":
                    return self._call_groq(prompt, system, json_mode, temperature, output_limit)
                return self._call_gemini(prompt, system, json_mode, temperature, output_limit)
            except Exception as exc:  # noqa: BLE001 - deliberately broad; try fallback provider
                logger.warning("LLM provider '%s' failed: %s", provider_name, exc)
                errors.append(f"{provider_name}: {exc}")
        raise LLMError(
            "All configured LLM providers failed. Details: " + "; ".join(errors)
        )

    @staticmethod
    def _validate_text(text: Optional[str], provider: str, *, json_mode: bool) -> str:
        if not text or not text.strip():
            raise LLMError(f"{provider} returned an empty response")
        cleaned = text.strip()
        if not json_mode and _is_degenerate_output(cleaned):
            raise LLMError(f"{provider} returned a repetitive/degenerate response")
        if not json_mode and _looks_incomplete_text(cleaned):
            raise LLMError(f"{provider} returned a response that appears truncated")
        return cleaned

    def _call_groq(
        self,
        prompt: str,
        system: Optional[str],
        json_mode: bool,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY not set")
        from groq import Groq  # local import: don't require the package if unused

        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT_SECONDS)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "max_tokens": max_output_tokens,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
            "presence_penalty": config.LLM_PRESENCE_PENALTY,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if _is_length_finish_reason(finish_reason):
            raise LLMError("Groq response reached the output token limit and was discarded")
        text = self._validate_text(
            choice.message.content,
            "Groq",
            json_mode=json_mode,
        )
        return LLMResponse(text=text, provider="groq", model=config.GROQ_MODEL)

    def _call_gemini(
        self,
        prompt: str,
        system: Optional[str],
        json_mode: bool,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        if not config.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY not set")
        # google.generativeai is deprecated (EOL) -- using the new unified
        # google-genai SDK instead. Install: pip install google-genai --break-system-packages
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT_SECONDS * 1000),
        )
        # Gemini 3.6 Flash rejects OpenAI-style penalty fields. Keep those
        # controls for Groq only and use the supported Gemini config fields.
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )

        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=gen_config,
        )
        candidates = getattr(resp, "candidates", None) or []
        if candidates and _is_length_finish_reason(getattr(candidates[0], "finish_reason", None)):
            raise LLMError("Gemini response reached the output token limit and was discarded")
        text = self._validate_text(
            getattr(resp, "text", None),
            "Gemini",
            json_mode=json_mode,
        )
        return LLMResponse(text=text, provider="gemini", model=config.GEMINI_MODEL)


def _looks_incomplete_text(text: str) -> bool:
    """Catch long prose cut off before a sentence/list boundary.

    This is deliberately conservative: short answers may legitimately omit a
    final full stop, but a long answer ending mid-word/phrase should not reach
    the farmer-facing UI.
    """
    words = re.findall(r"\S+", text)
    if len(words) < 18:
        return False
    terminal = ".!?।॥:;)]}»”\"'’"
    return not text.rstrip().endswith(tuple(terminal))


def _is_degenerate_output(text: str) -> bool:
    """Detect obvious looping without rejecting normal repeated table units.

    Only contiguous blocks of at least six words repeated three times, or long
    identical lines repeated three times, are rejected. Short phrases such as
    ``প্রতি কেজি`` may legitimately repeat in a market report.
    """
    words = re.findall(r"\S+", text.casefold())
    minimum_words = config.LLM_DEGENERACY_MIN_WORDS
    repeat_count = config.LLM_DEGENERACY_REPEAT_COUNT
    if len(words) < minimum_words:
        return False

    max_block = min(24, len(words) // repeat_count)
    for block_size in range(6, max_block + 1):
        for start in range(0, len(words) - block_size * repeat_count + 1):
            block = words[start : start + block_size]
            repeated = all(
                words[start + offset * block_size : start + (offset + 1) * block_size] == block
                for offset in range(1, repeat_count)
            )
            if repeated:
                return True

    long_lines = [
        re.sub(r"\s+", " ", line.casefold()).strip()
        for line in text.splitlines()
        if len(re.findall(r"\S+", line)) >= 8
    ]
    return any(long_lines.count(line) >= repeat_count for line in set(long_lines))


_singleton: Optional[LLMProvider] = None


def get_llm() -> LLMProvider:
    global _singleton
    if _singleton is None:
        _singleton = LLMProvider()
    return _singleton