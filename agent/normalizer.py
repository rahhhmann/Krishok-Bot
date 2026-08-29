"""agent/normalizer.py

Handles the script-mismatch problem flagged in the project plan (Section
3 / Section 9): the RAG corpus is entirely Bangla-script, but users may
type in Bangla, English, or Banglish (romanized Bangla). A raw Banglish
query embedded and matched against Bangla-script chunks retrieves badly
(verified empirically — see retriever.py's own test queries).

Approach:
    1. Detect script deterministically (character-class ratios — cheap,
       explainable, no model call needed).
    2. If already Bangla script: pass through unchanged.
    3. Otherwise (English or Banglish — these are NOT reliably
       distinguishable by character set alone, e.g. "ami dhan chashi"
       looks like English words but is Banglish): ask the LLM to
       translate/transliterate into standard Bangla, since no offline
       phonetic transliteration library is part of this project's
       dependency list and a hand-rolled phonetic mapper would be a
       fragile, low-accuracy reimplementation of what an LLM already
       does adequately for this domain's vocabulary.
    4. Validate the LLM's output is actually majority-Bangla-script
       before trusting it (LLMs sometimes ignore instructions and
       answer in English instead of transliterating). If validation
       fails, fall back to the original text and set a warning flag —
       per project rule, never claim transliteration succeeded when it
       didn't.

Known limitation (documented, not hidden): quality depends entirely on
the LLM's Banglish comprehension for agricultural vocabulary. This has
NOT been benchmarked against a labeled Banglish test set. Treat
normalization as best-effort, not guaranteed-correct.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from agent import config
from agent.llm_provider import LLMError, LLMProvider, get_llm
from agent.prompts import NORMALIZATION_SYSTEM_PROMPT, NORMALIZATION_USER_TEMPLATE

logger = logging.getLogger(__name__)

_BANGLA_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_ALPHA_LIKE_RE = re.compile(r"[^\s\d.,;:!?()\-]")

Script = str  # "bangla" | "latin" | "mixed" | "unknown"


@dataclass
class NormalizedQuery:
    original: str
    script: Script
    query_bn: str  # Bangla-script text to use for embedding/retrieval
    method: str  # "passthrough" | "llm_transliteration" | "fallback_failed"
    warning: Optional[str] = None


def detect_script(text: str) -> Script:
    """Character-ratio based script detection.

    Deterministic and cheap on purpose (Section 9: "do not pretend
    simple language detection solves this" — this function only
    answers "what script is it typed in", not "is this valid
    Banglish/English", which is why latin-script text still goes
    through LLM normalization rather than being treated as "solved".
    """
    stripped = text.strip()
    if not stripped:
        return "unknown"

    alpha_like = len(_ALPHA_LIKE_RE.findall(stripped))
    if alpha_like == 0:
        return "unknown"

    bangla = len(_BANGLA_CHAR_RE.findall(stripped))
    latin = len(_LATIN_CHAR_RE.findall(stripped))

    bangla_ratio = bangla / alpha_like
    latin_ratio = latin / alpha_like

    if bangla_ratio >= 0.7:
        return "bangla"
    if latin_ratio >= 0.7:
        return "latin"  # could be English OR Banglish — not distinguishable here
    return "mixed"


def _is_trustworthy_bangla_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    alpha_like = len(_ALPHA_LIKE_RE.findall(stripped))
    if alpha_like == 0:
        return False
    bangla_ratio = len(_BANGLA_CHAR_RE.findall(stripped)) / alpha_like
    return bangla_ratio >= config.NORMALIZATION_MIN_BANGLA_RATIO


def normalize(text: str, llm: Optional[LLMProvider] = None) -> NormalizedQuery:
    """Main entry point. Never raises — always returns a usable NormalizedQuery,
    with `warning` set if normalization degraded to a fallback."""
    text = text.strip()
    script = detect_script(text)

    if script == "bangla" or script == "unknown":
        return NormalizedQuery(
            original=text, script=script, query_bn=text, method="passthrough"
        )

    # script is "latin" or "mixed" -> attempt LLM transliteration/translation
    llm = llm or get_llm()
    try:
        prompt = NORMALIZATION_USER_TEMPLATE.format(query=text)
        response = llm.generate(
            prompt,
            system=NORMALIZATION_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=180,
        )
        candidate = response.text.strip().strip('"').strip()

        if _is_trustworthy_bangla_output(candidate):
            return NormalizedQuery(
                original=text,
                script=script,
                query_bn=candidate,
                method="llm_transliteration",
            )

        logger.warning(
            "Normalization output failed Bangla-ratio validation for query %r "
            "(got %r) — falling back to original text.",
            text[:60],
            candidate[:60],
        )
        return NormalizedQuery(
            original=text,
            script=script,
            query_bn=text,
            method="fallback_failed",
            warning=(
                "Bangla transliteration could not be verified; retrieval will "
                "run against the original (non-Bangla-script) text and may "
                "perform poorly."
            ),
        )

    except LLMError as exc:
        logger.error("Normalization LLM call failed: %s", exc)
        return NormalizedQuery(
            original=text,
            script=script,
            query_bn=text,
            method="fallback_failed",
            warning=f"LLM unavailable for normalization ({exc}); using raw query text.",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_queries = [
        ("Bangla", "ধানের বাদামী গাছ ফড়িং দমন করব কিভাবে"),
        ("English", "how to treat potato late blight"),
        ("Banglish", "dhaner poka doman korbo kivabe"),
        ("Mixed", "amar dhan gach e blast disease hoyeche"),
    ]
    for label, q in test_queries:
        result = normalize(q)
        print(f"\n[{label}] input: {q}")
        print(f"  script={result.script} method={result.method}")
        print(f"  query_bn={result.query_bn}")
        if result.warning:
            print(f"  WARNING: {result.warning}")
