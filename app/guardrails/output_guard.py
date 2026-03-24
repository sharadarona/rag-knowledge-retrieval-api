"""
Output Guardrails Layer
=======================
Runs AFTER the LLM generates an answer, BEFORE returning to Salesforce.

Checks:
  1. Blocked content patterns     — AI disclaimers, injected text
  2. Length validation            — too short or suspiciously long
  3. Hallucination heuristic      — answer vs retrieved sources overlap
  4. PII leak detection           — ensure LLM didn't expose PII
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from app.utils.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
log      = get_logger("guardrails.output")


# ── Patterns to block in LLM output ──────────────────────────────────────────

BLOCKED_OUTPUT_PATTERNS: list[str] = [
    r"<\s*script\s*>",
    r"ignore\s+previous\s+instructions?",
    r"as\s+an\s+ai\s+(language\s+)?model",
    r"i\s+(cannot|can't|am\s+unable\s+to)\s+provide",
    r"i\s+don't\s+have\s+access\s+to\s+real[\-\s]time",
    r"my\s+training\s+data",
]

_COMPILED_BLOCKED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_OUTPUT_PATTERNS]

# PII patterns — check if LLM leaked PII into its answer
_PII_LEAK_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),  # email
    re.compile(r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b'),                          # IN phone
    re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),                             # credit card
]


@dataclass
class OutputGuardrailResult:
    is_valid       : bool
    final_text     : str
    blocked_reason : str = ""
    pii_leaked     : bool = False


def run_output_guardrails(
    answer: str,
    retrieved_chunks: list[dict],
    case_id: str = ""
) -> OutputGuardrailResult:
    """
    Validates LLM output before returning to Salesforce.
    """
    if not settings.guardrails_enabled:
        return OutputGuardrailResult(is_valid=True, final_text=answer)

    # 1. Minimum length
    if not answer or len(answer.strip()) < 20:
        log.warning("output_too_short", case_id=case_id)
        return OutputGuardrailResult(
            is_valid=False, final_text="",
            blocked_reason="Answer too short or empty"
        )

    # 2. Maximum length
    if len(answer) > 4000:
        log.warning("output_too_long", case_id=case_id, length=len(answer))
        answer = answer[:4000] + "...[truncated]"

    # 3. Blocked pattern check
    for pattern in _COMPILED_BLOCKED:
        if pattern.search(answer):
            log.warning("output_blocked_pattern",
                        case_id=case_id, pattern=pattern.pattern)
            return OutputGuardrailResult(
                is_valid=False, final_text="",
                blocked_reason="Blocked content pattern in output"
            )

    # 4. PII leak check
    for pii_pattern in _PII_LEAK_PATTERNS:
        if pii_pattern.search(answer):
            log.warning("pii_leak_in_output", case_id=case_id)
            # Redact instead of block
            answer = pii_pattern.sub("[REDACTED]", answer)
            return OutputGuardrailResult(
                is_valid=True, final_text=answer, pii_leaked=True
            )

    # 5. Hallucination heuristic — check if answer mentions things
    #    not present in any retrieved chunk
    #    (lightweight version: if no chunks retrieved, flag low confidence)
    if not retrieved_chunks:
        log.warning("no_sources_for_answer", case_id=case_id)
        answer = (
            "Based on available information: " + answer +
            "\n\n⚠ Note: No specific knowledge base articles were found "
            "for this query. Please verify with a human agent."
        )

    return OutputGuardrailResult(is_valid=True, final_text=answer)
