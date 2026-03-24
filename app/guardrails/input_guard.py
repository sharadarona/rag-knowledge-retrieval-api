"""
Input Guardrails Layer
======================
Runs BEFORE any LLM call or vector retrieval.

Checks:
  1. Prompt injection detection   — regex patterns
  2. PII redaction                — Microsoft Presidio
  3. Toxicity / profanity filter  — keyword + heuristic scoring
  4. Text sanitization            — control chars, whitespace

Returns a GuardrailResult with cleaned text and safety verdict.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger   = get_settings.__class__  # placeholder — set below
settings = get_settings()
log      = get_logger("guardrails.input")


# ── Prompt Injection Patterns ─────────────────────────────────────────────────

INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all|prior|previous|the\s+above)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(a|an|if)\s+\w+",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*(you|ignore|forget)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"bypass\s+(safety|filter|guardrail|restriction)",
    r"<\s*script\s*>",
    r";\s*(drop|delete|truncate|insert|update)\s+",
    r"--\s*$",
    r"\|\s*(cat|ls|rm|wget|curl)\s+",   # shell injection
]

_COMPILED_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# ── PII Patterns (fallback regex when Presidio not available) ─────────────────

PII_REGEX: dict[str, str] = {
    "EMAIL"        : r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "PHONE_IN"     : r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b',
    "PHONE_US"     : r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "CREDIT_CARD"  : r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "AADHAAR"      : r'\b\d{4}\s\d{4}\s\d{4}\b',
    "PAN"          : r'\b[A-Z]{5}\d{4}[A-Z]\b',
    "SSN"          : r'\b\d{3}-\d{2}-\d{4}\b',
}

_COMPILED_PII = {k: re.compile(v) for k, v in PII_REGEX.items()}


# ── Toxicity Keywords (lightweight, for production use a proper model) ────────

TOXIC_KEYWORDS: list[str] = [
    "kill yourself", "you idiot", "stupid company",
    "f***", "damn you", "go to hell",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    is_safe        : bool
    clean_text     : str
    pii_redacted   : bool              = False
    redacted_types : list[str]         = field(default_factory=list)
    blocked_reason : str               = ""
    toxicity_score : float             = 0.0


# ── Main entry point ──────────────────────────────────────────────────────────

def run_input_guardrails(text: str, case_id: str = "") -> GuardrailResult:
    """
    Full input guardrail pipeline.
    Returns GuardrailResult — caller checks .is_safe before proceeding.
    """
    if not settings.guardrails_enabled:
        return GuardrailResult(is_safe=True, clean_text=text)

    # Step 1 — Sanitize
    text = _sanitize(text)

    # Step 2 — Injection check
    injected, pattern = _check_injection(text)
    if injected:
        log.warning("injection_detected", case_id=case_id, pattern=pattern)
        return GuardrailResult(
            is_safe=False,
            clean_text=text,
            blocked_reason=f"Prompt injection detected"
        )

    # Step 3 — PII redaction
    clean_text, redacted_types = _redact_pii(text)
    pii_found = len(redacted_types) > 0
    if pii_found:
        log.info("pii_redacted", case_id=case_id, types=redacted_types)

    # Step 4 — Toxicity
    toxicity_score = _score_toxicity(clean_text)
    if toxicity_score >= settings.toxicity_threshold:
        log.warning("toxicity_blocked", case_id=case_id, score=toxicity_score)
        return GuardrailResult(
            is_safe=False,
            clean_text=clean_text,
            pii_redacted=pii_found,
            redacted_types=redacted_types,
            blocked_reason="Content blocked by toxicity filter",
            toxicity_score=toxicity_score
        )

    return GuardrailResult(
        is_safe=True,
        clean_text=clean_text,
        pii_redacted=pii_found,
        redacted_types=redacted_types,
        toxicity_score=toxicity_score
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Strip control characters, normalise whitespace."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def _check_injection(text: str) -> tuple[bool, str]:
    for pattern in _COMPILED_INJECTIONS:
        m = pattern.search(text)
        if m:
            return True, pattern.pattern
    return False, ""


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Try Presidio first; fall back to regex if not installed.
    Presidio gives higher accuracy (NLP-based), regex is the safety net.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(
            text=text,
            entities=settings.pii_entity_list,
            language="en"
        )
        if results:
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            redacted_types = list({r.entity_type for r in results})
            return anonymized.text, redacted_types
        return text, []

    except ImportError:
        # Presidio not installed — use regex fallback
        redacted_types = []
        for pii_type, pattern in _COMPILED_PII.items():
            if pattern.search(text):
                text = pattern.sub(f"[REDACTED-{pii_type}]", text)
                redacted_types.append(pii_type)
        return text, redacted_types


def _score_toxicity(text: str) -> float:
    """
    Lightweight heuristic toxicity scorer.
    In production, replace with a real toxicity model
    (e.g. Perspective API, HuggingFace toxic-bert).
    Returns 0.0–1.0; higher = more toxic.
    """
    text_lower = text.lower()
    matches = sum(1 for kw in TOXIC_KEYWORDS if kw in text_lower)
    return min(matches * 0.35, 1.0)
