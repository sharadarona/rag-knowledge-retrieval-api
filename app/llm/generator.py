"""
LLM Answer Generator
=====================
Uses LangChain to:
  - Build a structured RAG prompt with injected context chunks
  - Call OpenAI with configured model / temperature / tokens
  - Return the answer with token usage
"""
from __future__ import annotations
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.models.schemas import RetrievedChunk

settings = get_settings()
log      = get_logger("llm.generator")

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model          = settings.llm_model,
            temperature    = settings.llm_temperature,
            max_tokens     = settings.llm_max_tokens,
            openai_api_key = settings.openai_api_key,
        )
        log.info("llm_initialized", model=settings.llm_model)
    return _llm


# ── Prompt template ───────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an expert Salesforce support assistant helping \
customer service agents resolve cases quickly and accurately.

RULES:
1. Answer ONLY using the knowledge base context provided below.
2. If the context does not contain enough information, say:
   "I could not find a specific answer in the knowledge base. \
Please escalate to a specialist."
3. Be concise and professional. Maximum 3 short paragraphs.
4. If you recommend an action, be specific and actionable.
5. Do NOT mention that you are an AI or reference your training data.
6. Do NOT make up information not present in the context.

KNOWLEDGE BASE CONTEXT:
{context}
"""

RAG_USER_PROMPT = """Case ID: {case_id}
{case_context}
Question: {question}

Please provide a clear, actionable answer based on the knowledge base above."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human",  RAG_USER_PROMPT),
])


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """
    Formats retrieved chunks into a numbered context string.
    Respects max_chars to avoid exceeding the LLM context window.
    """
    if not chunks:
        return "No relevant knowledge base articles found."

    parts  = []
    total  = 0
    for i, chunk in enumerate(chunks, 1):
        entry = (
            f"[{i}] Source: {chunk.source} (relevance: {chunk.score:.2f})\n"
            f"{chunk.text}\n"
        )
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n---\n".join(parts)


# ── Main generation function ──────────────────────────────────────────────────

def generate_answer(
    question     : str,
    chunks       : list[RetrievedChunk],
    case_id      : str,
    case_context : str = "",
) -> dict:
    """
    Generates an answer using LangChain RAG chain.

    Returns:
        {
          "answer"     : str,
          "tokens_used": int,
          "model_used" : str,
        }
    """
    llm     = _get_llm()
    context = _build_context(chunks)

    # Build LangChain chain: prompt → LLM → string output
    chain = _PROMPT | llm | StrOutputParser()

    log.info("llm_generating",
             case_id=case_id, chunks_used=len(chunks),
             question=question[:60])

    # Invoke with callback to capture token usage
    response = llm.invoke(
        _PROMPT.format_messages(
            context      = context,
            case_id      = case_id,
            case_context = f"Case Context: {case_context}" if case_context else "",
            question     = question,
        )
    )

    answer      = response.content.strip()
    tokens_used = 0

    # Extract token usage from response metadata
    if hasattr(response, "response_metadata"):
        usage       = response.response_metadata.get("token_usage", {})
        tokens_used = usage.get("total_tokens", 0)

    log.info("llm_answer_generated",
             case_id=case_id, tokens=tokens_used,
             answer_length=len(answer))

    return {
        "answer"     : answer,
        "tokens_used": tokens_used,
        "model_used" : settings.llm_model,
    }
