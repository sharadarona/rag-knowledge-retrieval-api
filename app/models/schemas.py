"""
Pydantic models — request/response schemas used across the entire system.
"""
from __future__ import annotations
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import re


# ── Enums ────────────────────────────────────────────────────────────────────

class KnowledgeDomain(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"

class SafetyStatus(str, Enum):
    PASSED  = "PASSED"
    BLOCKED = "BLOCKED"

class CacheStatus(str, Enum):
    HIT  = "HIT"
    MISS = "MISS"


# ── Inbound: from Salesforce ──────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """Payload sent by Salesforce Apex / Agentforce to the FastAPI Gateway."""

    case_id      : str  = Field(..., min_length=15, max_length=18,
                                description="Salesforce Case ID (15 or 18 chars)")
    question     : str  = Field(..., min_length=3,  max_length=1000,
                                description="Consumer or agent question")
    domain       : KnowledgeDomain = Field(
                                default=KnowledgeDomain.GENERAL,
                                description="Knowledge domain to search in Pinecone")
    context      : Optional[str] = Field(
                                default=None, max_length=2000,
                                description="Optional extra context (e.g. case description)")
    top_k        : Optional[int] = Field(
                                default=5, ge=1, le=20,
                                description="Number of Pinecone chunks to retrieve")

    @field_validator("case_id")
    @classmethod
    def validate_sf_id(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9]{15,18}$', v):
            raise ValueError("Invalid Salesforce ID format")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank")
        return v.strip()


# ── Retrieved chunk ───────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """Single chunk returned from Pinecone."""
    chunk_id  : str
    text      : str
    score     : float
    source    : str
    domain    : str
    metadata  : dict = Field(default_factory=dict)


# ── Outbound: to Salesforce ───────────────────────────────────────────────────

class RAGQueryResponse(BaseModel):
    """Full structured response returned to Salesforce."""
    case_id         : str
    answer          : str
    sources         : List[RetrievedChunk]
    safety_status   : SafetyStatus
    cache_status    : CacheStatus
    model_used      : str
    tokens_used     : int
    latency_ms      : int
    pii_redacted    : bool
    generated_at    : str


# ── Index ingest ──────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Payload for ingesting knowledge articles into Pinecone."""
    articles : List[KBArticle]
    domain   : KnowledgeDomain

class KBArticle(BaseModel):
    article_id  : str
    title       : str
    body        : str
    domain      : KnowledgeDomain
    metadata    : dict = Field(default_factory=dict)


# ── Internal agent state (used by LangGraph) ─────────────────────────────────

class AgentState(BaseModel):
    """Mutable state object passed through every LangGraph node."""
    # Input
    case_id           : str
    original_question : str
    clean_question    : str  = ""
    domain            : str  = "general"
    top_k             : int  = 5
    context           : str  = ""

    # Guardrails
    input_safe        : bool = True
    pii_redacted      : bool = False
    blocked_reason    : str  = ""

    # Cache
    cache_hit         : bool = False
    cached_answer     : str  = ""

    # Retrieval
    retrieved_chunks  : List[dict] = Field(default_factory=list)
    retrieval_done    : bool = False

    # LLM
    answer            : str  = ""
    tokens_used       : int  = 0
    model_used        : str  = ""

    # Output
    output_safe       : bool = True
    final_answer      : str  = ""

    # Metadata
    latency_ms        : int  = 0
    error             : str  = ""
    retry_count       : int  = 0
