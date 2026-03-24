"""
FastAPI Route Handlers
======================
Exposes:
  POST /query          — Main RAG endpoint (called by Salesforce)
  POST /ingest         — Ingest KB articles into Pinecone
  GET  /health         — System health check
  GET  /               — Service info
"""
from __future__ import annotations
import hmac
import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import RAGQueryRequest, RAGQueryResponse, IngestRequest
from app.agents.rag_graph import run_rag_pipeline
from app.retrieval.pinecone_store import ingest_articles, pinecone_health
from app.cache.redis_cache import cache_health
from app.utils.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
log      = get_logger("api.routes")
router   = APIRouter()


# ── Security ──────────────────────────────────────────────────────────────────

async def verify_api_key(request: Request) -> bool:
    """HMAC constant-time API key verification."""
    incoming = request.headers.get("X-API-Key", "")
    expected = settings.gateway_api_key

    valid = hmac.compare_digest(
        hashlib.sha256(incoming.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    )
    if not valid:
        log.warning("unauthorized_request", ip=request.client.host)
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="RAG Query — called by Salesforce Apex / Agentforce",
)
async def query(
    body : RAGQueryRequest,
    _auth: bool = Depends(verify_api_key),
):
    """
    Main endpoint. Salesforce sends a question + case context.
    The LangGraph pipeline orchestrates: guardrails → cache →
    Pinecone retrieval → LLM → output validation → cache write.
    """
    log.info("query_received",
             case_id=body.case_id, domain=body.domain,
             question=body.question[:60])

    try:
        return await run_rag_pipeline(
            case_id  = body.case_id,
            question = body.question,
            domain   = body.domain.value,
            top_k    = body.top_k or settings.pinecone_top_k,
            context  = body.context or "",
        )
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post(
    "/ingest",
    summary="Ingest knowledge base articles into Pinecone",
)
async def ingest(
    body : IngestRequest,
    _auth: bool = Depends(verify_api_key),
):
    """
    Ingests knowledge base articles (e.g. Salesforce Knowledge articles
    exported as JSON) into the Pinecone index.
    Chunks each article and stores embeddings with metadata.
    """
    log.info("ingest_request",
             article_count=len(body.articles), domain=body.domain)

    stats = ingest_articles(body.articles)
    return {"status": "ok", **stats}


@router.get("/health", summary="Health check")
async def health():
    """Returns status of all dependencies."""
    pc_health    = pinecone_health()
    cache_status = cache_health()

    overall = (
        "healthy"
        if pc_health.get("status") == "connected"
        else "degraded"
    )

    return {
        "status"       : overall,
        "timestamp"    : datetime.now(timezone.utc).isoformat(),
        "pinecone"     : pc_health,
        "cache"        : cache_status,
        "llm_model"    : settings.llm_model,
        "embedding"    : settings.embedding_model,
        "environment"  : settings.app_env,
    }


@router.get("/", summary="Service info")
async def root():
    return {
        "service"   : "Enterprise RAG AI Gateway",
        "version"   : "2.0.0",
        "stack"     : ["FastAPI", "LangGraph", "LangChain",
                       "Pinecone", "Redis", "OpenAI"],
        "endpoints" : ["/query", "/ingest", "/health", "/docs"],
    }
