"""
LangGraph RAG Orchestrator
==========================
Implements the full pipeline as a LangGraph StateGraph.

Node execution order:
  input_guardrails
       │
       ├── BLOCKED → blocked_response
       │
  cache_lookup
       │
       ├── HIT → return_cached
       │
  vector_retrieval   (Pinecone)
       │
  llm_generation     (LangChain + OpenAI)
       │
  output_guardrails
       │
       ├── BLOCKED → blocked_response
       │
  cache_write        (Redis)
       │
  format_response

Each node reads from and writes to AgentState.
LangGraph routes between nodes based on state conditions.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import StateGraph, END

from app.models.schemas import AgentState, SafetyStatus, CacheStatus, RAGQueryResponse
from app.guardrails.input_guard  import run_input_guardrails
from app.guardrails.output_guard import run_output_guardrails
from app.cache.redis_cache       import cache_get, cache_set
from app.retrieval.pinecone_store import retrieve_chunks
from app.llm.generator           import generate_answer
from app.utils.config            import get_settings
from app.utils.logger            import get_logger

settings = get_settings()
log      = get_logger("agents.rag_graph")


# ════════════════════════════════════════════════════════════════════════════
#  NODE FUNCTIONS
#  Each function receives the current AgentState dict and returns
#  a dict of fields to update (LangGraph merges this into the state).
# ════════════════════════════════════════════════════════════════════════════

def node_input_guardrails(state: AgentState) -> dict:
    """
    Node 1: Run input guardrails on the original question.
    Sets input_safe, clean_question, pii_redacted, blocked_reason.
    """
    log.info("node_input_guardrails", case_id=state.case_id)

    result = run_input_guardrails(
        text    = state.original_question,
        case_id = state.case_id,
    )

    return {
        "input_safe"    : result.is_safe,
        "clean_question": result.clean_text if result.is_safe else state.original_question,
        "pii_redacted"  : result.pii_redacted,
        "blocked_reason": result.blocked_reason,
    }


def node_cache_lookup(state: AgentState) -> dict:
    """
    Node 2: Check Redis for a cached answer.
    Sets cache_hit and cached_answer.
    """
    log.info("node_cache_lookup",
             case_id=state.case_id, domain=state.domain)

    cached = cache_get(
        question = state.clean_question,
        domain   = state.domain,
    )

    if cached:
        log.info("cache_hit", case_id=state.case_id)
        return {"cache_hit": True, "cached_answer": cached}

    return {"cache_hit": False}


def node_vector_retrieval(state: AgentState) -> dict:
    """
    Node 3: Query Pinecone for relevant knowledge base chunks.
    Sets retrieved_chunks, retrieval_done.
    """
    log.info("node_vector_retrieval",
             case_id=state.case_id, domain=state.domain,
             top_k=state.top_k)

    chunks = retrieve_chunks(
        question = state.clean_question,
        domain   = state.domain,
        top_k    = state.top_k,
    )

    return {
        "retrieved_chunks": [c.model_dump() for c in chunks],
        "retrieval_done"  : True,
    }


def node_llm_generation(state: AgentState) -> dict:
    """
    Node 4: Generate answer using LangChain + OpenAI.
    Sets answer, tokens_used, model_used.
    """
    from app.models.schemas import RetrievedChunk

    log.info("node_llm_generation",
             case_id=state.case_id,
             chunks_available=len(state.retrieved_chunks))

    # Reconstruct chunk objects
    chunks = [RetrievedChunk(**c) for c in state.retrieved_chunks]

    result = generate_answer(
        question     = state.clean_question,
        chunks       = chunks,
        case_id      = state.case_id,
        case_context = state.context,
    )

    return {
        "answer"     : result["answer"],
        "tokens_used": result["tokens_used"],
        "model_used" : result["model_used"],
    }


def node_output_guardrails(state: AgentState) -> dict:
    """
    Node 5: Validate the LLM answer before returning.
    Sets output_safe, final_answer, blocked_reason.
    """
    log.info("node_output_guardrails", case_id=state.case_id)

    result = run_output_guardrails(
        answer           = state.answer,
        retrieved_chunks = state.retrieved_chunks,
        case_id          = state.case_id,
    )

    return {
        "output_safe" : result.is_valid,
        "final_answer": result.final_text if result.is_valid else "",
        "blocked_reason": result.blocked_reason if not result.is_valid else state.blocked_reason,
    }


def node_cache_write(state: AgentState) -> dict:
    """
    Node 6: Write the validated answer to Redis for future cache hits.
    """
    if state.final_answer:
        cache_set(
            question = state.clean_question,
            domain   = state.domain,
            answer   = state.final_answer,
        )
        log.info("cache_written", case_id=state.case_id, domain=state.domain)
    return {"final_answer": state.final_answer}


# ════════════════════════════════════════════════════════════════════════════
#  ROUTING FUNCTIONS (conditional edges in LangGraph)
# ════════════════════════════════════════════════════════════════════════════

def route_after_input_guard(state: AgentState) -> Literal["cache_lookup", "blocked"]:
    return "cache_lookup" if state.input_safe else "blocked"


def route_after_cache(state: AgentState) -> Literal["vector_retrieval", "return_cached"]:
    return "return_cached" if state.cache_hit else "vector_retrieval"


def route_after_output_guard(state: AgentState) -> Literal["cache_write", "blocked"]:
    return "cache_write" if state.output_safe else "blocked"


# ════════════════════════════════════════════════════════════════════════════
#  BUILD THE GRAPH
# ════════════════════════════════════════════════════════════════════════════

def build_rag_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.
    Called once at application startup.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("input_guardrails",  node_input_guardrails)
    graph.add_node("cache_lookup",      node_cache_lookup)
    graph.add_node("vector_retrieval",  node_vector_retrieval)
    graph.add_node("llm_generation",    node_llm_generation)
    graph.add_node("output_guardrails", node_output_guardrails)
    graph.add_node("cache_write",       node_cache_write)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("input_guardrails")

    # ── Conditional edges ──────────────────────────────────────────────────
    graph.add_conditional_edges(
        "input_guardrails",
        route_after_input_guard,
        {"cache_lookup": "cache_lookup", "blocked": END}
    )

    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {"return_cached": END, "vector_retrieval": "vector_retrieval"}
    )

    # ── Linear edges ───────────────────────────────────────────────────────
    graph.add_edge("vector_retrieval",  "llm_generation")
    graph.add_edge("llm_generation",    "output_guardrails")

    graph.add_conditional_edges(
        "output_guardrails",
        route_after_output_guard,
        {"cache_write": "cache_write", "blocked": END}
    )

    graph.add_edge("cache_write", END)

    return graph.compile()


# ── Compiled graph singleton ──────────────────────────────────────────────────
_compiled_graph = None

def get_rag_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
        log.info("langgraph_compiled")
    return _compiled_graph


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

async def run_rag_pipeline(
    case_id          : str,
    question         : str,
    domain           : str  = "general",
    top_k            : int  = 5,
    context          : str  = "",
) -> RAGQueryResponse:
    """
    Main entry point called by the FastAPI route handler.
    Runs the full LangGraph pipeline and returns a RAGQueryResponse.
    """
    start_ms = int(time.time() * 1000)

    # Build initial state
    initial_state = AgentState(
        case_id           = case_id,
        original_question = question,
        clean_question    = question,
        domain            = domain,
        top_k             = top_k,
        context           = context,
    )

    log.info("pipeline_start",
             case_id=case_id, domain=domain, question=question[:60])

    # Run LangGraph
    graph       = get_rag_graph()
    final_state = await graph.ainvoke(initial_state.model_dump())

    latency_ms  = int(time.time() * 1000) - start_ms

    # ── Determine outcome ─────────────────────────────────────────────────
    state = AgentState(**final_state)

    if not state.input_safe or not state.output_safe:
        answer        = "[This query could not be processed due to safety restrictions.]"
        safety_status = SafetyStatus.BLOCKED
    elif state.cache_hit:
        answer        = state.cached_answer
        safety_status = SafetyStatus.PASSED
    else:
        answer        = state.final_answer or state.answer
        safety_status = SafetyStatus.PASSED

    cache_status = CacheStatus.HIT if state.cache_hit else CacheStatus.MISS

    from app.models.schemas import RetrievedChunk
    sources = [RetrievedChunk(**c) for c in state.retrieved_chunks]

    log.info("pipeline_complete",
             case_id=case_id, latency_ms=latency_ms,
             cache=cache_status, safety=safety_status,
             tokens=state.tokens_used)

    return RAGQueryResponse(
        case_id       = case_id,
        answer        = answer,
        sources       = sources,
        safety_status = safety_status,
        cache_status  = cache_status,
        model_used    = state.model_used or settings.llm_model,
        tokens_used   = state.tokens_used,
        latency_ms    = latency_ms,
        pii_redacted  = state.pii_redacted,
        generated_at  = datetime.now(timezone.utc).isoformat(),
    )
