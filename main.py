"""
Enterprise RAG AI Gateway — FastAPI Application
================================================
Entry point. Configures:
  - Logging
  - CORS
  - Lifespan (startup / shutdown events)
  - Routes
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.config import get_settings
from app.utils.logger import configure_logging, get_logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    configure_logging()
    log = get_logger("main")
    log.info("startup",
             env=settings.app_env,
             model=settings.llm_model,
             index=settings.pinecone_index_name)

    # Pre-warm the LangGraph compiled graph
    from app.agents.rag_graph import get_rag_graph
    get_rag_graph()
    log.info("langgraph_ready")

    yield

    log.info("shutdown")


app = FastAPI(
    title       = "Enterprise RAG AI Gateway",
    description = (
        "Production-grade RAG pipeline for Salesforce.\n\n"
        "Stack: FastAPI · LangGraph · LangChain · Pinecone · Redis · OpenAI\n\n"
        "Flow: Salesforce → Guardrails → Redis Cache → "
        "Pinecone Retrieval → LLM → Output Validation → Response"
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = settings.app_port,
        reload  = settings.app_env == "development",
    )
