"""
Pinecone Vector Store
=====================
Manages:
  - Index creation and connection
  - Namespace-per-domain routing (billing / technical / general)
  - Embedding generation via OpenAI
  - Similarity search with metadata filtering
  - Knowledge base article ingestion (chunking + upsert)
"""
from __future__ import annotations
import uuid
import time
from typing import Optional

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.models.schemas import RetrievedChunk, KBArticle

settings = get_settings()
log      = get_logger("retrieval.pinecone")

# ── Lazy singletons ───────────────────────────────────────────────────────────
_pinecone_index  = None
_embeddings_model = None


def _get_embeddings():
    global _embeddings_model
    if _embeddings_model is None:
        from langchain_openai import OpenAIEmbeddings
        _embeddings_model = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        log.info("embeddings_model_loaded", model=settings.embedding_model)
    return _embeddings_model


def _get_index():
    """Connect to Pinecone index, create if not exists."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        log.info("creating_pinecone_index", name=settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
        log.info("pinecone_index_ready", name=settings.pinecone_index_name)

    _pinecone_index = pc.Index(settings.pinecone_index_name)
    log.info("pinecone_connected", index=settings.pinecone_index_name)
    return _pinecone_index


def _domain_to_namespace(domain: str) -> str:
    """Maps domain string to Pinecone namespace."""
    return settings.namespace_map.get(domain, settings.pinecone_ns_general)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for better retrieval coverage.
    chunk_size and overlap are in characters.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_articles(articles: list[KBArticle]) -> dict:
    """
    Ingests knowledge base articles into Pinecone.
    Steps: chunk → embed → upsert with metadata.
    Returns stats dict.
    """
    index      = _get_index()
    embeddings = _get_embeddings()

    total_vectors = 0
    errors        = 0

    for article in articles:
        try:
            namespace = _domain_to_namespace(article.domain)
            chunks    = _chunk_text(article.body)

            vectors = []
            for i, chunk in enumerate(chunks):
                chunk_id  = f"{article.article_id}_chunk_{i}"
                embedding = embeddings.embed_query(chunk)
                vectors.append({
                    "id"     : chunk_id,
                    "values" : embedding,
                    "metadata": {
                        "article_id"  : article.article_id,
                        "title"       : article.title,
                        "chunk_index" : i,
                        "chunk_text"  : chunk,
                        "domain"      : article.domain,
                        "source"      : article.title,
                        **article.metadata,
                    }
                })

            # Upsert in batches of 100 (Pinecone limit)
            batch_size = 100
            for j in range(0, len(vectors), batch_size):
                batch = vectors[j:j + batch_size]
                index.upsert(vectors=batch, namespace=namespace)

            total_vectors += len(vectors)
            log.info("article_ingested",
                     article_id=article.article_id, chunks=len(chunks),
                     namespace=namespace)

        except Exception as e:
            errors += 1
            log.error("article_ingest_failed",
                      article_id=article.article_id, error=str(e))

    return {
        "articles_processed": len(articles),
        "vectors_upserted"  : total_vectors,
        "errors"            : errors,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    domain: str,
    top_k: int = 5,
    metadata_filter: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Embeds the question and queries Pinecone for the top-k
    most similar chunks in the specified namespace/domain.

    Returns a list of RetrievedChunk sorted by score descending.
    """
    index      = _get_index()
    embeddings = _get_embeddings()
    namespace  = _domain_to_namespace(domain)

    log.info("pinecone_query",
             domain=domain, namespace=namespace, top_k=top_k,
             question=question[:60])

    try:
        query_vector = embeddings.embed_query(question)

        query_params = {
            "vector"         : query_vector,
            "top_k"          : top_k,
            "namespace"      : namespace,
            "include_metadata": True,
        }
        if metadata_filter:
            query_params["filter"] = metadata_filter

        result  = index.query(**query_params)
        matches = result.get("matches", [])

        chunks = []
        for m in matches:
            meta = m.get("metadata", {})
            chunks.append(RetrievedChunk(
                chunk_id = m["id"],
                text     = meta.get("chunk_text", ""),
                score    = round(float(m.get("score", 0.0)), 4),
                source   = meta.get("source", meta.get("title", "Unknown")),
                domain   = meta.get("domain", domain),
                metadata = meta,
            ))

        log.info("pinecone_results",
                 count=len(chunks),
                 top_score=chunks[0].score if chunks else 0)
        return chunks

    except Exception as e:
        log.error("pinecone_query_failed", error=str(e))
        return []


def pinecone_health() -> dict:
    """Returns Pinecone index stats for the /health endpoint."""
    try:
        index = _get_index()
        stats = index.describe_index_stats()
        return {
            "status"         : "connected",
            "index"          : settings.pinecone_index_name,
            "total_vectors"  : stats.get("total_vector_count", 0),
            "namespaces"     : list(stats.get("namespaces", {}).keys()),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
