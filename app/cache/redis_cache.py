"""
Redis Semantic Cache Layer
==========================
Wraps Redis with two caching strategies:

  1. Semantic cache  — embeds the query and checks cosine similarity
                       against cached query vectors. Similar questions
                       (score >= threshold) return the cached answer.
                       Uses LangChain's RedisSemanticCache under the hood.

  2. Exact key cache — simple Redis GET/SET on a hash of the query string.
                       Used as a secondary fast-path if semantic cache misses.

  3. In-memory fallback — if Redis is unreachable, uses a simple dict cache
                          so the system degrades gracefully.
"""
from __future__ import annotations
import hashlib
import json
import time
from typing import Optional

from app.utils.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
log      = get_logger("cache.redis")


# ── In-memory fallback ────────────────────────────────────────────────────────

_memory_cache: dict[str, tuple[str, float]] = {}   # key → (value, expiry_epoch)


def _memory_get(key: str) -> Optional[str]:
    if key in _memory_cache:
        value, expiry = _memory_cache[key]
        if time.time() < expiry:
            return value
        del _memory_cache[key]
    return None


def _memory_set(key: str, value: str, ttl: int) -> None:
    _memory_cache[key] = (value, time.time() + ttl)


# ── Cache key ─────────────────────────────────────────────────────────────────

def _make_key(question: str, domain: str) -> str:
    """Deterministic cache key from question + domain."""
    raw = f"{domain}::{question.lower().strip()}"
    return "rag:exact:" + hashlib.sha256(raw.encode()).hexdigest()


# ── Redis client singleton ────────────────────────────────────────────────────

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            _redis_client.ping()
            log.info("redis_connected", url=settings.redis_url)
        except Exception as e:
            log.warning("redis_unavailable", error=str(e))
            _redis_client = None
    return _redis_client


# ── Semantic cache via LangChain ─────────────────────────────────────────────

_semantic_cache = None

def _get_semantic_cache():
    """
    Returns a LangChain RedisSemanticCache instance (lazy init).
    Falls back to None if Redis or OpenAI not available.
    """
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache
    try:
        from langchain_community.cache import RedisSemanticCache
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        _semantic_cache = RedisSemanticCache(
            redis_url=settings.redis_url,
            embedding=embeddings,
            score_threshold=settings.redis_semantic_score_threshold,
        )
        log.info("semantic_cache_initialized")
        return _semantic_cache
    except Exception as e:
        log.warning("semantic_cache_init_failed", error=str(e))
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def cache_get(question: str, domain: str) -> Optional[str]:
    """
    Try to get a cached answer.
    Order: Redis exact → in-memory fallback.
    (Semantic cache is handled separately via LangChain's set_llm_cache.)
    Returns the cached answer string, or None on miss.
    """
    key = _make_key(question, domain)

    # Try Redis exact cache
    r = _get_redis()
    if r:
        try:
            value = r.get(key)
            if value:
                log.info("cache_hit_redis_exact",
                         question=question[:60], domain=domain)
                return value
        except Exception as e:
            log.warning("redis_get_error", error=str(e))

    # Try in-memory fallback
    if settings.cache_fallback_enabled:
        value = _memory_get(key)
        if value:
            log.info("cache_hit_memory", question=question[:60])
            return value

    return None


def cache_set(question: str, domain: str, answer: str) -> None:
    """
    Store an answer in Redis (exact key) + memory fallback.
    """
    key = _make_key(question, domain)
    ttl = settings.redis_ttl_seconds

    # Write to Redis
    r = _get_redis()
    if r:
        try:
            r.setex(key, ttl, answer)
            log.info("cache_set_redis",
                     question=question[:60], domain=domain, ttl=ttl)
        except Exception as e:
            log.warning("redis_set_error", error=str(e))

    # Always write to memory fallback
    if settings.cache_fallback_enabled:
        _memory_set(key, answer, ttl)


def cache_invalidate(question: str, domain: str) -> None:
    """Removes a specific entry from cache (e.g. when KB article is updated)."""
    key = _make_key(question, domain)
    r = _get_redis()
    if r:
        try:
            r.delete(key)
        except Exception:
            pass
    _memory_cache.pop(key, None)


def cache_health() -> dict:
    """Returns cache health status for the /health endpoint."""
    r = _get_redis()
    redis_ok = False
    if r:
        try:
            r.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "redis_connected"  : redis_ok,
        "memory_entries"   : len(_memory_cache),
        "semantic_cache"   : _semantic_cache is not None,
    }
