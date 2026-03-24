"""
Central configuration — loaded once at startup from .env file.
All components import settings from here.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── Security ─────────────────────────────────────────────────
    gateway_api_key         : str  = "change-me"

    # ── OpenAI ───────────────────────────────────────────────────
    openai_api_key          : str  = ""
    llm_model               : str  = "gpt-4o-mini"
    embedding_model         : str  = "text-embedding-3-small"
    llm_temperature         : float = 0.1
    llm_max_tokens          : int   = 600

    # ── Pinecone ─────────────────────────────────────────────────
    pinecone_api_key        : str  = ""
    pinecone_index_name     : str  = "salesforce-kb"
    pinecone_dimension      : int  = 1536
    pinecone_metric         : str  = "cosine"
    pinecone_cloud          : str  = "aws"
    pinecone_region         : str  = "us-east-1"
    pinecone_ns_billing     : str  = "billing"
    pinecone_ns_technical   : str  = "technical"
    pinecone_ns_general     : str  = "general"
    pinecone_top_k          : int  = 5

    # ── Redis ────────────────────────────────────────────────────
    redis_url               : str  = "redis://localhost:6379"
    redis_ttl_seconds       : int  = 3600
    redis_semantic_score_threshold : float = 0.95
    cache_fallback_enabled  : bool = True

    # ── LangGraph ────────────────────────────────────────────────
    langgraph_max_retries   : int  = 3
    langgraph_timeout_seconds : int = 30

    # ── Guardrails ───────────────────────────────────────────────
    guardrails_enabled      : bool = True
    toxicity_threshold      : float = 0.7
    pii_entities            : str  = "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD"

    # ── App ──────────────────────────────────────────────────────
    app_env                 : str  = "development"
    log_level               : str  = "INFO"
    app_port                : int  = 8000

    @property
    def pii_entity_list(self) -> list[str]:
        return [e.strip() for e in self.pii_entities.split(",") if e.strip()]

    @property
    def namespace_map(self) -> dict[str, str]:
        return {
            "billing"  : self.pinecone_ns_billing,
            "technical": self.pinecone_ns_technical,
            "general"  : self.pinecone_ns_general,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
