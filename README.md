# rag-knowledge-retrieval-api
 This is second/backend part of project2-rag-knowledge-retriever.This implements the RAG for 
 Salesforce knowledge base and provides output based upon request from Salesforce using various gen AI technologies and tools.


---

## Architecture Overview

```
Salesforce (Agentforce / Apex / Flow)
          │
          │  POST /query   X-API-Key header
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI AI Gateway                        │
│          Auth (HMAC) · Rate limiting · Routing              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           LangGraph Agent (StateGraph Orchestrator)         │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ Input Guardrails│  Injection · PII redaction · Toxicity  │
│  └────────┬────────┘                                        │
│           │ safe                    blocked ──► END          │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  Redis Cache    │  Semantic + exact key cache             │
│  └────────┬────────┘                                        │
│           │ miss                    hit ──────► END          │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ Pinecone Query  │  Embed → namespace → top-K chunks       │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  LLM Generation │  LangChain · context injection          │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │Output Guardrails│  Safety · PII leak · hallucination      │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  Cache Write    │  Store answer in Redis for next hit     │
│  └────────┬────────┘                                        │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
  JSON Response → Salesforce
  { answer, sources, safety_status, cache_status,
    model_used, tokens_used, latency_ms, pii_redacted }
```

---

## Project Structure

```
rag_enterprise/
├── main.py                              ← FastAPI app entrypoint
├── requirements.txt
├── .env.example                         ← copy to .env
│
├── app/
│   ├── api/
│   │   └── routes.py                   ← /query  /ingest  /health
│   │
│   ├── agents/
│   │   └── rag_graph.py                ← LangGraph StateGraph (all nodes)
│   │
│   ├── guardrails/
│   │   ├── input_guard.py              ← Injection · PII · Toxicity
│   │   └── output_guard.py             ← Safety · leak · hallucination
│   │
│   ├── cache/
│   │   └── redis_cache.py              ← Semantic + exact + memory fallback
│   │
│   ├── retrieval/
│   │   └── pinecone_store.py           ← Ingest · namespace routing · query
│   │
│   ├── llm/
│   │   └── generator.py                ← LangChain RAG chain · prompt
│   │
│   ├── models/
│   │   └── schemas.py                  ← All Pydantic models + AgentState
│   │
│   └── utils/
│       ├── config.py                   ← Settings (pydantic-settings)
│       └── logger.py                   ← Structured logging (structlog)
│
├── salesforce/
│   └── RAGGatewayQueueable.cls         ← Apex Queueable + InvocableMethod
│
├── scripts/
│   └── ingest_sample_kb.py             ← Load sample KB articles to Pinecone
│
└── tests/
    └── test_guardrails.py              ← 34 tests (no external deps needed)
```

---

## Quick Start

### Step 1 — Clone and set up environment
```bash
cd rag_enterprise
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY, PINECONE_API_KEY, GATEWAY_API_KEY
```

### Step 2 — Start Redis locally
```bash
# macOS
brew install redis && brew services start redis

# Ubuntu/Debian
sudo apt install redis-server && sudo service redis start

# Docker (simplest)
docker run -d -p 6379:6379 redis:alpine
```

### Step 3 — Ingest sample knowledge base articles
```bash
python scripts/ingest_sample_kb.py
```
This creates the Pinecone index (if it doesn't exist) and uploads
5 sample KB articles across billing, technical, and general namespaces.

### Step 4 — Start the server
```bash
uvicorn main:app --reload --port 8000
```

### Step 5 — Test the endpoint
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-gateway-api-key" \
  -d '{
    "case_id"  : "500Hs00001XyZaABC",
    "question" : "How do I reset a customer portal password?",
    "domain"   : "technical",
    "top_k"    : 5
  }'
```

Expected response:
```json
{
  "case_id"       : "500Hs00001XyZaABC",
  "answer"        : "To reset a customer portal password...",
  "sources"       : [{"chunk_id": "KB001_chunk_0", "score": 0.94, ...}],
  "safety_status" : "PASSED",
  "cache_status"  : "MISS",
  "model_used"    : "gpt-4o-mini",
  "tokens_used"   : 312,
  "latency_ms"    : 1847,
  "pii_redacted"  : false,
  "generated_at"  : "2026-03-16T10:30:00+00:00"
}
```

Ask the same question a second time — `cache_status` becomes `"HIT"` and `latency_ms` drops to < 5ms.

---

## LangGraph Node Flow (Step by Step)

| # | Node | What it does | On failure |
|---|------|---|---|
| 1 | `input_guardrails` | Injection check · PII redact · toxicity score | → `END` (BLOCKED) |
| 2 | `cache_lookup` | Redis exact key lookup | → continue |
| — | — | Cache HIT → return immediately | → `END` (HIT) |
| 3 | `vector_retrieval` | Embed question → Pinecone query → top-K chunks | empty list |
| 4 | `llm_generation` | Build RAG prompt → LangChain → OpenAI | exception |
| 5 | `output_guardrails` | Safety check · PII leak · hallucination heuristic | → `END` (BLOCKED) |
| 6 | `cache_write` | Store answer in Redis for future hits | silent fail |

---

## Pinecone Namespace Strategy

One Pinecone index, three namespaces — each KB article is routed
to the correct namespace at ingest time and queries only search
the relevant namespace:

| Domain value | Pinecone namespace | Articles |
|---|---|---|
| `technical` | `technical` | Password reset, API errors, integrations |
| `billing` | `billing` | Invoices, refunds, plan upgrades |
| `general` | `general` | GDPR, onboarding, account management |

---

## Redis Caching

Two layers, automatic fallback:

**Layer 1 — Exact Redis cache**
SHA-256 hash of `domain::question.lower()` as the key.
TTL: 3600 seconds (configurable via `REDIS_TTL_SECONDS`).
Same question from different cases = same cached answer.

**Layer 2 — In-memory fallback**
Python dict with expiry timestamps. Used when Redis is
unavailable so the system never crashes — it just skips caching.

**Semantic cache** (LangChain RedisSemanticCache)
Embeds the query and checks cosine similarity against cached queries.
If score >= `REDIS_SEMANTIC_SCORE_THRESHOLD` (default 0.95),
returns the cached answer for semantically similar questions.

---

## Salesforce Integration

### Named Credential
```
Label  : RAG AI Gateway
Name   : RAG_AI_Gateway
URL    : https://your-app.onrender.com
Header : X-API-Key = your-GATEWAY_API_KEY-value
```

### Custom Fields on Case (add to your SFDX package)
| API Name | Type | Purpose |
|---|---|---|
| `AI_RAG_Answer__c` | LongTextArea 32768 | RAG-generated answer |
| `AI_Model_Used__c` | Text 100 | LLM model name |
| `AI_Cache_Status__c` | Text 10 | HIT or MISS |
| `AI_Summary_Generated_At__c` | DateTime | Timestamp |

### Apex Usage
```apex
// From a Flow Action or direct Apex:
List<RAGGatewayQueueable.RAGInput> inputs = new List<RAGGatewayQueueable.RAGInput>();
RAGGatewayQueueable.RAGInput inp = new RAGGatewayQueueable.RAGInput();
inp.caseId   = case.Id;
inp.question = 'How do I reset the customer portal password?';
inp.domain   = 'technical';
inputs.add(inp);
RAGGatewayQueueable.queryRAG(inputs);
```

---

## Deploy to Production (Render.com)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "Enterprise RAG v2"
git push origin main

# 2. Create Web Service on render.com
#    Build command : pip install -r requirements.txt
#    Start command : uvicorn main:app --host 0.0.0.0 --port $PORT

# 3. Add environment variables in Render dashboard
#    (copy all values from your .env file)

# 4. Add a Redis instance
#    Render Dashboard → New → Redis
#    Copy the Redis URL into REDIS_URL env var

# 5. Update Salesforce Named Credential with the Render HTTPS URL
```

---

## Run Tests
```bash
pytest tests/test_guardrails.py -v
# Expected: 34 passed
```

---

## Cost Estimate

| Component | Cost |
|---|---|
| OpenAI embeddings | ~$0.00002 per 1K tokens (~free for dev) |
| OpenAI GPT-4o-mini per query | ~$0.0001–0.0003 |
| Pinecone Serverless | Free tier: 100K vectors, 2M queries/month |
| Redis (Render) | Free tier: 25MB (holds ~10K cached answers) |
| **1,000 queries/month** | **~$0.10–0.30 total** |
