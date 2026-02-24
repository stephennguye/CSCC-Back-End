# Research: AI Call Center Backend

**Phase**: 0 — Outline & Research
**Branch**: `001-ai-call-center-backend`
**Date**: 2026-02-24
**Status**: Complete — all unknowns resolved

---

## 1. Primary Database

**Decision**: PostgreSQL (asyncpg + SQLAlchemy async)

**Rationale**:
- Strong relational model for CallSession → Message → Claim / Reminder relationships with referential integrity
- Native JSONB for flexible metadata without sacrificing queryability (useful for claim fields that may expand)
- asyncpg delivers non-blocking I/O compatible with the all-async mandate (Constitution II, VII)
- SQLAlchemy async ORM abstracts connection pooling and schema migrations (Alembic)
- Future analytics (aggregates, window functions, full-text search) are first-class PostgreSQL features
- pg_crypto extension available for column-level encryption to satisfy FR-028/FR-029 (PII at rest)

**Alternatives considered**:
- MySQL/MariaDB: rejected — inferior JSONB support and asyncpg ecosystem is more mature
- MongoDB: rejected — document store complicates relational integrity between Session→Claim→Reminder
- SQLite: rejected — not suitable for 100-concurrent-session write load

---

## 2. LLM Integration

### Primary: OpenAI API

**Decision**: OpenAI SDK (`openai` Python package) with streaming enabled

**Rationale**: Best-in-class streaming speed, stable async client (`AsyncOpenAI`), GPT-4o / GPT-4o-mini support for structured JSON output (claim extraction schema)

### Fallback: HuggingFace Inference Client

**Decision**: `huggingface_hub` Python client targeting self-hosted or serverless inference API

**Model selection**:

| Role | Model | Rationale |
|------|-------|-----------|
| LLM fallback (primary) | `mistralai/Mistral-7B-Instruct-v0.2` | Strong instruction following, widely benchmarked, fits 16 GB VRAM |
| LLM fallback (alt 1) | `meta-llama/Llama-3-8B-Instruct` | Best open-source 8B quality; requires Llama license agreement |
| LLM fallback (alt 2) | `Qwen2.5-7B-Instruct` | Excellent multilingual (Vietnamese), strong at structured output |

**Alternatives considered**:
- Anthropic Claude (original constitution entry): replaced with HuggingFace per constitution v1.0.1 — cost and data privacy concerns for self-hosted deployment

---

## 3. Speech-to-Text (STT)

**Decision**: `faster-whisper` (CTranslate2-optimized Whisper), self-hosted

**Rationale**:
- 4× speed improvement over original OpenAI Whisper at same accuracy
- Streaming word-level timestamps enable partial transcript delivery (FR-002 — incremental transcription)
- Free/OSS — no per-call API cost
- Vietnamese language support satisfies multilingual call center requirement
- Runs on GPU (CUDA) or CPU; configurable model size (base/medium/large-v3)

**Confidence handling**: faster-whisper returns per-segment `avg_logprob`; segments below configurable threshold trigger FR-002a (prompt caller to repeat)

**Alternatives considered**:
- Deepgram (original constitution): replaced per v1.0.1 — cost and external dependency
- OpenAI Whisper API: replaced — not streamable, latency too high for real-time target

---

## 4. Text-to-Speech (TTS)

**Decision**: Coqui TTS (primary) + edge-tts (fallback), both free/OSS

**Rationale**:
- Coqui TTS: VITS/XTTS models produce natural speech; supports Vietnamese; streaming chunk output
- edge-tts: Microsoft Edge TTS via unofficial free API; no cost, low latency, Vietnamese voices available
- Both abstracted behind `TTSPort` in Application — swap-transparent

**Alternatives considered**:
- ElevenLabs / Azure TTS (original constitution): replaced per v1.0.1 — cost and vendor lock-in

---

## 5. Vector Database & Embeddings

**Decision**: ChromaDB (local/embedded) + BAAI/bge-m3 embeddings

**Rationale**:
- ChromaDB: zero-cost, zero-ops for initial deployment; embedded mode eliminates a network hop in the retrieval path
- BAAI/bge-m3: multilingual (100+ languages including Vietnamese), top MTEB benchmark for semantic search, 768-dim vectors
- Can migrate to Qdrant/Weaviate later with only an infrastructure adapter swap (VectorRepository interface in Domain isolates this)

**Embedding model alternatives**:
- `intfloat/multilingual-e5-large`: solid multilingual alternative, slightly lower MTEB score than bge-m3

**Alternatives considered**:
- Pinecone / Weaviate cloud: rejected for initial deployment — cost and external dependency; VectorRepository interface keeps door open

---

## 6. Session Cache & Ephemeral State

**Decision**: Redis (aioredis) with the following key responsibilities:

| Responsibility | Redis Pattern |
|----------------|--------------|
| Session state (turn context, barge-in flag) | Hash per session ID, TTL = call max duration + grace |
| WebSocket presence | Set of active session IDs, atomic add/remove |
| Short-term conversation buffer | List per session ID (latest N turns for LLM context window) |
| Rate limiting | Sliding window counter per IP / session |
| Streaming coordination | Pub/Sub channel per session for barge-in signal |

**Rationale**: All-async via aioredis; atomic operations prevent race conditions in concurrent WS handlers; data is ephemeral by design (no durability needed — PostgreSQL holds durable state)

---

## 7. Asynchronous Background Workers

**Decision**: Python `asyncio` background tasks via FastAPI `BackgroundTasks` with optional Celery/ARQ upgrade path

**Rationale**:
- FR-013a and FR-017a require claim extraction and reminder generation to be enqueued on session close, non-blocking
- For 100 concurrent sessions, FastAPI `BackgroundTasks` is sufficient at launch
- ARQ (async Redis Queue) or Celery with Redis broker can be introduced without interface changes when volume demands it

**Worker isolation**: `ExtractClaimsUseCase` and `GenerateReminderUseCase` accept only `session_id: UUID` — idempotent, retriable, no shared mutable state

---

## 8. Clean Architecture Layer Boundaries

**Decision**: Strict four-layer Clean Architecture with explicit Protocol interfaces at each boundary

**Enforcement mechanism**: `import-linter` (CI gate) with forbidden-import rules:
- `domain` must not import from `application`, `interface`, or `infrastructure`
- `application` must not import from `interface` or `infrastructure`

**Dependency injection**: `infrastructure` wired at startup via FastAPI `Depends` + container (manual or `dependency-injector` library)

---

## 9. Observability Stack

**Decision**: structlog + OpenTelemetry (traces) + Prometheus (metrics)

| Concern | Tool |
|---------|------|
| Structured logs | structlog with JSON renderer in production |
| Distributed tracing | OpenTelemetry SDK + OTLP exporter (Jaeger/Zipkin compatible) |
| Metrics | prometheus-client (FastAPI middleware for HTTP; manual instruments for WS/LLM) |
| Correlation | `session_id` + `call_leg_id` injected into every log/span via structlog `contextvars` |

---

## 10. Security Design

**Decision**: Defense-in-depth across prompt injection, PII, and network surface

| Risk | Mitigation |
|------|-----------|
| Prompt injection via transcripts | `PromptSanitizer` in Domain (role separation, instruction-override pattern detection) |
| PII at rest (student_name, transcript) | PostgreSQL `pgcrypto` AES-256 column encryption on `Claim.student_name` and `Message.content` |
| WebSocket payload tampering | Pydantic strict models on every inbound frame type |
| LLM timeout / backpressure | asyncio timeout wrapper + circuit breaker on LLMPort; barge-in via Redis Pub/Sub |
| Memory growth in long calls | Short-term buffer capped to N turns in Redis; sliding window context truncation in StreamConversationUseCase |
| Secrets | All credentials via `ENVIRONMENT_VARIABLE` or Vault — detect-secrets pre-commit hook |

---

## 11. WebRTC / Audio Transport

**Decision**: WebSocket binary frames for audio data (MVP); WebRTC SFU stub for future scale

**Rationale**:
- WebSocket delivers sub-100 ms audio under LAN/low-latency conditions and is simpler to implement
- At 100 concurrent sessions WebSocket is sufficient; WebRTC SFU (Mediasoup/LiveKit) remains viable when voice load increases
- `webrtc_handler.py` stub in Infrastructure preserves extension point without blocking MVP

---

## 12. Resolved Unknowns Summary

| Unknown | Resolution |
|---------|-----------|
| Primary DB choice | PostgreSQL — relational integrity + JSONB + pgcrypto |
| LLM fallback model | Mistral-7B-Instruct-v0.2 via HuggingFace Inference Client |
| Embedding model | BAAI/bge-m3 (multilingual, Vietnamese-strong) |
| STT engine | faster-whisper (self-hosted, streaming, confidence scores) |
| TTS engine | Coqui TTS (primary) + edge-tts (fallback) |
| Background worker mechanism | FastAPI BackgroundTasks (ARQ upgrade path documented) |
| PII encryption at rest | pgcrypto AES-256 on Claim.student_name + Message.content |
| Barge-in coordination | Redis Pub/Sub per-session channel |
| CI import guard | import-linter with forbidden-import rules |
| WebRTC vs WebSocket | WebSocket MVP; WebRTC SFU stub for future |
