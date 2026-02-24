# Implementation Plan: AI Call Center Backend

**Branch**: ``001-ai-call-center-backend`` | **Date**: 2026-02-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from ``/specs/001-ai-call-center-backend/spec.md``

## Summary

Build a real-time AI-powered call center backend that accepts streaming audio over WebSocket, transcribes speech (faster-whisper), generates context-aware responses via RAG (ChromaDB + OpenAI primary / HuggingFace fallback), converts responses to speech (Coqui TTS primary / edge-tts fallback), and streams synthesized audio back to the caller  all within a 1.5 s first-token latency budget. Post-call, the system asynchronously extracts structured claims and reminders from transcripts. Durable state lives in PostgreSQL; ephemeral session state in Redis. Architecture follows Clean Architecture with strict inward dependency flow enforced by CI import guards.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: FastAPI, Starlette WebSockets, asyncio (uvicorn), OpenAI SDK, ``huggingface_hub`` (HuggingFace Inference Client), faster-whisper, Coqui TTS, edge-tts, ChromaDB, aioredis, asyncpg, SQLAlchemy (async), Pydantic v2, structlog, OpenTelemetry, prometheus-client, Locust
**Storage**: PostgreSQL (primary durable  CallSession, Message, Claim, Reminder, Document); Redis (ephemeral  session state, WS presence, short-term memory buffer, rate limiting, streaming coordination); ChromaDB (vector embeddings for RAG)
**Testing**: pytest-asyncio, pytest-mock, Locust (load); Testcontainers for Infrastructure integration tests
**Target Platform**: Linux server (containerized, Docker/Compose)
**Project Type**: web-service (WebSocket + REST API)
**Performance Goals**: First-token latency <1.5 s p95 (SC-001); sustain 100 concurrent sessions without latency degradation (SC-002)
**Constraints**: No blocking calls in async event loop; full end-to-end streaming (STTLLMTTSWS); PII encrypted at rest (DB-level or field-level); auth enforced at gateway layer (out of scope for this feature)
**Scale/Scope**: 100 concurrent call sessions; 5 REST endpoint groups + 1 WebSocket endpoint; single monorepo service

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Clean Architecture |  PASS | Four-layer structure (Domain / Application / Interface / Infrastructure). Repository pattern for all data access. CI import-guard enforced. Dependency arrows point inward only. |
| II. Real-Time System Design |  PASS | WebSocket primary channel (FR-001). Streaming pipeline STTLLMTTSWS. Barge-in supported (FR-005). All I/O uses async/await. |
| III. RAG Safety |  PASS | Prompt injection defense (FR-025). Relevance ranking (FR-011). Confidence scoring and source attribution included in RAG response payloads. ``VectorRepository`` interface defined in Domain. |
| IV. Reliability & Resilience |  PASS | LLM automatic fallback (FR-006, SC-005 2 s). Session cleanup on disconnect (FR-008). Circuit breakers on all external HTTP/gRPC clients. Graceful degradation when RAG unavailable. |
| V. Observability |  PASS | Structured JSON logs (FR-026). Prometheus metrics (FR-027). Correlation ID = session ID on every log entry and trace span. Domain error taxonomy defined. |
| VI. Security |  PASS | Pydantic validation on all WS payloads (FR-023). PII encrypted at rest (FR-028, FR-029). Prompt injection defenses (FR-025). Secrets via env vars / Vault. Auth at gateway (spec assumption). |
| VII. Performance |  PASS | SC-001 <1.5 s first-token p95. SC-002 100 concurrent sessions. Full streaming end-to-end. Load tests in CI for any change to the real-time call path. |

**Constitution Check: ALL GATES PASS  cleared for Phase 0.**

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-call-center-backend/
 plan.md              # This file (/speckit.plan command output)
 research.md          # Phase 0 output (/speckit.plan command)
 data-model.md        # Phase 1 output (/speckit.plan command)
 quickstart.md        # Phase 1 output (/speckit.plan command)
 contracts/           # Phase 1 output (/speckit.plan command)
    websocket.md
    rest-api.md
 tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
 domain/
    entities/
       call_session.py      # CallSession entity + session state lifecycle
       message.py           # Single conversation turn (caller | ai)
       claim.py             # Structured post-call claim record
       reminder.py          # Actionable follow-up record
       document.py          # Knowledge base source document
       embedding.py         # Vectorized document chunk
    value_objects/
       session_state.py     # Enum: active | ended | error
       speaker_role.py      # Enum: caller | ai
       confidence_score.py  # Typed float with threshold check
    repositories/            # Interfaces (Protocol / ABC)  no framework imports
       call_session_repository.py
       claim_repository.py
       reminder_repository.py
       document_repository.py
       vector_repository.py
    services/
       prompt_sanitizer.py  # Input sanitization / prompt injection guard
    errors.py                # TranscriptionError, LLMTimeoutError, RAGGroundingError, ...
 application/
    use_cases/
       handle_call.py           # HandleCallUseCase  main real-time call orchestrator
       stream_conversation.py   # StreamConversationUseCase  STT->RAG->LLM->TTS pipeline
       extract_claims.py        # ExtractClaimsUseCase  async background worker
       generate_reminder.py     # GenerateReminderUseCase  async background worker
       retrieve_knowledge.py    # RetrieveKnowledgeUseCase  RAG retrieval
    ports/                       # Abstractions consumed by use-cases (no infra imports)
        llm_port.py              # LLMPort protocol
        stt_port.py              # STTPort protocol
        tts_port.py              # TTSPort protocol
 interface/
    ws/
       call_controller.py       # WebSocket upgrade + audio frame dispatch
    rest/
       conversations.py         # GET /conversations/{session_id}/history
       claims.py                # GET /conversations/{session_id}/claims
       reminders.py             # GET /conversations/{session_id}/reminders
       documents.py             # POST /documents/ingest
       health.py                # GET /health
    dtos/
       ws_messages.py           # Pydantic models for all WS frame types
       rest_responses.py        # Pydantic response models for REST endpoints
    validators/
        audio_frame.py           # Audio codec / size validation
 infrastructure/
     llm/
        openai_client.py          # OpenAI SDK adapter (implements LLMPort)
        huggingface_client.py     # HuggingFace Inference Client adapter (LLMPort fallback)
                                     #   Model: mistralai/Mistral-7B-Instruct-v0.2 (primary)
                                     #   Alts:  meta-llama/Llama-3-8B-Instruct, Qwen2.5-7B-Instruct
     stt/
        faster_whisper_adapter.py # faster-whisper adapter (implements STTPort)
     tts/
        coqui_tts_adapter.py      # Coqui TTS adapter (implements TTSPort, primary)
        edge_tts_adapter.py       # edge-tts adapter (implements TTSPort, fallback)
     db/
        postgres/
           models.py                  # SQLAlchemy ORM models (PII fields encrypted)
           call_session_repo.py       # CallSessionRepository implementation
           claim_repo.py              # ClaimRepository implementation
           reminder_repo.py           # ReminderRepository implementation
           document_repo.py           # DocumentRepository implementation
        chroma/
            vector_repo.py             # VectorRepository implementation (ChromaDB)
                                          #   Embeddings: BAAI/bge-m3 (primary, multilingual/Vietnamese)
                                          #   Alt: intfloat/multilingual-e5-large
     cache/
        redis_client.py               # aioredis  session cache, WS presence, rate limiter
     signaling/
         webrtc_handler.py             # WebRTC SFU signaling (future-ready stub)

tests/
 unit/
    domain/                           # >=90% branch coverage required
    application/                      # >=90% branch coverage required
 integration/
    infrastructure/                   # Testcontainers: Postgres, Redis, ChromaDB
    interface/
 contract/
     ws/                               # WebSocket payload schema contract tests
```

**Structure Decision**: Single Clean Architecture monorepo service. Domain and Application layers have zero framework imports (enforced by CI import guard). Infrastructure wires concrete adapters via dependency injection. ``src/`` root chosen over ``app/`` to avoid FastAPI naming convention conflicts and cleanly separate Clean Architecture layers from ASGI framework concerns.
