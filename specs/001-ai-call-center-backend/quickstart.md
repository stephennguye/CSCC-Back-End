# Quickstart: AI Call Center Backend

**Branch**: `001-ai-call-center-backend`
**Date**: 2026-02-24

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Python | 3.12 | `python --version` |
| Docker | 24.x | For PostgreSQL, Redis, ChromaDB containers |
| Docker Compose | 2.x | `docker compose version` |
| CUDA (optional) | 11.8+ | Required for GPU-accelerated faster-whisper and Coqui TTS |

---

## 1. Clone and Set Up Environment

```bash
git clone <repo-url>
cd CSCC-Back-End
git checkout 001-ai-call-center-backend

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Configure Environment Variables

Copy the example file and fill in required values:

```bash
cp .env.example .env
```

**Required variables** (`.env`):

```dotenv
# Database
DATABASE_URL=postgresql+asyncpg://cscc:cscc_pass@localhost:5432/cscc_db
PGCRYPTO_KEY=<32-byte-hex-key>          # AES-256 key for PII field encryption

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM — Primary (OpenAI)
OPENAI_API_KEY=sk-...

# LLM — Fallback (HuggingFace)
HUGGINGFACE_API_TOKEN=hf_...
HUGGINGFACE_FALLBACK_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# STT
FASTER_WHISPER_MODEL=large-v3           # Options: tiny, base, medium, large-v3
ASR_CONFIDENCE_THRESHOLD=0.70

# TTS
TTS_PRIMARY=coqui                       # Options: coqui, edge_tts
TTS_COQUI_MODEL=tts_models/en/ljspeech/vits

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3

# ChromaDB
CHROMA_DB_PATH=./data/chroma

# Observability
LOG_LEVEL=INFO
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
PROMETHEUS_PORT=9090
```

**Optional** (debug only — MUST NOT be enabled in production):

```dotenv
LLM_DEBUG_LOG_PROMPTS=false             # Logs raw prompts — opt-in debug only
```

---

## 3. Start Infrastructure Services

```bash
docker compose up -d postgres redis
```

Postgres starts on `localhost:5432`, Redis on `localhost:6379`.

**First-time database setup**:

```bash
alembic upgrade head
```

---

## 4. Run the Development Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`.

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

---

## 5. Run Tests

```bash
# All unit tests (Domain + Application — no external services needed)
pytest tests/unit/ -v

# Integration tests (requires Docker — starts Testcontainers automatically)
pytest tests/integration/ -v

# Contract tests (WebSocket payload schemas)
pytest tests/contract/ -v

# Full suite
pytest -v
```

Coverage check:

```bash
pytest tests/unit/ --cov=src/domain --cov=src/application --cov-report=term-missing
# Must meet >=90% branch coverage on Domain and Application layers
```

---

## 6. Ingest a Knowledge Base Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Enrollment Policy Spring 2026",
    "content": "Full text of the document..."
  }'
```

---

## 7. Simulate a WebSocket Call (Manual Test)

Using `websocat` or any WebSocket client:

```bash
websocat ws://localhost:8000/ws/calls/550e8400-e29b-41d4-a716-446655440000
```

Send an audio chunk frame:

```json
{"type":"audio.chunk","session_id":"550e8400-e29b-41d4-a716-446655440000","payload":{"sequence":1,"codec":"pcm_16khz_mono","data":"<base64-audio>"}}
```

End the session:

```json
{"type":"session.end","session_id":"550e8400-e29b-41d4-a716-446655440000","payload":{}}
```

Retrieve history after the session ends:

```bash
curl http://localhost:8000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/history
```

---

## 8. Project Structure Reference

```text
src/
├── domain/          # Pure business logic — no framework imports
├── application/     # Use-cases + port interfaces
├── interface/       # FastAPI controllers, DTOs, validators
└── infrastructure/  # Concrete adapters (DB, LLM, STT, TTS, Redis, ChromaDB)

tests/
├── unit/            # Domain + Application (≥90% branch coverage)
├── integration/     # Infrastructure + Interface (Testcontainers)
└── contract/        # WS payload schema tests
```

See [data-model.md](data-model.md) for entity definitions, [contracts/websocket.md](contracts/websocket.md) for WebSocket frame schemas, and [contracts/rest-api.md](contracts/rest-api.md) for REST endpoint contracts.

---

## 9. Architecture Overview

```text
Caller Audio
    │
    ▼ WebSocket binary/JSON frames
┌──────────────────────────────────────────────────────┐
│  Interface Layer                                      │
│  CallController (ws/call_controller.py)               │
└────────────────────────┬─────────────────────────────┘
                         │ use-case calls
┌────────────────────────▼─────────────────────────────┐
│  Application Layer                                    │
│  HandleCallUseCase → StreamConversationUseCase        │
│      ├── STTPort → faster-whisper                     │
│      ├── RetrieveKnowledgeUseCase → VectorRepository  │
│      ├── LLMPort → OpenAI (primary) / HuggingFace     │
│      └── TTSPort → Coqui TTS (primary) / edge-tts     │
│                                                       │
│  On session close (async background):                 │
│      ├── ExtractClaimsUseCase                         │
│      └── GenerateReminderUseCase                      │
└────────────────────────┬─────────────────────────────┘
                         │ repository / adapter calls
┌────────────────────────▼─────────────────────────────┐
│  Infrastructure Layer                                 │
│  PostgreSQL │ Redis │ ChromaDB │ LLM SDKs │ STT │ TTS│
└──────────────────────────────────────────────────────┘
```

---

## 10. Key Architecture Constraints (Reminder)

| Constraint | Enforcement |
|------------|-------------|
| No blocking calls in async loop | `flake8-async` in CI |
| No cross-layer imports | `import-linter` in CI |
| PII encrypted at rest | pgcrypto on `Message.content`, `Claim.student_name` |
| ≥90% branch coverage (Domain + Application) | `pytest-cov` CI gate |
| First-token latency <1.5 s p95 | Locust load test CI gate |
| All WS payloads validated via Pydantic | Contract tests |
