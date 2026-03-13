# Vietnamese AI Call Center — Backend

A real-time voice AI system for Vietnamese airline ticket booking, powered by a **Task-Oriented Dialogue (TOD) pipeline**. Built with FastAPI, Clean Architecture, and designed for low-latency conversational flows.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [TOD Pipeline](#tod-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [WebSocket Protocol](#websocket-protocol)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Overview

The system handles end-to-end Vietnamese voice conversations for airline booking:

1. **Accepts** streaming audio over WebSocket
2. **Transcribes** Vietnamese speech via faster-whisper (STT)
3. **Understands** intent and extracts slots via NLU (JointIDSF + PhoBERT)
4. **Tracks** dialogue state across turns via Hybrid DST
5. **Decides** next action via Rule-based Policy engine
6. **Generates** natural Vietnamese responses via Template NLG
7. **Synthesizes** speech and streams audio back (TTS)
8. **Visualizes** pipeline state in real-time via WebSocket

### Key Characteristics

| Attribute | Value |
|---|---|
| Language | Vietnamese (air travel domain) |
| Framework | FastAPI + async Python 3.12+ |
| Pipeline | NLU → DST → Policy → NLG (4-stage TOD) |
| Concurrency | 100+ concurrent sessions via asyncio |
| First-token latency | < 1.5s (p95 target) |
| Dataset | PhoATIS — 28 intents, 82 slot types |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INTERFACE LAYER                           │
│  ┌──────────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  WebSocket Handler   │  │  REST API    │  │  Dialogue    │  │
│  │  (real-time voice)   │  │  (history)   │  │  REST API    │  │
│  └──────────────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                    APPLICATION LAYER                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  HandleCall → StreamConversation → TODPipeline             │  │
│  │              RetrieveKnowledge (FAQ fallback)              │  │
│  │              ExtractClaims → GenerateReminder              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Ports: NLUPort | DSTPort | PolicyPort | NLGPort                │
│         LLMPort | STTPort | TTSPort                             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                      DOMAIN LAYER                               │
│                                                                  │
│  Entities: DialogueState | CallSession | Message | Claim        │
│  Value Objects: NLUResult | SlotValue | PolicyDecision          │
│  Enums: PolicyAction (clarify|request_slot|confirm|execute|faq) │
│  Constants: BOOKING_SLOTS (10) | REQUIRED_SLOTS (3)            │
│  Services: PromptSanitizer                                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│                                                                  │
│  NLU:    PhoBERT adapter (keyword mock → ONNX JointIDSF)       │
│  DST:    HybridDSTAdapter (rule-based + confidence gating)      │
│  Policy: RulePolicyAdapter (decision tree)                      │
│  NLG:    TemplateNLGAdapter (~25 Vietnamese templates)          │
│                                                                  │
│  STT: faster-whisper  |  TTS: Coqui / edge-tts                 │
│  LLM: OpenAI / HuggingFace (FAQ fallback)                      │
│  DB:  PostgreSQL (asyncpg)  |  Cache: Redis                    │
│  RAG: ChromaDB + BAAI/bge-m3 (FAQ knowledge base)              │
└─────────────────────────────────────────────────────────────────┘
```

**Clean Architecture** enforces strict inward dependency flow:
- **Domain** — zero external dependencies; pure business logic
- **Application** — depends only on Domain; defines port contracts
- **Interface** — HTTP/WebSocket routing, DTOs, validation
- **Infrastructure** — concrete implementations of all ports

---

## TOD Pipeline

The core innovation: a **4-stage Task-Oriented Dialogue** pipeline replacing the previous RAG-only approach for booking flows.

### Pipeline Stages

```
User utterance: "Tôi muốn đặt vé đi Đà Nẵng ngày thứ sáu"
                              │
                    ┌─────────▼─────────┐
                    │   NLU (PhoBERT)   │
                    │                    │
                    │  Intent: atis_flight (0.92)
                    │  Slots:                  │
                    │    toloc.city_name: Đà Nẵng
                    │    depart_date.day_name: thứ sáu
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   DST (Hybrid)    │
                    │                    │
                    │  Accumulate slots  │
                    │  Confidence gate ≥ 0.5
                    │  Handle corrections│
                    │  Track turn history│
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Policy (Rules)   │
                    │                    │
                    │  Missing: fromloc  │
                    │  → request_slot   │
                    │  Target: fromloc.city_name
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  NLG (Templates)  │
                    │                    │
                    │  "Dạ, anh/chị muốn│
                    │   bay từ đâu ạ?"  │
                    └───────────────────┘
```

### Dialogue State (10 Booking Slots)

| Slot | Type | Required | Example |
|---|---|---|---|
| `fromloc.city_name` | string | Yes | Hà Nội |
| `toloc.city_name` | string | Yes | Đà Nẵng |
| `depart_date.day_name` | string | Yes | thứ sáu |
| `depart_date.month_name` | string | No | tháng 3 |
| `depart_date.day_number` | string | No | 15 |
| `depart_time.period_of_day` | string | No | sáng |
| `return_date.day_name` | string | No | chủ nhật |
| `airline_name` | string | No | Vietnam Airlines |
| `class_type` | string | No | thương gia |
| `round_trip` | string | No | khứ hồi |

### Policy Decision Tree

```
Intent is FAQ? ──────────────── → action: faq (route to RAG+LLM)
        │ no
Confidence < 0.5? ──────────── → action: clarify
        │ no
Missing required slots? ─────── → action: request_slot
        │ no
All slots filled? ───────────── → action: confirm
        │
User confirmed? ─────────────── → action: execute (generate booking ID)
User denied? ────────────────── → action: request_slot (re-ask)
```

### Vietnamese NLG Templates

The NLG adapter uses ~25 polite Vietnamese templates with register markers:

- **Greeting**: "Xin chào! Tôi là trợ lý đặt vé máy bay. Tôi có thể giúp gì cho anh/chị ạ?"
- **Request slot**: "Dạ, anh/chị muốn bay từ đâu ạ?" / "Anh/chị muốn bay ngày nào ạ?"
- **Confirm**: "Dạ, anh/chị xác nhận đặt vé từ {fromloc} đến {toloc} ngày {date} ạ?"
- **Execute**: "Dạ, đã đặt vé thành công! Mã đặt chỗ: {booking_id}"
- **Clarify**: "Dạ, xin lỗi, anh/chị có thể nói rõ hơn được không ạ?"

---

## Technology Stack

### Core

| Component | Technology | Purpose |
|---|---|---|
| Framework | FastAPI + Uvicorn | Async-first ASGI; WebSocket support |
| Runtime | Python 3.12+ / asyncio | Non-blocking I/O for concurrent sessions |
| Validation | Pydantic v2 | Type-safe DTOs; OpenAPI generation |

### Speech & Language

| Component | Primary | Fallback |
|---|---|---|
| STT | faster-whisper | — |
| TTS | Coqui TTS | edge-tts |
| LLM (FAQ) | OpenAI gpt-4-turbo | HuggingFace Mistral-7B |
| Embeddings | BAAI/bge-m3 | — |
| NLU | PhoBERT keyword adapter | ONNX JointIDSF (planned) |

### Data

| Layer | Technology | Purpose |
|---|---|---|
| Transactional | PostgreSQL 15+ (asyncpg) | Sessions, transcripts, claims |
| Cache | Redis 7+ | Ephemeral state, rate limits |
| Vector store | ChromaDB | RAG knowledge base |

### Observability

| Domain | Technology |
|---|---|
| Logging | structlog (JSON) |
| Tracing | OpenTelemetry + OTLP |
| Metrics | Prometheus |

---

## Project Structure

```
src/
├── main.py                          # App factory; startup/shutdown
│
├── domain/
│   ├── entities/
│   │   ├── dialogue_state.py        # DialogueState, NLUResult, SlotValue, PolicyDecision
│   │   ├── call_session.py          # CallSession entity
│   │   ├── claim.py                 # Claim entity
│   │   ├── message.py               # Message (transcript turn)
│   │   ├── document.py              # Knowledge base document
│   │   ├── embedding.py             # Vector chunk
│   │   └── reminder.py              # Reminder entity
│   ├── repositories/                # Abstract repository interfaces
│   ├── services/
│   │   └── prompt_sanitizer.py      # Prompt injection defense
│   ├── value_objects/               # ConfidenceScore, SessionState, SpeakerRole
│   └── errors.py                    # Domain-specific errors
│
├── application/
│   ├── ports/
│   │   ├── nlu_port.py              # NLUPort protocol
│   │   ├── dst_port.py              # DSTPort protocol
│   │   ├── policy_port.py           # PolicyPort protocol
│   │   ├── nlg_port.py              # NLGPort protocol
│   │   ├── llm_port.py              # LLMPort protocol
│   │   ├── stt_port.py              # STTPort protocol
│   │   └── tts_port.py              # TTSPort protocol
│   └── use_cases/
│       ├── tod_pipeline.py          # TODPipelineUseCase (NLU→DST→Policy→NLG)
│       ├── handle_call.py           # Call lifecycle orchestrator
│       ├── stream_conversation.py   # Real-time STT→TOD→TTS pipeline
│       ├── retrieve_knowledge.py    # RAG for FAQ intents
│       ├── extract_claims.py        # Post-call claim extraction
│       ├── generate_reminder.py     # Post-call reminder generation
│       └── ingest_document.py       # Knowledge base ingestion
│
├── infrastructure/
│   ├── nlu/
│   │   └── phobert_nlu_adapter.py   # Keyword-based NLU (→ ONNX JointIDSF)
│   ├── dst/
│   │   └── hybrid_dst_adapter.py    # Rule-based DST with confidence gating
│   ├── policy/
│   │   └── rule_policy_adapter.py   # Decision tree policy engine
│   ├── nlg/
│   │   └── template_nlg_adapter.py  # Vietnamese template NLG
│   ├── stt/
│   │   └── faster_whisper_adapter.py
│   ├── tts/
│   │   ├── coqui_tts_adapter.py
│   │   └── edge_tts_adapter.py
│   ├── llm/
│   │   ├── openai_client.py
│   │   └── huggingface_client.py
│   ├── db/
│   │   ├── postgres/                # SQLAlchemy models & repos
│   │   └── chroma/                  # Vector store repo
│   ├── cache/
│   │   └── redis_client.py
│   ├── observability/               # Metrics, circuit breaker
│   └── signaling/                   # WebRTC handler
│
└── interface/
    ├── rest/
    │   ├── sessions.py              # POST /api/v1/sessions
    │   ├── dialogue.py              # POST /api/v1/dialogue/turn
    │   ├── conversations.py         # GET /api/v1/conversations/{id}/history
    │   ├── claims.py                # GET /api/v1/conversations/{id}/claims
    │   ├── reminders.py             # GET /api/v1/conversations/{id}/reminders
    │   ├── documents.py             # POST /api/v1/documents/ingest
    │   └── health.py                # GET /api/v1/health
    ├── ws/
    │   └── call_controller.py       # WebSocket call handler
    ├── dtos/
    │   ├── dialogue_dtos.py         # TOD pipeline DTOs
    │   ├── ws_messages.py           # WebSocket frame models
    │   └── rest_responses.py        # REST response models
    ├── middleware/
    │   └── rate_limiter.py
    └── validators/
        └── audio_frame.py
```

---

## API Reference

### REST Endpoints

#### Dialogue Turn (TOD Pipeline)

```http
POST /api/v1/dialogue/turn
Content-Type: application/json

{
  "session_id": "uuid",
  "user_text": "Tôi muốn đặt vé đi Đà Nẵng"
}

Response:
{
  "session_id": "uuid",
  "response_text": "Dạ, anh/chị muốn bay từ đâu ạ?",
  "nlu": {
    "intent": "atis_flight",
    "confidence": 0.92,
    "slots": { "toloc.city_name": "Đà Nẵng" }
  },
  "dialogue_state": {
    "slots": { "toloc.city_name": "Đà Nẵng", ... },
    "turn_count": 1,
    "confirmed": false
  },
  "action": "request_slot",
  "target_slot": "fromloc.city_name"
}
```

#### Other Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/sessions` | Create session, get JWT + WS URL |
| GET | `/api/v1/conversations/{id}/history` | Conversation transcript |
| GET | `/api/v1/conversations/{id}/claims` | Extracted claims |
| GET | `/api/v1/conversations/{id}/reminders` | Generated reminders |
| POST | `/api/v1/documents/ingest` | Ingest knowledge base docs |
| GET | `/api/v1/health` | Service health check |

---

## WebSocket Protocol

### Connection

```
WS ws://host:8000/ws/calls/{session_id}
```

### Inbound Frames (Client → Server)

| Type | Purpose |
|---|---|
| `call_start` | Signal call begin |
| `audio.chunk` | PCM audio data (base64) |
| `audio.end` | End of utterance |
| `barge_in` | User interrupted AI |
| `call_end` | Signal call end |

### Outbound Frames (Server → Client)

| Type | Purpose |
|---|---|
| `transcript.partial` | Incremental STT text |
| `transcript.final` | Final transcript |
| `audio.frame` | TTS audio chunk |
| `audio.end` | End of TTS response |
| `call_status` | State transition (listening/thinking/speaking) |
| `pipeline_state` | Real-time TOD pipeline visualization data |
| `error` | Error with code and message |

### Pipeline State Frame

```json
{
  "type": "pipeline_state",
  "payload": {
    "nlu": {
      "intent": "atis_flight",
      "confidence": 0.92,
      "slots": { "toloc.city_name": "Đà Nẵng" }
    },
    "dialogue_state": {
      "slots": { ... },
      "turn_count": 2,
      "confirmed": false
    },
    "action": "request_slot",
    "target_slot": "fromloc.city_name",
    "response_text": "Dạ, anh/chị muốn bay từ đâu ạ?"
  }
}
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for PostgreSQL, Redis)
- API keys: OpenAI (or HuggingFace) for FAQ fallback

### Setup

```bash
cd CSCC-Back-End

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d             # PostgreSQL + Redis

# Run database migrations
alembic upgrade head

# Start the server
uvicorn src.main:create_app --reload --host 0.0.0.0 --port 8000
```

---

## Configuration

Environment variables (`.env`):

```bash
# TOD Pipeline (no API keys needed for core booking flow)
# The NLU/DST/Policy/NLG adapters are self-contained

# LLM (FAQ fallback only)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo
HUGGINGFACE_API_KEY=hf_...

# Speech
COQUI_MODEL=tts_models/vi/...
EDGE_TTS_FALLBACK=true

# RAG (FAQ knowledge base)
VECTOR_STORE_PATH=./data/chroma
EMBEDDING_MODEL=BAAI/bge-m3

# Database
POSTGRES_URL=postgresql+asyncpg://user:password@localhost/cscc
REDIS_URL=redis://localhost:6379/0

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Testing

```bash
# Unit tests
pytest tests/unit/ -v --cov=src --cov-fail-under=90

# Integration tests (requires Docker containers)
pytest tests/integration/ -v

# Load test
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10
```

### Architecture Enforcement

Import guards via `import-linter` ensure Clean Architecture boundaries:

- Domain cannot import from Application/Infrastructure/Interface
- Application cannot import from Infrastructure/Interface
- Interface cannot import from Infrastructure directly

---

## Deployment

```bash
# Build Docker image
docker build -t cscc-backend:latest .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### Latency Budget (Target: < 1.5s p95)

| Stage | Budget | Typical |
|---|---|---|
| Network + buffering | 100ms | 50ms |
| STT (faster-whisper) | 300ms | 100–200ms |
| TOD Pipeline (NLU→DST→Policy→NLG) | 50ms | 10–30ms |
| TTS (Coqui) | 400ms | 300–400ms |
| Margin | 150ms | — |

The TOD pipeline is significantly faster than the RAG+LLM path since it uses local rule-based processing instead of external API calls.

---

## Roadmap

- [ ] Train JointIDSF + PhoBERT on PhoATIS dataset → replace keyword NLU adapter
- [ ] Export ONNX model for production inference
- [ ] Add Streamlit demo UI (Phase 5)
- [ ] Evaluation metrics: intent accuracy, slot F1, task completion rate
- [ ] Vietnamese TTS voice fine-tuning

---

**Status**: In Development
**Last Updated**: 2026-03-13
