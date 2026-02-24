<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.0.1 (PATCH — technology stack refinements)

Modified principles: None

Modified sections:
  - Technology Stack: LLM fallback changed Anthropic → HuggingFace Inference Client
  - Technology Stack: STT changed Deepgram → faster-whisper (self-hosted, free)
  - Technology Stack: TTS changed ElevenLabs/Azure → Coqui TTS + edge-tts (free)
  - Technology Stack: Vector DB narrowed to ChromaDB (local, free)

Added sections: None
Removed sections: None

Templates reviewed:
  - .specify/templates/plan-template.md        ✅ aligned (Constitution Check gate present)
  - .specify/templates/spec-template.md        ✅ aligned (FR/acceptance criteria structure compatible)
  - .specify/templates/tasks-template.md       ✅ aligned (phase/story task structure compatible)
  - .specify/templates/constitution-template.md ✅ no changes required

Follow-up TODOs:
  - None. All placeholders resolved.
-->

# CSCC Backend Constitution

## Core Principles

### I. Clean Architecture Enforcement

The codebase MUST be organized into four strictly separated layers:

- **Domain** — pure business logic and entities; zero imports from FastAPI,
  Redis, or any external SDK or framework.
- **Application** — use-cases and orchestration; depends only on Domain
  abstractions (interfaces/protocols).
- **Interface** — controllers, DTOs, WebSocket handlers, gateway adapters;
  depends on Application and Domain.
- **Infrastructure** — concrete implementations: database clients, Redis
  adapters, LLM SDKs, STT/TTS integrations; depends on Interface/Application
  abstractions only via dependency injection.

Dependency arrows MUST always point inward (toward Domain). Any cross-layer
import violation MUST be caught by automated lint/import-guard rules in CI.
The repository pattern MUST be used for all data access; no layer above
Infrastructure may hold a direct DB or cache client reference.

**Rationale**: Enforces testability, prevents framework lock-in, and isolates
blast radius when external dependencies change.

### II. Real-Time System Design

The system MUST support low-latency, bidirectional communication:

- WebSocket is the primary real-time channel; REST endpoints are permitted
  only for non-streaming control-plane operations (auth, config, health).
- Partial/incremental streaming responses MUST be emitted as soon as the first
  token is available — never buffered to completion before sending.
- Barge-in (interruption) MUST be supported: the server MUST be able to
  immediately halt an in-progress LLM/TTS stream when a new audio event
  arrives from the client.
- All async boundaries (network I/O, LLM calls, vector search) MUST use
  `async`/`await`; no blocking calls are permitted in the async event loop.

**Rationale**: Call-center UX requires near-human conversational cadence;
buffering or blocking degrades perceived quality below acceptable thresholds.

### III. RAG Safety

Every retrieval-augmented generation (RAG) operation MUST satisfy:

- **Grounding**: LLM responses MUST be traceable to retrieved source documents.
- **Source attribution**: Each response payload MUST include document
  identifiers and chunk references used in context construction.
- **Confidence scoring**: A numeric confidence score MUST accompany every RAG
  result; callers MUST surface or log scores below the configured threshold.
- **Prompt-injection defenses**: All user-supplied text (audio transcripts,
  text messages) MUST be sanitized and framed with adversarial-input guards
  before insertion into prompt templates.
- **Repository abstraction**: Vector search MUST be accessed exclusively
  through a `VectorRepository` interface defined in Domain; Infrastructure
  holds the concrete implementation.

**Rationale**: Call-center agents rely on accurate, verifiable information.
Hallucinations or injected instructions in a live call carry direct business
and legal risk.

### IV. Reliability & Resilience

- All I/O operations (LLM, STT, TTS, DB, Redis, external APIs) MUST be
  `async` and wrapped with timeout and retry logic.
- LLM provider failures MUST trigger automatic fallback to a secondary
  provider defined in configuration; the call MUST not hard-fail on a single
  provider outage.
- Redis MUST be used for ephemeral session state (WIP transcripts, turn
  context, barge-in signals); durable state persists to the primary database.
- Graceful degradation is REQUIRED: when a non-critical subsystem (e.g., RAG,
  sentiment analysis) is unavailable, the core call flow MUST continue with
  reduced capability rather than returning an error to the caller.
- Circuit breakers MUST be applied to all third-party HTTP/gRPC clients.

**Rationale**: Production call centers operate 24/7; any unhandled cascade
failure directly impacts active customer calls.

### V. Observability

- Structured JSON logging is MANDATORY for every service boundary crossing
  (request in, response out, error, external call).
- A correlation ID (session ID + call leg ID) MUST be injected into every log
  entry, trace span, and error response for end-to-end traceability.
- All internal services MUST expose Prometheus-compatible metrics endpoints;
  at minimum: request count, latency histograms (p50/p95/p99), and error
  rates per integration.
- A clear error taxonomy MUST be maintained in Domain (e.g.,
  `TranscriptionError`, `LLMTimeoutError`, `RAGGroundingError`) so that
  monitoring dashboards and alerts can be built against stable error codes.
- Debug-level verbose logging of raw prompts/responses MUST be
  opt-in via environment variable and MUST NOT be enabled in production by
  default.

**Rationale**: Diagnosing real-time issues in a live call without structured
observability is operationally impossible at scale.

### VI. Security

- All WebSocket message payloads MUST be validated against strict schemas
  (Pydantic models or equivalent) before processing; malformed messages MUST
  be rejected with a structured error, not silently dropped.
- Client-supplied audio and text MUST be treated as untrusted input at all
  times; transcription output MUST be re-validated before use in prompts.
- Prompt injection MUST be defended against via input sanitization, role
  separation in prompt templates, and output scanning for instruction-override
  patterns.
- All secrets (API keys, DB credentials, JWT signing keys) MUST be sourced
  from environment variables or a secrets manager; hardcoded credentials are
  strictly forbidden and MUST be caught by pre-commit secret-scanning hooks.
- Authentication/authorization MUST be enforced on every WebSocket upgrade
  handshake; unauthenticated connections MUST be rejected before any data
  processing begins.

**Rationale**: Call-center systems handle PII and sensitive conversation data;
a single injection or credential leak can have regulatory and reputational
consequences.

### VII. Performance

- First-token latency (transcript receipt → first TTS audio chunk delivered
  to client) MUST target <1.5 s under p95 load.
- No synchronous blocking operations (file I/O, CPU-intensive parsing,
  synchronous HTTP) are permitted in the async event loop; offload to
  thread/process pools where unavoidable.
- Streaming MUST be used end-to-end: STT → LLM (token streaming) → TTS
  (chunk streaming) → WebSocket (binary frames); no stage may buffer a full
  result before forwarding downstream.
- Load tests validating the <1.5 s first-token SLA MUST be part of the CI
  pipeline for any change touching the real-time call path.

**Rationale**: Latency above ~2 s is perceptible as "unnatural" pause in a
phone conversation, directly harming customer experience and CSAT scores.

## Technology Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.12+ |
| Web / WebSocket | FastAPI + Starlette WebSockets |
| Async runtime | asyncio (uvicorn) |
| LLM integration | OpenAI SDK (primary); HuggingFace Inference Client (fallback) |
| STT | faster-whisper — self-hosted Whisper (free/OSS, abstracted via `STTRepository`) |
| TTS | Coqui TTS (primary, free/OSS); edge-tts (secondary, free, abstracted via `TTSRepository`) |
| Vector DB | ChromaDB (local/embedded, free/OSS, abstracted via `VectorRepository`) |
| Cache / ephemeral state | Redis (aioredis) |
| Primary DB | PostgreSQL (asyncpg / SQLAlchemy async) |
| Observability | structlog + OpenTelemetry + Prometheus |
| Secrets | Environment variables / HashiCorp Vault |
| Testing | pytest-asyncio, pytest-mock, Locust (load) |

Technology choices MUST be recorded in feature plans; alternatives MUST be
justified against this stack before introduction.

## Quality Gates & Development Workflow

All pull requests targeting `main` or `develop` MUST pass the following gates
before merge:

1. **Import-guard / architecture lint** — no cross-layer dependency violations.
2. **Unit tests** — Domain and Application layers MUST have ≥90% branch
   coverage; Infrastructure integration tests MUST be run with Testcontainers.
3. **Async safety** — no `time.sleep`, `requests.get`, or other blocking calls
   in async code paths (enforced by `flake8-async` or equivalent).
4. **Secret scanning** — `detect-secrets` or `gitleaks` pre-commit hook passes.
5. **Schema validation** — all WebSocket payload models validated via Pydantic.
6. **Load test gate** — first-token latency SLA (<1.5 s p95) validated for
   any change to the real-time call path.
7. **Structured log review** — new log statements MUST use structured fields,
   not f-string interpolation of sensitive data.

Feature branches MUST reference a spec (`specs/###-feature-name/spec.md`) and
a plan (`plan.md`) before implementation begins. The Constitution Check in
`plan-template.md` MUST be completed and pass all seven principles.

## Governance

This constitution supersedes all individual feature decisions, PR comments, and
verbal agreements. Amendments require:

1. A pull request modifying `.specify/memory/constitution.md`.
2. The Sync Impact Report (HTML comment at top of file) updated with version
   change, modified/added/removed sections, and template propagation status.
3. Version bump following semantic rules:
   - **MAJOR** — removal or redefinition of an existing principle, or removal
     of a mandatory Quality Gate.
   - **MINOR** — addition of a new principle or section, or materially expanded
     guidance to an existing principle.
   - **PATCH** — wording clarifications, typo fixes, non-semantic refinements.
4. All dependent templates (`.specify/templates/`) MUST be reviewed and updated
   in the same PR when a MINOR or MAJOR bump occurs.
5. A compliance review checkpoint MUST be added to the milestone board within
   30 days of any MAJOR amendment.

Use `.specify/templates/` for session-specific runtime guidance. Refer to
`plan-template.md` for the Constitution Check gate at the start of every
feature plan.

**Version**: 1.0.1 | **Ratified**: 2026-02-24 | **Last Amended**: 2026-02-24
