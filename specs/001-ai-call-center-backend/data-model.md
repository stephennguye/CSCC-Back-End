# Data Model: AI Call Center Backend

**Phase**: 1 — Design & Contracts
**Branch**: `001-ai-call-center-backend`
**Date**: 2026-02-24

---

## Entity Overview

```text
CallSession (1) ──< Message (many)
CallSession (1) ──< Claim (0..1)
CallSession (1) ──< Reminder (many)
Document (1) ──< Embedding (many)
```

---

## Entities

### CallSession

The single canonical entity for an active or completed call. "Conversation ID" is a synonym for session ID across all REST endpoints.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `id` | UUID | PK, not null | Canonical session / conversation ID |
| `state` | `SessionState` (enum) | not null, default `active` | `active` \| `ended` \| `error` |
| `created_at` | timestamp with tz | not null, default now() | Call start time |
| `ended_at` | timestamp with tz | nullable | Set on session close |
| `metadata` | JSONB | nullable | Caller info, codec, routing metadata |

**Ephemeral fields (Redis only — not persisted to Postgres)**:

| Field | Redis type | TTL | Notes |
|-------|-----------|-----|-------|
| `short_term_buffer` | List | call max + grace | Last N turns for LLM context window |
| `ws_presence` | Set membership | call max + grace | Active WebSocket connection flag |
| `barge_in_channel` | Pub/Sub | per-call | Signals in-progress LLM/TTS stream to halt |
| `turn_state` | Hash | call max + grace | `generating` \| `idle` — tracks AI response in flight |

**State transitions**:

```text
(created) → active → ended
                 └──→ error
```

**Validation rules**:
- `ended_at` must be null when `state = active`
- `ended_at` must be set when `state = ended` or `error`

---

### Message

A single conversation turn. Belongs to a `CallSession`.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `id` | UUID | PK, not null | |
| `session_id` | UUID | FK → CallSession.id, not null | |
| `role` | `SpeakerRole` (enum) | not null | `user` \| `ai` |
| `content` | text | not null, **encrypted at rest** | Transcript text — PII (FR-029) |
| `confidence_score` | float | nullable, 0.0–1.0 | ASR confidence; null for AI turns |
| `timestamp` | timestamp with tz | not null | UTC wall-clock time of turn start |
| `sequence_number` | integer | not null | Monotonically increasing within session |

**Validation rules**:
- `content` must not be empty
- `confidence_score` must be in [0.0, 1.0] when present
- `role = user` should have `confidence_score`; `role = ai` should not

**Encryption**: `content` column encrypted via pgcrypto AES-256 using server-managed key. Application layer receives decrypted plaintext via ORM transparently.

---

### Claim

Structured post-call record extracted asynchronously from a completed `CallSession`.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `id` | UUID | PK, not null | |
| `session_id` | UUID | FK → CallSession.id, unique, not null | One claim per session |
| `student_name` | text | nullable, **encrypted at rest** | PII — FR-028; null if not determinable |
| `issue_category` | text | nullable | e.g., `enrollment`, `financial_aid`, `transcript` |
| `urgency_level` | `UrgencyLevel` (enum) | nullable | `low` \| `medium` \| `high` \| `critical` |
| `confidence` | float | nullable, 0.0–1.0 | Extraction confidence score produced by the claim extraction LLM; null if unavailable |
| `requested_action` | text | nullable | Free text describing caller's requested action |
| `follow_up_date` | date | nullable | Parsed target follow-up date; null if not determinable |
| `extracted_at` | timestamp with tz | not null | When background worker completed extraction |
| `schema_version` | text | not null, default `v1` | For determinism guarantee (FR-014) |

**Validation rules**:
- Unresolvable fields MUST be stored as `null` — never guessed (FR-013 note)
- `schema_version` must match the active extraction schema version on write
- `session_id` is unique — at most one `Claim` per `CallSession`

**Encryption**: `student_name` encrypted via pgcrypto AES-256 (FR-028).

---

### Reminder

An actionable follow-up item derived asynchronously from a completed `CallSession`.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `id` | UUID | PK, not null | |
| `session_id` | UUID | FK → CallSession.id, not null | |
| `description` | text | not null | Natural language description of the follow-up item |
| `target_due_at` | timestamp with tz | nullable | Parsed follow-up date/time; null if not determinable |
| `created_at` | timestamp with tz | not null, default now() | |

**Validation rules**:
- `description` must not be empty
- Multiple reminders per session are permitted
- `target_due_at` is null when no parseable date or time was detected (FR-018)

---

### Document

A knowledge base source record ingested via the document ingestion API.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `id` | UUID | PK, not null | |
| `title` | text | nullable | Human-readable document title |
| `source` | text | nullable | Origin URL or file path |
| `content` | text | not null | Full raw extracted text |
| `ingested_at` | timestamp with tz | not null, default now() | |
| `metadata` | JSONB | nullable | Format, version, author, etc. |

---

### Embedding

A vectorized representation of a `Document` chunk for RAG similarity search. Stored in ChromaDB (not PostgreSQL).

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | ChromaDB document ID |
| `document_id` | UUID | Reference to source `Document.id` |
| `chunk_index` | integer | Position of chunk within source document |
| `chunk_text` | text | The raw text of this chunk |
| `vector` | float[] (768-dim) | BAAI/bge-m3 embedding vector |
| `metadata` | dict | document_id, chunk_index, source, title |

---

## Value Objects

### SessionState

```python
class SessionState(str, Enum):
    active = "active"
    ended = "ended"
    error = "error"
```

### SpeakerRole

```python
class SpeakerRole(str, Enum):
    user = "user"
    ai = "ai"
```

### UrgencyLevel

```python
class UrgencyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"
```

### ConfidenceScore

```python
@dataclass(frozen=True)
class ConfidenceScore:
    value: float  # 0.0 – 1.0

    def is_below_threshold(self, threshold: float) -> bool:
        return self.value < threshold
```

---

## Domain Errors

Defined in `src/domain/errors.py`:

| Error Class | Trigger Condition |
|-------------|------------------|
| `TranscriptionError` | faster-whisper fails or returns invalid output |
| `LLMTimeoutError` | OpenAI / HuggingFace call exceeds configured timeout |
| `LLMFallbackExhaustedError` | Both primary and fallback LLM unavailable |
| `RAGGroundingError` | Retrieval returns no usable context above confidence threshold |
| `PromptInjectionDetectedError` | Sanitizer detects instruction-override pattern in input |
| `SessionNotFoundError` | Session ID does not exist in Postgres |
| `SessionAlreadyEndedError` | Operation attempted on a session in `ended` or `error` state |
| `ClaimExtractionError` | Background worker fails to produce a valid claim schema |

---

## Database Schema Notes

### Migrations

Alembic manages PostgreSQL schema migrations. Migration files live in `alembic/versions/`.

### Indexes

| Table | Index | Purpose |
|-------|-------|---------|
| `message` | `(session_id, sequence_number)` | Ordered transcript retrieval |
| `claim` | `session_id` (unique) | Claim lookup by session |
| `reminder` | `session_id` | Reminders by session |
| `reminder` | `target_due_at` | Future: date/time-based reminder scheduling |

### Encryption Key Management

- Server-managed key stored in environment variable `PGCRYPTO_KEY` (or Vault path)
- Key rotation requires re-encrypting affected columns; migration script provided separately

---

## Post-Design Constitution Check (Re-evaluation)

| Principle | Status | Design decision |
|-----------|--------|-----------------|
| I. Clean Architecture | ✅ PASS | Entities are pure Python dataclasses — no ORM or FastAPI imports in Domain |
| III. RAG Safety | ✅ PASS | `ConfidenceScore` value object enables threshold checks; `Embedding` includes `document_id` for source attribution |
| VI. Security | ✅ PASS | `Message.content` and `Claim.student_name` encrypted at rest; encryption handled in Infrastructure ORM layer, transparent to Domain |
