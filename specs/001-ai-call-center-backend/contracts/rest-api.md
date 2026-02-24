# REST API Contract: AI Call Center Backend

**Phase**: 1 — Design & Contracts
**Branch**: `001-ai-call-center-backend`
**Date**: 2026-02-24

---

## Base URL

`http://{host}/api/v1`

---

## Common Conventions

- All request/response bodies are `application/json`
- All timestamps are ISO 8601 UTC (`2026-02-24T10:30:00Z`)
- All IDs are UUIDs (`550e8400-e29b-41d4-a716-446655440000`)
- Rate limiting: sliding window per IP; 429 returned with `Retry-After` header when exceeded (FR-024)
- Malformed requests return 400 with structured error body (FR-023)

### Standard Error Response

```json
{
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "No session found with the provided ID.",
    "request_id": "req-uuid"
  }
}
```

---

## Endpoints

---

### POST `/sessions`

Create a new call session and obtain a short-lived session token. The client MUST call this endpoint before opening a WebSocket connection; the returned `session_id` and `token` are required for the WebSocket upgrade (frontend FR-027).

**Request body**: optional

```json
{ "session_id": "550e8400-e29b-41d4-a716-446655440000" }
```

- **Omit `session_id`** (or send `{}`) to create a **new** `CallSession`.
- **Provide `session_id`** to issue a **new token for an existing active session** (used during WebSocket reconnection — see WebSocket Reconnection Protocol). No new `CallSession` record is created.

**Response 201 Created**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 300,
  "ws_url": "ws://{host}/ws/calls/550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | UUID | Canonical ID for this call session; used in all subsequent REST and WebSocket calls |
| `token` | string | Short-lived JWT (TTL: `expires_in` seconds); passed as `Authorization: Bearer <token>` on the WebSocket upgrade request |
| `expires_in` | integer | Token lifetime in seconds (default: 300). Client must initiate the WebSocket before expiry. |
| `ws_url` | string | Convenience field; the pre-formed WebSocket URL for this session |

**Behaviour**:
- Creates a `CallSession` record in `active` state in PostgreSQL.
- Issues a signed JWT scoped to this `session_id`; token is validated by the gateway on WebSocket upgrade.
- A WebSocket upgrade attempt with an expired or absent token is rejected with HTTP 401.

**Error responses**:

| Status | Code | Condition |
|--------|------|-----------|
| 429 | `RATE_LIMIT_EXCEEDED` | Too many session creation requests |
| 500 | `SESSION_CREATE_FAILED` | Internal error creating session record |

---

### GET `/conversations/{session_id}/history`

Retrieve the ordered transcript for a completed or active session.

**Path parameters**:

| Parameter | Type | Notes |
|-----------|------|-------|
| `session_id` | UUID | Canonical session / conversation ID |

**Query parameters**:

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `limit` | integer | 100 | Max messages to return |
| `offset` | integer | 0 | Pagination offset |

**Response 200**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "ended",
  "created_at": "2026-02-24T10:00:00Z",
  "ended_at": "2026-02-24T10:12:45Z",
  "messages": [
    {
      "id": "msg-uuid-1",
      "role": "user",
      "content": "I need help with my enrollment status.",
      "confidence_score": 0.94,
      "timestamp": "2026-02-24T10:00:15Z",
      "sequence_number": 1
    },
    {
      "id": "msg-uuid-2",
      "role": "ai",
      "content": "I can help with that. Your enrollment status is...",
      "confidence_score": null,
      "timestamp": "2026-02-24T10:00:16Z",
      "sequence_number": 2
    }
  ],
  "total": 14,
  "limit": 100,
  "offset": 0
}
```

**Error responses**:

| Status | Code | Condition |
|--------|------|-----------|
| 404 | `SESSION_NOT_FOUND` | Session ID does not exist |
| 400 | `INVALID_UUID` | Malformed session_id |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |

---

### GET `/conversations/{session_id}/claims`

Retrieve the structured claim extracted from a session.

**Path parameters**:

| Parameter | Type | Notes |
|-----------|------|-------|
| `session_id` | UUID | |

**Response 200**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim": {
    "id": "claim-uuid",
    "student_name": "Nguyen Van A",
    "issue_category": "enrollment",
    "urgency_level": "high",
    "confidence": 0.91,
    "requested_action": "Update enrollment status for Spring 2026",
    "follow_up_date": "2026-03-01",
    "extracted_at": "2026-02-24T10:13:05Z",
    "schema_version": "v1"
  }
}
```

**Notes**:
- Fields that could not be determined from the transcript are returned as `null`
- `confidence` is the extraction confidence score (0.0–1.0); `null` if the extraction model did not emit a score
- `student_name` is decrypted by the application layer before inclusion in the response; it MUST NOT appear as ciphertext

**Response 200 — claim not yet ready**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim": null,
  "claim_status": "pending"
}
```

**Response 200 — session ended, no claim extractable**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim": null,
  "claim_status": "not_extractable"
}
```

**Error responses**:

| Status | Code | Condition |
|--------|------|-----------|
| 404 | `SESSION_NOT_FOUND` | Session ID does not exist |
| 400 | `INVALID_UUID` | Malformed session_id |
| 429 | `RATE_LIMIT_EXCEEDED` | |

---

### GET `/conversations/{session_id}/reminders`

Retrieve all reminders generated from a session.

**Path parameters**:

| Parameter | Type | Notes |
|-----------|------|-------|
| `session_id` | UUID | |

**Response 200**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reminders": [
    {
      "id": "reminder-uuid-1",
      "description": "Follow up with student Nguyen Van A regarding enrollment confirmation",
      "target_due_at": "2026-03-01T09:00:00Z",
      "created_at": "2026-02-24T10:13:10Z"
    },
    {
      "id": "reminder-uuid-2",
      "description": "Send updated enrollment policy document to student",
      "target_due_at": null,
      "created_at": "2026-02-24T10:13:10Z"
    }
  ],
  "total": 2,
  "reminders_status": "complete"
}
```

**`reminders_status` values**:

| Value | Meaning |
|-------|---------|
| `pending` | Background worker not yet complete |
| `complete` | Worker finished; array may be empty if no reminders detected |

**Error responses**:

| Status | Code | Condition |
|--------|------|-----------|
| 404 | `SESSION_NOT_FOUND` | Session ID does not exist |
| 400 | `INVALID_UUID` | Malformed session_id |
| 429 | `RATE_LIMIT_EXCEEDED` | |

---

### POST `/documents/ingest`

Ingest a document into the knowledge base (FR-010, FR-012).

**Request body**:

```json
{
  "title": "Enrollment Policy Spring 2026",
  "source": "https://institution.edu/policies/enrollment-2026.pdf",
  "content": "Full text content of the document...",
  "metadata": {
    "format": "pdf",
    "version": "2026-01",
    "author": "Registrar Office"
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `title` | string | no | Human-readable title |
| `source` | string | no | Origin URL or identifier |
| `content` | string | yes | Raw extracted text; max 2 MB |
| `metadata` | object | no | Arbitrary key-value metadata |

**Response 202 Accepted**:

```json
{
  "document_id": "doc-uuid",
  "status": "ingesting",
  "message": "Document accepted for processing. Embeddings will be generated asynchronously."
}
```

**Notes**:
- Chunking and embedding generation are asynchronous background operations
- Document is queryable via RAG once embeddings are complete

**Error responses**:

| Status | Code | Condition |
|--------|------|-----------|
| 400 | `INVALID_PAYLOAD` | Missing required fields or content too large |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Non-JSON content type |
| 429 | `RATE_LIMIT_EXCEEDED` | |

---

### GET `/health`

Reports operational status of all dependent services (FR-022).

**Response 200 — all healthy**:

```json
{
  "status": "healthy",
  "timestamp": "2026-02-24T10:30:00Z",
  "services": {
    "postgres": { "status": "healthy", "latency_ms": 3 },
    "redis": { "status": "healthy", "latency_ms": 1 },
    "chromadb": { "status": "healthy", "latency_ms": 5 },
    "openai": { "status": "healthy", "latency_ms": 120 },
    "huggingface": { "status": "healthy", "latency_ms": 85 },
    "faster_whisper": { "status": "healthy", "latency_ms": 0 },
    "tts": { "status": "healthy", "latency_ms": 0 }
  }
}
```

**Response 207 — degraded (non-critical subsystem down)**:

```json
{
  "status": "degraded",
  "timestamp": "2026-02-24T10:30:00Z",
  "services": {
    "postgres": { "status": "healthy", "latency_ms": 3 },
    "redis": { "status": "healthy", "latency_ms": 1 },
    "chromadb": { "status": "unhealthy", "error": "connection refused" },
    "openai": { "status": "healthy", "latency_ms": 120 },
    "huggingface": { "status": "healthy", "latency_ms": 85 },
    "faster_whisper": { "status": "healthy", "latency_ms": 0 },
    "tts": { "status": "healthy", "latency_ms": 0 }
  }
}
```

**Response 503 — critical service down**:

Returned when PostgreSQL, Redis, or both LLM providers are unavailable. Body follows same shape as 207.

**Notes**:
- Health check runs lightweight probes (ping/SELECT 1 / ChromaDB heartbeat)
- Degraded status MUST be reported within 5 seconds of a dependency failure (SC-008)
- This endpoint is exempt from rate limiting

---

## Rate Limiting

Applied to all endpoints except `/health`.

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Total requests allowed per window |
| `X-RateLimit-Remaining` | Remaining requests in current window |
| `X-RateLimit-Reset` | UTC epoch seconds when window resets |
| `Retry-After` | Seconds to wait before retrying (on 429) |

Default limits (configurable via environment variables):

| Endpoint group | Limit | Window |
|----------------|-------|--------|
| `/sessions` | 20 req | 60 s |
| `/conversations/*` | 60 req | 60 s |
| `/documents/ingest` | 10 req | 60 s |

---

## Payload Validation

All inbound REST payloads are validated via Pydantic strict models in `interface/dtos/rest_responses.py`. Validation failures return 400 with the standard error body including a `details` array of field-level errors:

```json
{
  "error": {
    "code": "INVALID_PAYLOAD",
    "message": "Request body failed validation.",
    "details": [
      { "field": "content", "issue": "field required" }
    ],
    "request_id": "req-uuid"
  }
}
```
