# WebSocket Contract: AI Call Center Backend

**Phase**: 1 — Design & Contracts
**Branch**: `001-ai-call-center-backend`
**Date**: 2026-02-24

---

## Connection

**Endpoint**: `ws://{host}/ws/calls/{session_id}`

**Upgrade requirements**:
- `session_id` must be a valid UUID
- Authentication token must be present (enforced at gateway layer)
- Connection rejected with HTTP 401 if auth fails; HTTP 404 if session not found

---

## Frame Format

All text frames are JSON. Binary frames carry raw audio data.

### Frame Envelope (inbound and outbound text frames)

```json
{
  "type": "<frame_type>",
  "session_id": "<uuid>",
  "payload": { ... }
}
```

---

## Inbound Frames (Client → Server)

### `audio.chunk`

Delivers a chunk of streaming caller audio.

```json
{
  "type": "audio.chunk",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "sequence": 42,
    "codec": "pcm_16khz_mono",
    "data": "<base64-encoded binary>"
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `sequence` | integer | yes | Monotonically increasing chunk index |
| `codec` | string | yes | Supported: `pcm_16khz_mono`, `opus_48khz` |
| `data` | string (base64) | yes | Raw audio bytes, max 4 KB per chunk |

**Alternative**: Binary frames may be sent directly without JSON envelope for performance. When binary frames are used, sequence tracking is managed via Redis session state.

---

### `audio.end`

Signals that the caller has finished speaking (end of utterance).

```json
{
  "type": "audio.end",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "sequence": 103
  }
}
```

---

### `session.resume`

Client signals intent to resume an existing session after a connection drop. Must be sent as the first text frame immediately after the WebSocket upgrade.

```json
{
  "type": "session.resume",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "last_sequence": 103
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `last_sequence` | integer | yes | Last `audio.chunk` sequence the client successfully sent; server discards any buffered frames with sequence ≤ this value |

**Server behaviour on receipt**:
1. Validates the `session_id` is in `active` state.
2. Emits `session.state` with `state: "listening"` to signal resumption.
3. If session is `ended` or `error`, responds with an `error` frame (`SESSION_ENDED`) and closes the connection.

---

### `session.end`

Caller initiates graceful session termination.

```json
{
  "type": "session.end",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {}
}
```

**Server behavior on receipt**:
1. Flush remaining TTS buffer
2. Persist conversation transcript to PostgreSQL
3. Enqueue `ExtractClaimsUseCase` and `GenerateReminderUseCase` as background tasks
4. Transition `CallSession.state` to `ended`
5. Close WebSocket with code 1000

---

## Outbound Frames (Server → Client)

### `transcript.partial`

Incremental ASR transcript — emitted during utterance (FR-002).

```json
{
  "type": "transcript.partial",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "text": "I need help with",
    "confidence": 0.87,
    "segment_id": "seg-001"
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `text` | string | Partial transcript text so far |
| `confidence` | float | ASR confidence score (0.0–1.0) |
| `segment_id` | string | Groups partials belonging to the same utterance |

---

### `transcript.final`

Final ASR transcript for a complete utterance.

```json
{
  "type": "transcript.final",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "text": "I need help with my enrollment status.",
    "confidence": 0.94,
    "segment_id": "seg-001"
  }
}
```

**Below-threshold behavior**: If `confidence < ASR_CONFIDENCE_THRESHOLD` (configurable, default `0.70`), the server emits `transcript.low_confidence` instead (see below) and does NOT forward to LLM (FR-002a).

---

### `transcript.low_confidence`

Emitted when ASR confidence is below threshold; prompts caller to repeat (FR-002a).

```json
{
  "type": "transcript.low_confidence",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "segment_id": "seg-002",
    "prompt_message": "I didn't catch that — could you repeat?"
  }
}
```

**Server behavior**: TTS synthesizes and streams `prompt_message` back to caller. Low-confidence segment is discarded.

---

### `response.token`

A single streamed token from the LLM (enables low-latency streaming display if needed).

```json
{
  "type": "response.token",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "token": "Your",
    "turn_id": "turn-007"
  }
}
```

---

### `transcript.ai_final`

Complete assembled AI turn — emitted once after `audio.response.end` for each AI turn. Provides the full AI response text with a server-authoritative timestamp, enabling the client to build an accurate transcript and post-call export that mirrors the server's `Message` records (frontend FR-012, FR-022).

```json
{
  "type": "transcript.ai_final",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "turn_id": "turn-007",
    "text": "Your enrollment status is currently active for Spring 2026.",
    "timestamp": "2026-02-24T10:00:16Z",
    "sequence_number": 4
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `turn_id` | string | Matches `response.token` and `audio.response` frames for this turn |
| `text` | string | Full concatenated AI response text for this turn |
| `timestamp` | ISO 8601 UTC | Server wall-clock time of turn start; matches the `Message.timestamp` persisted to PostgreSQL |
| `sequence_number` | integer | Matches `Message.sequence_number` in storage |

---

### `audio.response`

Delivers synthesized TTS audio to the client. **Binary WebSocket frames are the primary delivery mechanism**; JSON+base64 is a fallback for environments where binary WebSocket frames are unsupported.

#### Primary: Binary Frame Delivery

When binary frames are used, the server first emits an `audio.response.start` JSON text frame to announce metadata, followed by one or more raw binary WebSocket frames carrying PCM/WAV data, completed by `audio.response.end`.

**`audio.response.start`** (JSON text frame):

```json
{
  "type": "audio.response.start",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "turn_id": "turn-007",
    "codec": "pcm_16khz_mono"
  }
}
```

Followed by N raw **binary WebSocket frames** — each frame carries a chunk of PCM/WAV audio bytes, max 4 KB per frame. The client buffers and begins playback as soon as the first binary frame arrives (FR-016).

Completed by `audio.response.end` (see below).

#### Fallback: JSON Envelope

Used only when binary frames are not supported by the transport layer.

```json
{
  "type": "audio.response",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "sequence": 1,
    "turn_id": "turn-007",
    "codec": "pcm_16khz_mono",
    "data": "<base64-encoded binary>"
  }
}
```

---

### `audio.response.end`

Signals that TTS streaming for the current turn is complete.

```json
{
  "type": "audio.response.end",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "turn_id": "turn-007"
  }
}
```

---

### `session.state`

Emitted by the server on every call-state transition. The client MUST update its UI state indicator on receipt (frontend FR-003, SC-004).

```json
{
  "type": "session.state",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "state": "listening",
    "previous_state": "connecting",
    "timestamp": "2026-02-24T10:00:05Z"
  }
}
```

| `state` value | Trigger |
|---------------|---------|
| `connecting` | WebSocket upgrade accepted; session initialising |
| `listening` | Server ready to receive user audio (also sent on successful `session.resume`) |
| `ai_thinking` | User utterance received; LLM processing in progress |
| `ai_speaking` | First `audio.response` binary frame (or `audio.response.start`) emitted |
| `call_ended` | `session.end` processed and `CallSession.state` set to `ended` |
| `reconnecting` | Server detects client disconnect and the session is still resumable |
| `error` | Unrecoverable error; see accompanying `error` frame for details |

**Timing guarantees**: `session.state` MUST be emitted before or concurrently with the first frame of the new state (e.g., `session.state {ai_speaking}` is emitted no later than the first binary audio frame).

---

### `barge_in.ack`

Acknowledges that the server has halted the in-progress AI response due to caller interruption (FR-005).

```json
{
  "type": "barge_in.ack",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "halted_turn_id": "turn-007"
  }
}
```

---

### `llm.fallback`

Notifies the client that the system switched to the fallback LLM (FR-006). Informational only — no client action required.

```json
{
  "type": "llm.fallback",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "reason": "primary_llm_timeout",
    "fallback_model": "mistralai/Mistral-7B-Instruct-v0.2"
  }
}
```

---

### `error`

Structured error notification.

```json
{
  "type": "error",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "code": "LLM_TIMEOUT",
    "message": "Language model did not respond within the timeout window.",
    "recoverable": true
  }
}
```

| `code` value | Recoverable | Meaning |
|---|---|---|
| `LLM_TIMEOUT` | true | Primary LLM timed out; fallback triggered |
| `LLM_FALLBACK_EXHAUSTED` | false | Both LLM providers unavailable |
| `TRANSCRIPTION_ERROR` | true | STT processing failed for this segment |
| `PROMPT_INJECTION_DETECTED` | false | Input sanitizer blocked malicious payload |
| `SESSION_ENDED` | false | Operation attempted on closed session |

**Non-recoverable errors**: Server closes the WebSocket with code 1011 after emitting.

---

## Reconnection Protocol

The frontend MUST automatically reconnect without user intervention when the WebSocket drops (frontend FR-018, US-5).

### Client-Side Reconnection Flow

1. On `onclose` / `onerror`, client enters **Reconnecting** UI state.
2. Client obtains a fresh session token by calling `POST /sessions` **with the existing `session_id`** — token rotation only, no new session created.
3. Client opens a new WebSocket to `ws://{host}/ws/calls/{session_id}` with the new token.
4. Client sends `session.resume` as the first text frame, including `last_sequence`.
5. Server responds with `session.state { state: "listening" }` on success.
6. Call continues from the point of interruption; in-flight transcript is preserved.

### Session Resumability Rules

| Condition | Server behaviour |
|-----------|------------------|
| Session is `active`, reconnect within **60 seconds** | Resumable; server emits `session.state { state: "listening" }` |
| Session is `active`, reconnect after **60 seconds** | Non-resumable; server emits `error { code: "SESSION_EXPIRED" }` and closes |
| Session is `ended` or `error` | Non-resumable; server emits `error { code: "SESSION_ENDED" }` and closes |

### Retry Schedule

The client MUST use exponential back-off with jitter:

| Attempt | Delay |
|---------|-------|
| 1 | 500 ms |
| 2 | 1 s |
| 3 | 2 s |
| 4+ | 4 s (max) |

After **5 failed attempts** the client MUST surface a permanent error with an option to start a new call (frontend FR-018, US-5 AC-3).

### Token Refresh for Reconnection

`POST /sessions` accepts `{ "session_id": "<existing-uuid>" }` in the request body to issue a new token for an existing active session, without creating a new `CallSession` record.

```json
POST /api/v1/sessions
{ "session_id": "550e8400-e29b-41d4-a716-446655440000" }

201 →
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 300,
  "ws_url": "ws://{host}/ws/calls/550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Barge-In Flow

When the server detects `audio.chunk` while `turn_state = generating`:

1. Publish cancel signal to Redis `barge_in:{session_id}` Pub/Sub channel
2. LLM streaming coroutine detects cancel signal and raises `CancelledError`
3. TTS streaming halts mid-chunk
4. Server emits `barge_in.ack`
5. New audio frame is processed from the top of the pipeline

---

## RAG Source Attribution

All `response.token` and `audio.response` turns produced with RAG context MUST include source attribution in an accompanying `rag.context` frame:

```json
{
  "type": "rag.context",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "turn_id": "turn-007",
    "sources": [
      {
        "document_id": "doc-uuid",
        "chunk_index": 3,
        "title": "Enrollment Policy 2025",
        "confidence": 0.91
      }
    ]
  }
}
```

This satisfies Constitution III (grounding, source attribution).

---

## Validation

All inbound text frames MUST be validated against Pydantic strict models in `interface/dtos/ws_messages.py` before processing. Malformed frames are rejected with an `error` frame (code `INVALID_PAYLOAD`) and the frame is discarded — the connection is NOT closed.
