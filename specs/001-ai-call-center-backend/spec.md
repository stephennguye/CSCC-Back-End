# Feature Specification: AI Call Center Backend

**Feature Branch**: `001-ai-call-center-backend`
**Created**: 2026-02-24
**Status**: Draft
**Input**: AI-powered call center backend with real-time voice pipeline, RAG retrieval, session management, claim extraction, and reminder generation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Real-Time AI Voice Call (Priority: P1)

A student calls the support line. The system receives their audio in real time, converts it to text, retrieves relevant knowledge, generates a context-aware response, and streams synthesized speech back to the caller — all within the same conversation turn.

**Why this priority**: This is the foundational capability that all other features depend on. Without a working real-time voice pipeline, the product does not exist.

**Independent Test**: Can be fully tested by simulating an inbound voice session over WebSocket, speaking a question, and verifying that a synthesized audio response is returned within the latency target.

**Acceptance Scenarios**:

1. **Given** a caller connects and speaks, **When** their audio is received, **Then** the system transcribes speech to text in real time without waiting for the utterance to complete.
2. **Given** a transcription is available, **When** the AI generates a response, **Then** the first words of synthesized audio are streamed back to the caller within 1.5 seconds.
3. **Given** a caller speaks while the AI is responding, **When** the interruption is detected, **Then** the current AI response is halted and the new input is processed.
4. **Given** the primary LLM is unreachable, **When** the pipeline requires a response, **Then** the system automatically switches to the fallback model without call disruption.

---

### User Story 2 - Knowledge-Grounded Responses (Priority: P2)

During a call, the AI retrieves relevant context from the institution's knowledge base so its answers are accurate and grounded in official information rather than general model knowledge.

**Why this priority**: Without RAG, the AI may hallucinate or give incorrect institution-specific answers, eroding trust and usability.

**Independent Test**: Can be fully tested by ingesting a set of documents and issuing a question whose answer exists only in those documents, then verifying the response cites the correct information.

**Acceptance Scenarios**:

1. **Given** documents have been ingested into the knowledge base, **When** a caller asks a question related to those documents, **Then** the system retrieves the most relevant document chunks and incorporates them into the response.
2. **Given** a query is issued, **When** no sufficiently relevant documents exist, **Then** the AI responds transparently without fabricating document-based details.
3. **Given** multiple relevant documents exist, **When** the retrieval runs, **Then** results are ranked by relevance and the top-ranked context is used.

---

### User Story 3 - Structured Claim Extraction (Priority: P3)

After a conversation, a supervisor or downstream system can retrieve a structured summary of the call, including the student's name, the type of issue, urgency level, requested action, and any follow-up date mentioned.

**Why this priority**: Claim extraction converts unstructured conversations into actionable records, enabling case management and audit trails.

**Independent Test**: Can be fully tested by running a simulated conversation transcript through the extraction pipeline and asserting that all defined fields are populated or marked absent.

**Acceptance Scenarios**:

1. **Given** a conversation ends, **When** claim extraction runs, **Then** a structured claim record is produced containing: student_name, issue_category, urgency_level, requested_action, and follow_up_date.
2. **Given** a field cannot be determined from the conversation, **When** extraction completes, **Then** the field is marked as absent rather than guessed.
3. **Given** a claim record is created, **When** queried via the REST API, **Then** the full structured claim is returned with the associated session ID.

---

### User Story 4 - Reminder Generation (Priority: P4)

The system detects actionable commitments or follow-up items mentioned during a call and generates structured reminders that can be reviewed and acted upon.

**Why this priority**: Reminders close the loop on call outcomes by ensuring nothing is forgotten; they depend on conversation persistence but are independent of claim extraction.

**Independent Test**: Can be fully tested by running a transcript containing explicit follow-up language through the reminder pipeline and asserting that reminders are created with correct content and persisted.

**Acceptance Scenarios**:

1. **Given** a conversation contains phrases like "I will follow up on Monday" or "call back in 3 days", **When** reminder generation runs, **Then** a structured reminder record is created with a description and target date.
2. **Given** a reminder is created, **When** retrieved via the REST API, **Then** the reminder contains the source session ID, description, and target follow-up date.
3. **Given** no actionable items are detected, **When** reminder generation runs, **Then** no reminder records are created for that conversation.

---

### User Story 5 - Conversation History Access (Priority: P5)

A supervisor or integration system retrieves the full transcript, claims, and reminders for a past call via REST endpoints for audit, quality assurance, or downstream processing.

**Why this priority**: History access provides observability over completed interactions and enables human review of AI behavior.

**Independent Test**: Can be fully tested by completing a call, then querying the REST API to retrieve the conversation transcript, associated claims, and associated reminders.

**Acceptance Scenarios**:

1. **Given** a call has completed, **When** the conversation history endpoint is queried with the session ID, **Then** the full ordered transcript is returned.
2. **Given** claims have been extracted, **When** the claims endpoint is queried by session ID, **Then** the structured claim record is returned.
3. **Given** reminders exist, **When** the reminders endpoint is queried by session ID, **Then** all associated reminder records are returned.
4. **Given** a health check is requested, **When** the health endpoint is called, **Then** the system reports the status of all dependent services.

---

### Edge Cases

- What happens when the primary LLM provider is rate-limited or unavailable mid-call?
- How does the system handle a caller disconnecting abruptly before a response is sent?
- What if the ASR produces low-confidence transcription for an utterance? → **Resolved**: System prompts caller to repeat; low-confidence segment is discarded and not forwarded to the LLM (FR-002a).
- What if the knowledge base returns no relevant documents for a query?
- What if claim extraction cannot determine required fields from the conversation?
- What happens when a reminder's target date cannot be parsed from the transcript?
- How does the system behave when a WebSocket connection drops and the client reconnects?

## Requirements *(mandatory)*

### Functional Requirements

#### Real-Time Voice Pipeline

- **FR-001**: System MUST accept streaming audio input from connected callers in real time.
- **FR-002**: System MUST transcribe incoming audio to text incrementally, without waiting for an utterance to complete.
- **FR-002a**: When ASR confidence for a transcript segment falls below the configured threshold, the system MUST prompt the caller to repeat their utterance (e.g., "I didn't catch that — could you repeat?") and MUST NOT forward the low-confidence segment to the language model.
- **FR-003**: System MUST generate text responses using a language model with access to retrieved knowledge context.
- **FR-004**: System MUST convert text responses to audio and stream the synthesized speech back to the caller.
- **FR-005**: System MUST detect when a caller speaks during an active AI response and interrupt the current output to process the new input.
- **FR-006**: System MUST automatically switch to a secondary language model when the primary model is unavailable or fails, without interrupting the call.

#### Session Management

- **FR-007**: System MUST create and maintain a unique session for each active call, recording its state and short-term conversation history.
- **FR-008**: System MUST track WebSocket connection presence per session and clean up session state when a call ends or the connection is lost.
- **FR-009**: System MUST persist completed conversation transcripts to durable storage after a session ends.

#### RAG Knowledge Retrieval

- **FR-010**: System MUST support ingestion of documents into a searchable knowledge base, including processing them into retrievable chunks with associated embeddings.
- **FR-011**: System MUST retrieve the most relevant knowledge chunks for a given query and rank them by relevance before including them in the response context.
- **FR-012**: System MUST expose document ingestion through a dedicated API endpoint.

#### Claim Extraction

- **FR-013**: System MUST extract the following structured fields from each completed conversation transcript: student_name, issue_category, urgency_level, requested_action, follow_up_date.
- **FR-013a**: Claim extraction MUST execute asynchronously — it is enqueued on session close and processed by a background worker; session teardown MUST NOT block on extraction completion.
- **FR-014**: Claim extraction MUST be schema-driven and deterministic — the same transcript MUST always produce the same claim output.
- **FR-015**: System MUST persist extracted claims and associate them with the source conversation.
- **FR-016**: System MUST expose claims via a REST endpoint queryable by session ID.

#### Reminder Generation

- **FR-017**: System MUST detect actionable follow-up commitments within conversation transcripts.
- **FR-017a**: Reminder generation MUST execute asynchronously — it is enqueued on session close and processed by a background worker independently of claim extraction; session teardown MUST NOT block on reminder generation completion.
- **FR-018**: System MUST generate structured reminder records for each detected actionable item, including a description and target follow-up date where determinable.
- **FR-019**: System MUST persist reminders and associated them with the source conversation.
- **FR-020**: System MUST expose reminders via a REST endpoint queryable by session ID.

#### API & Observability

- **FR-021**: System MUST expose conversation history via a REST endpoint queryable by session ID.
- **FR-022**: System MUST provide a health check endpoint reporting the operational status of all dependent services.
- **FR-023**: System MUST validate all inbound payloads and reject malformed requests with descriptive error responses.
- **FR-024**: System MUST enforce rate limiting on all API endpoints.
- **FR-025**: System MUST defend against prompt injection by sanitizing conversational inputs before forwarding to the language model.
- **FR-026**: System MUST emit structured logs for all significant events (call start/end, LLM calls, errors, fallbacks).
- **FR-027**: System MUST support operational monitoring, enabling teams to track performance, identify bottlenecks, and trace the path of individual requests through the system.

#### Security & Privacy

- **FR-028**: System MUST ensure that all PII fields (student_name, conversation transcript content) are encrypted at rest using DB-level or field-level encryption; plain-text storage of PII is not permitted.
- **FR-029**: System MUST apply encryption at rest consistently across durable storage for CallSession transcripts, Message records, and Claim records.

### Key Entities

- **CallSession**: The single authoritative entity representing an active or completed call (formerly also referred to as "conversation"). The session ID serves as the conversation ID in all REST endpoints. Holds session identifier, connection state, short-term conversation buffer, and lifecycle timestamps. Ephemeral state lives in cache; completed records persist to durable storage.
- **Message**: A single turn in a conversation — either from the caller or the AI. Contains speaker role, transcript text, and timestamp. Belongs to a CallSession.
- **Claim**: Structured data record extracted from a completed CallSession. Contains student_name *(PII — encrypted at rest)*, issue_category, urgency_level, requested_action, follow_up_date, and a reference to the source CallSession (by session ID).
- **Reminder**: An actionable follow-up item derived from a completed CallSession. Contains a description, optional target date, creation timestamp, and a reference to the source CallSession (by session ID).
- **Document**: A knowledge base source record. Contains raw content, chunked segments, and metadata (title, source, ingestion date).
- **Embedding**: A vectorized representation of a document chunk, stored for similarity search. Associated with a Document chunk.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Callers receive the first word of an AI audio response within 1.5 seconds of finishing an utterance, measured from the end of the caller's speech to the start of the AI's audible reply.
- **SC-002**: The system sustains at least **100 simultaneous call sessions** without measurable latency degradation on any individual session (latency target per SC-001 must hold at full concurrency).
- **SC-003**: Claim extraction produces a fully populated structured record for at least 95% of conversations where the caller explicitly provided the relevant information.
- **SC-004**: Conversation transcripts, claims, and reminders are retrievable via REST within 500 milliseconds of the request.
- **SC-005**: The system recovers from primary LLM failure and routes to the fallback model within 2 seconds, with no call drop.
- **SC-006**: 100% of completed calls result in a persisted conversation transcript accessible via the history API.
- **SC-007**: Zero prompt injection attacks succeed in altering system behavior or leaking internal context, as verified by adversarial test scenarios.
- **SC-008**: All API endpoints respond to health checks accurately, reporting degraded status within 5 seconds of a dependency failure.

## Assumptions

- `CallSession` is the single canonical entity for both active and completed calls; "conversation ID" is a synonym for session ID used in REST endpoint paths (no separate `Conversation` entity exists).
- The caller's audio is delivered over WebSocket using a supported audio codec; codec negotiation details are an implementation concern.
- The institution's knowledge base documents are provided in a standard text-extractable format (PDF, Markdown, plain text).
- A single claim schema applies uniformly to all call types; per-call-type schemas are out of scope for this feature.
- Authentication and authorization for API access is handled at the infrastructure or gateway layer and is not part of this feature's scope.
- Reminders are generated post-call (or at session close) via an asynchronous background worker, not in real time during the conversation and not inline during session teardown.
- Data retention policy for conversations, claims, and reminders follows institutional standards and is configured externally.

## Clarifications

### Session 2026-02-24

- Q: What is the target number of simultaneous call sessions the system must sustain without latency degradation? → A: ~100 concurrent sessions
- Q: How must PII (student_name, transcript content) be protected at rest? → A: Encrypted at rest — DB-level or field-level encryption for PII fields
- Q: What is the execution model for claim extraction and reminder generation? → A: Asynchronous — enqueued on session close, processed by background worker (non-blocking)
- Q: How should the system handle low-confidence ASR transcription? → A: Prompt caller to repeat; discard low-confidence segment, do not forward to LLM
- Q: Are `CallSession` and `Conversation` the same entity, or separate? → A: Single entity — `CallSession` is the conversation record; session ID = conversation ID across all REST APIs
