"""Task-Oriented Dialogue pipeline orchestrator."""

from __future__ import annotations

import os
import re
import unicodedata

import structlog

from src.application.ports.dst_port import DSTPort
from src.application.ports.nlg_port import NLGPort
from src.application.ports.nlu_port import NLUPort
from src.application.ports.policy_port import PolicyPort
from src.domain.entities.dialogue_state import (
    DialogueState,
    NLUResult,
    PolicyAction,
    PolicyDecision,
)

logger = structlog.get_logger(__name__)

# ── Keyword-based intent override ────────────────────────────────────────────
# JointBERT (PhoATIS) has no affirm/deny/greet/farewell intents.
# We detect these common Vietnamese phrases BEFORE running the model
# and override the intent when matched.

_AFFIRM_PATTERNS: list[str] = [
    "vâng", "vang", "có", "co", "đúng", "dung", "đúng rồi", "dung roi",
    "ok", "okay", "được", "duoc", "ừ", "u", "uh", "đồng ý", "dong y",
    "xác nhận", "xac nhan", "phải", "phai", "chính xác", "chinh xac",
    "yes", "yeah", "đúng vậy", "dung vay", "rồi", "roi",
    "dạ", "da", "dạ vâng", "da vang", "dạ có", "da co",
    "vâng ạ", "vang a", "có ạ", "co a", "đúng rồi ạ", "dung roi a",
    "vâng đúng rồi", "vang dung roi",
]

_DENY_PATTERNS: list[str] = [
    "không", "khong", "sai", "sai rồi", "sai roi", "chưa đúng", "chua dung",
    "không phải", "khong phai", "no", "chưa", "chua", "không đúng", "khong dung",
    "thay đổi", "thay doi", "sửa", "sua", "hủy", "huy",
]

_GREET_PATTERNS: list[str] = [
    "xin chào", "xin chao", "chào", "chao", "hello", "hi",
    "chào bạn", "chao ban", "alo",
]

_FAREWELL_PATTERNS: list[str] = [
    "tạm biệt", "tam biet", "bye", "goodbye", "chào tạm biệt",
    "hẹn gặp lại", "hen gap lai",
    "cảm ơn", "cam on", "cám ơn", "xin cảm ơn",
    "kết thúc cuộc gọi", "ket thuc cuoc goi",
    "kết thúc", "ket thuc",
    "ngắt máy", "ngat may",
    "dừng cuộc gọi", "dung cuoc goi",
    "end call", "hang up",
]

# Whisper often hallucinates short confirmation audio as farewell phrases.
# These are REMOVED from farewell patterns and handled specially below:
# "cảm ơn", "cam on" — only treated as farewell if user explicitly wants to end.

# Patterns that Whisper commonly hallucinates for very short audio clips.
# When the dialogue is in CONFIRM state (waiting for yes/no), these
# hallucinations should be treated as affirmation rather than their
# literal meaning.
_WHISPER_HALLUCINATION_FAREWELL: list[str] = [
    "cảm ơn", "cam on", "cám ơn", "cam on",
    "cảm ơn các bạn đã theo dõi", "cam on cac ban da theo doi",
    "cảm ơn bạn đã theo dõi", "hẹn gặp lại các bạn",
    "xin cảm ơn", "xin cam on",
]


def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching.

    Also handles đ/Đ → d which is a separate letter, not a diacritic.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _normalize_for_match(text: str) -> str:
    """Strip punctuation, extra whitespace, and lowercase for keyword matching.

    Also applies NFC normalization to handle composed vs decomposed Unicode
    from different STT engines.
    """
    # NFC normalize to ensure consistent Unicode representation
    text = unicodedata.normalize("NFC", text)
    # Remove trailing/leading punctuation that Whisper appends
    text = re.sub(r"[.,;:!?]+", "", text)
    return " ".join(text.lower().split())


def _detect_keyword_intent(
    text: str,
    *,
    awaiting_confirmation: bool = False,
    has_booking: bool = False,
) -> str | None:
    """Return an override intent if the text matches a keyword pattern.

    Args:
        text: Raw transcription from STT (may include punctuation).
        awaiting_confirmation: True when the policy is in CONFIRM state.
            Used to catch Whisper hallucinations that turn "vâng" into
            "Cảm ơn." etc.
        has_booking: True when the user has filled slots (booking in progress).
            Suppresses greet detection so mid-booking greetings don't
            interrupt the flow.
    """
    clean = _normalize_for_match(text)
    ascii_clean = _strip_diacritics(clean)

    # Only check short utterances (1-10 words) — longer ones are real speech
    word_count = len(clean.split())
    if word_count > 10:
        return None

    # ── Affirm (highest priority when awaiting confirmation) ──────────
    for pattern in _AFFIRM_PATTERNS:
        pat_norm = _normalize_for_match(pattern)
        pat_ascii = _strip_diacritics(pat_norm)
        if clean == pat_norm or ascii_clean == pat_ascii:
            return "affirm"

    # ── Deny ──────────────────────────────────────────────────────────
    for pattern in _DENY_PATTERNS:
        pat_norm = _normalize_for_match(pattern)
        pat_ascii = _strip_diacritics(pat_norm)
        if clean == pat_norm or ascii_clean == pat_ascii:
            return "deny"

    # ── Whisper hallucination guard ───────────────────────────────────
    # When awaiting confirmation, Whisper often hallucinates short "vâng"
    # audio as "Cảm ơn." or "Tạm biệt." — treat these as affirmation.
    if awaiting_confirmation:
        for pattern in _WHISPER_HALLUCINATION_FAREWELL:
            pat_norm = _normalize_for_match(pattern)
            pat_ascii = _strip_diacritics(pat_norm)
            if clean == pat_norm or ascii_clean == pat_ascii:
                return "affirm"
        # Also catch farewell patterns as affirmation during confirmation
        for pattern in _FAREWELL_PATTERNS:
            pat_norm = _normalize_for_match(pattern)
            pat_ascii = _strip_diacritics(pat_norm)
            if clean == pat_norm or ascii_clean == pat_ascii:
                return "affirm"

    # ── Greet — only when not in active booking ─────────────────────
    if not has_booking:
        for pattern in _GREET_PATTERNS:
            pat_norm = _normalize_for_match(pattern)
            pat_ascii = _strip_diacritics(pat_norm)
            if clean == pat_norm or ascii_clean == pat_ascii or clean.startswith(pat_norm):
                return "greet"

    # ── Farewell (only when NOT awaiting confirmation) ────────────────
    for pattern in _FAREWELL_PATTERNS:
        pat_norm = _normalize_for_match(pattern)
        pat_ascii = _strip_diacritics(pat_norm)
        if clean == pat_norm or ascii_clean == pat_ascii or clean.startswith(pat_norm):
            return "farewell"

    return None


def _is_farewell_text(text: str) -> bool:
    """Broad farewell detection using substring matching as a safety net.

    This catches farewell phrases even when exact keyword matching fails
    due to Unicode normalization differences between STT output and patterns.
    """
    normalized = unicodedata.normalize("NFC", text.lower().strip())
    ascii_text = _strip_diacritics(normalized)
    farewell_substrings = [
        "tam biet", "goodbye", "bye", "ket thuc",
        "ngat may", "dung cuoc goi", "end call", "hang up",
        "hen gap lai",
    ]
    return any(sub in ascii_text for sub in farewell_substrings)


_MAX_SESSIONS = int(os.environ.get("TOD_MAX_SESSIONS", "1000"))


class TODPipelineUseCase:
    """Orchestrates the NLU -> DST -> Policy -> NLG pipeline.

    Manages per-session dialogue state and routes each user turn
    through the four-stage TOD pipeline.
    """

    def __init__(
        self,
        nlu: NLUPort,
        dst: DSTPort,
        policy: PolicyPort,
        nlg: NLGPort,
    ) -> None:
        self._nlu = nlu
        self._dst = dst
        self._policy = policy
        self._nlg = nlg
        self._states: dict[str, DialogueState] = {}

    def _evict_if_needed(self) -> None:
        """Evict the oldest session state if at capacity."""
        if len(self._states) > _MAX_SESSIONS:
            oldest_key = next(iter(self._states))
            del self._states[oldest_key]

    def get_or_create_state(self, session_id: str) -> DialogueState:
        """Get existing dialogue state or create a new one."""
        if session_id not in self._states:
            self._evict_if_needed()
            self._states[session_id] = DialogueState.create(session_id)
        return self._states[session_id]

    def clear_state(self, session_id: str) -> None:
        """Remove dialogue state for a session."""
        self._states.pop(session_id, None)

    async def process_turn(
        self,
        session_id: str,
        user_text: str,
    ) -> dict[str, object]:
        """Process a single dialogue turn through the full pipeline.

        Args:
            session_id: Session identifier.
            user_text: User input text (from STT or text input).

        Returns:
            Dict with response_text, nlu, state, action, target_slot
            for both API response and pipeline visualization.
        """
        log = logger.bind(session_id=session_id, turn_text=user_text[:80])
        state = self.get_or_create_state(session_id)

        # 0a. Post-execute handling: keep state so user can add details
        #     (dates, airline, class) to the same booking.  Only reset
        #     for a genuinely new booking when NLU detects NEW city slots.
        #     This check runs AFTER NLU below; see step 2b.

        # 0b. Keyword intent override — JointBERT (PhoATIS) lacks
        #     affirm/deny/greet/farewell intents, so we detect them first.
        #     When all required slots are filled and we're waiting for
        #     confirmation, Whisper hallucinations (e.g., "Cảm ơn." instead
        #     of "Vâng.") are caught and treated as affirmation.
        awaiting_confirm = (
            not state.missing_required() and not state.confirmed and not state.executed
        )
        keyword_intent = _detect_keyword_intent(
            user_text,
            awaiting_confirmation=awaiting_confirm,
            has_booking=bool(state.filled_slots()),
        )

        # 1. NLU: understand user intent and extract slots
        nlu_result = await self._nlu.understand(user_text)
        # Ensure raw text is always available for downstream DST keyword fill
        nlu_result.raw_text = user_text

        # Apply keyword override when detected
        if keyword_intent:
            log.info(
                "tod_keyword_override",
                original_intent=nlu_result.intent,
                override_intent=keyword_intent,
            )
            nlu_result = NLUResult(
                intent=keyword_intent,
                intent_confidence=0.95,
                slots=nlu_result.slots,
                raw_text=nlu_result.raw_text,
            )

        log.info(
            "tod_nlu_complete",
            intent=nlu_result.intent,
            confidence=round(nlu_result.intent_confidence, 2),
            slot_count=len(nlu_result.slots),
            slots={s.name: s.value for s in nlu_result.slots},
        )

        # 2a. Post-execute: check if this is a NEW booking (new cities)
        #     or supplementary info for the current booking.
        if state.executed:
            new_city_slots = {
                s.name for s in nlu_result.slots
                if s.name in ("fromloc.city_name", "toloc.city_name")
            }
            if new_city_slots:
                # New cities → start a fresh booking
                log.info("tod_new_booking_after_execute", session_id=session_id)
                state.reset_for_new_booking()
            elif keyword_intent == "farewell":
                # Farewell after execute → end the call
                log.info("tod_farewell_after_execute", session_id=session_id)
            elif keyword_intent == "greet":
                # Greeting after execute → don't reset, let policy handle
                pass
            elif _is_farewell_text(user_text):
                # Safety net: detect farewell from raw text even when keyword
                # detection misses due to Unicode normalization differences
                keyword_intent = "farewell"
                log.info("tod_farewell_text_fallback_after_execute", session_id=session_id)
            else:
                # Supplementary info (dates, airline, class) → update current booking
                # and re-confirm with the updated details
                log.info("tod_supplement_after_execute", session_id=session_id)
                state.executed = False
                state.confirmed = False

        # 2b. Farewell shortcut: if farewell detected (keyword or fallback),
        #     skip DST/policy and go straight to farewell response.
        if keyword_intent == "farewell":
            state.turn_count += 1
            self._states[session_id] = state
            decision = PolicyDecision(action=PolicyAction.FAREWELL)
            response_text = self._nlg.generate(decision, state)
            log.info("tod_farewell_shortcut", action=decision.action.value)
            state_dict = state.to_dict()
            state_dict["slots"] = state.filled_slots()
            return {
                "response_text": response_text,
                "nlu": {
                    "intent": nlu_result.intent,
                    "confidence": nlu_result.intent_confidence,
                    "slots": {s.name: s.value for s in nlu_result.slots},
                },
                "state": state_dict,
                "action": decision.action.value,
                "target_slot": decision.target_slot,
                "should_end_call": True,
            }

        # 2c. DST: update belief state
        state = self._dst.update(state, nlu_result)
        self._states[session_id] = state
        log.debug(
            "tod_dst_updated",
            turn_count=state.turn_count,
            filled_slots=state.filled_slots(),
            missing_required=state.missing_required(),
        )

        # 3. Policy: decide next action
        decision = self._policy.decide(state)
        log.info(
            "tod_policy_decided",
            action=decision.action.value,
            target_slot=decision.target_slot,
        )

        # 4. Handle confirmation / denial from user
        if state.intent == "affirm" and decision.action == PolicyAction.CONFIRM:
            state.confirmed = True
            decision = self._policy.decide(state)
            log.info("tod_confirmation_accepted", new_action=decision.action.value)
        elif state.intent == "deny" and decision.action == PolicyAction.CONFIRM:
            # User denied — reset confirmation, keep filled slots so user
            # can correct specific details rather than starting from scratch.
            state.confirmed = False
            # If user provided new slot values in same turn, DST already updated them.
            # Re-run policy — if all slots still filled, re-confirm with updated info.
            # Only ask for clarification if no new slots were provided.
            if nlu_result.slots:
                decision = self._policy.decide(state)
                log.info("tod_denial_with_correction", new_action=decision.action.value)
            else:
                decision = PolicyDecision(action=PolicyAction.CLARIFY)
                log.info("tod_denial_correction", new_action=decision.action.value)

        # 5. NLG: generate response
        response_text = self._nlg.generate(decision, state)

        # 6. Post-EXECUTE: mark as executed so the policy knows not to
        #    re-confirm.  State is kept so user sees booking details.
        if decision.action == PolicyAction.EXECUTE:
            state.executed = True
            log.info("tod_post_execute", session_id=session_id)
        log.debug("tod_nlg_generated", response_length=len(response_text))

        state_dict = state.to_dict()
        state_dict["slots"] = state.filled_slots()  # Only show filled, not all None slots

        return {
            "response_text": response_text,
            "nlu": {
                "intent": nlu_result.intent,
                "confidence": nlu_result.intent_confidence,
                "slots": {s.name: s.value for s in nlu_result.slots},
            },
            "state": state_dict,
            "action": decision.action.value,
            "target_slot": decision.target_slot,
            "should_end_call": decision.action == PolicyAction.FAREWELL,
        }
