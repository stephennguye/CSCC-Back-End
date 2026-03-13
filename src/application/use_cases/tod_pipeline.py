"""Task-Oriented Dialogue pipeline orchestrator."""

from __future__ import annotations

import logging
from dataclasses import asdict

from src.application.ports.dst_port import DSTPort
from src.application.ports.nlg_port import NLGPort
from src.application.ports.nlu_port import NLUPort
from src.application.ports.policy_port import PolicyPort
from src.domain.entities.dialogue_state import DialogueState, PolicyAction

logger = logging.getLogger(__name__)


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

    def get_or_create_state(self, session_id: str) -> DialogueState:
        """Get existing dialogue state or create a new one."""
        if session_id not in self._states:
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
        state = self.get_or_create_state(session_id)

        # 1. NLU: understand user intent and extract slots
        nlu_result = await self._nlu.understand(user_text)
        logger.info(
            "NLU result: intent=%s conf=%.2f slots=%d",
            nlu_result.intent,
            nlu_result.intent_confidence,
            len(nlu_result.slots),
        )

        # 2. DST: update belief state
        state = self._dst.update(state, nlu_result)
        self._states[session_id] = state

        # 3. Policy: decide next action
        decision = self._policy.decide(state)
        logger.info(
            "Policy: action=%s target=%s",
            decision.action.value,
            decision.target_slot,
        )

        # 4. Handle confirmation from user
        if (
            state.intent in ("affirm", "confirm", "yes")
            and decision.action == PolicyAction.CONFIRM
        ):
            state.confirmed = True
            decision = self._policy.decide(state)

        # 5. NLG: generate response
        response_text = self._nlg.generate(decision, state)

        return {
            "response_text": response_text,
            "nlu": {
                "intent": nlu_result.intent,
                "confidence": nlu_result.intent_confidence,
                "slots": {s.name: s.value for s in nlu_result.slots},
            },
            "state": state.to_dict(),
            "action": decision.action.value,
            "target_slot": decision.target_slot,
        }
