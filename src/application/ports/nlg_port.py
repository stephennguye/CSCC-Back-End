"""Port interface for Natural Language Generation."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.dialogue_state import DialogueState, PolicyDecision


class NLGPort(Protocol):
    """Adapter interface for response generation."""

    def generate(
        self,
        decision: PolicyDecision,
        state: DialogueState,
    ) -> str:
        """Generate Vietnamese response text.

        Args:
            decision: Policy decision (action + target slot).
            state: Current dialogue state for slot value insertion.

        Returns:
            Vietnamese response text.
        """
        ...
