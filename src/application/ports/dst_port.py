"""Port interface for Dialogue State Tracking."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.dialogue_state import DialogueState, NLUResult


class DSTPort(Protocol):
    """Adapter interface for dialogue state tracking."""

    def update(
        self,
        state: DialogueState,
        nlu_result: NLUResult,
    ) -> DialogueState:
        """Update belief state from NLU output.

        Args:
            state: Current dialogue state.
            nlu_result: NLU extraction from current turn.

        Returns:
            Updated dialogue state.
        """
        ...
