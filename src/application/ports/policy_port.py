"""Port interface for dialogue policy."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.dialogue_state import DialogueState, PolicyDecision


class PolicyPort(Protocol):
    """Adapter interface for dialogue policy decisions."""

    def decide(self, state: DialogueState) -> PolicyDecision:
        """Determine next action based on current state.

        Args:
            state: Current dialogue belief state.

        Returns:
            PolicyDecision with action and optional target slot.
        """
        ...
