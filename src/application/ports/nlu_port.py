"""Port interface for Natural Language Understanding."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.dialogue_state import NLUResult


class NLUPort(Protocol):
    """Adapter interface for NLU model inference."""

    async def understand(self, text: str) -> NLUResult:
        """Parse user text into intent + slots.

        Args:
            text: Vietnamese text input from STT or user.

        Returns:
            NLUResult with intent, confidence, and extracted slots.
        """
        ...
