"""Hybrid Dialogue State Tracking adapter.

Merges NLU slot extractions into the cumulative dialogue state,
applying confidence thresholds and slot overwrite rules.
"""

from __future__ import annotations

import re

from src.domain.entities.dialogue_state import DialogueState, NLUResult

# Minimum confidence to accept a slot value
_SLOT_CONFIDENCE_THRESHOLD: float = 0.5

# Trailing punctuation that STT may append to slot values
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")


def _clean_slot_value(value: str) -> str:
    """Strip trailing punctuation and excess whitespace from slot values."""
    return _TRAILING_PUNCT.sub("", value).strip()


class HybridDSTAdapter:
    """Rule-based DST that merges NLU slots into the belief state.

    Implements the DSTPort protocol.
    """

    def update(
        self,
        state: DialogueState,
        nlu_result: NLUResult,
    ) -> DialogueState:
        """Update belief state from NLU output.

        - Overwrites slot values when the new extraction has higher confidence.
        - Increments turn counter.
        - Stores intent from latest turn.
        """
        state.intent = nlu_result.intent
        state.intent_confidence = nlu_result.intent_confidence
        state.turn_count += 1

        for slot in nlu_result.slots:
            if slot.confidence >= _SLOT_CONFIDENCE_THRESHOLD:
                state.slots[slot.name] = _clean_slot_value(slot.value)

        # Record turn in history
        state.history.append({
            "turn": state.turn_count,
            "intent": nlu_result.intent,
            "slots": {s.name: s.value for s in nlu_result.slots},
        })

        return state
