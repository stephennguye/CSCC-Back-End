"""Rule-based dialogue policy adapter.

Decides the next system action based on current dialogue state:
greet, request missing slots, confirm, execute, or escalate.
"""

from __future__ import annotations

from src.domain.entities.dialogue_state import (
    DialogueState,
    PolicyAction,
    PolicyDecision,
)


class RulePolicyAdapter:
    """Deterministic policy using hand-crafted rules.

    Implements the PolicyPort protocol.
    """

    def decide(self, state: DialogueState) -> PolicyDecision:
        """Determine next action based on current dialogue state."""
        # Greeting intent — always respond regardless of state
        if state.intent == "greet":
            return PolicyDecision(action=PolicyAction.GREET)

        # Farewell intent — always respond regardless of state
        if state.intent == "farewell":
            return PolicyDecision(action=PolicyAction.FAREWELL)

        # Post-execution: booking was already processed.
        # If the user provides new flight info, start a new booking.
        # Otherwise, offer further assistance.
        if state.executed:
            return PolicyDecision(action=PolicyAction.PROVIDE_INFO)

        # Info-request intents — respond with a polite clarification
        if state.intent in ("atis_abbreviation", "atis_ground_service"):
            return PolicyDecision(action=PolicyAction.CLARIFY)

        # Booking flow: check required slots
        missing = state.missing_required()

        if missing:
            return PolicyDecision(
                action=PolicyAction.REQUEST_SLOT,
                target_slot=missing[0],
            )

        # All required slots filled — confirm or execute
        if not state.confirmed:
            return PolicyDecision(action=PolicyAction.CONFIRM)

        # Confirmed — execute the booking
        return PolicyDecision(action=PolicyAction.EXECUTE)
