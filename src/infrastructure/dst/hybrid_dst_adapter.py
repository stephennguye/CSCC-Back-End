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

# Pattern to extract time from utterances like "6 giờ", "lúc 8 giờ sáng"
_TIME_PATTERN = re.compile(
    r"(?:lúc\s+)?(\d{1,2})\s*giờ(?:\s*(sáng|chiều|tối))?", re.IGNORECASE
)

# Pattern to extract Vietnamese dates like "ngày 20 tháng 1", "20 tháng 3",
# "ngày 5", or compound ranges "từ 20 tháng 1 đến ngày 21 tháng 3"
_DATE_PATTERN = re.compile(
    r"(?:ngày\s+)?(\d{1,2})\s*(?:tháng\s*(\d{1,2}))?",
    re.IGNORECASE,
)

# Keyword rules for slots that NLU may miss.
# Each entry: (slot_name, keyword, canonical_value)
_KEYWORD_SLOT_RULES: list[tuple[str, str, str]] = [
    ("round_trip", "khứ hồi", "khứ hồi"),
    ("round_trip", "một chiều", "một chiều"),
    ("class_type", "phổ thông", "phổ thông"),
    ("class_type", "thương gia", "thương gia"),
    ("class_type", "hạng nhất", "hạng nhất"),
    ("airline_name", "vietnam airlines", "Vietnam Airlines"),
    ("airline_name", "vietjet", "Vietjet Air"),
    ("airline_name", "bamboo", "Bamboo Airways"),
]


def _clean_slot_value(value: str) -> str:
    """Strip trailing punctuation and excess whitespace from slot values."""
    return _TRAILING_PUNCT.sub("", value).strip()


def _normalize_date_value(slot_name: str, value: str) -> str | None:
    """Strip redundant Vietnamese prefixes from date slot values.

    Returns None if the value is just a label word without actual data
    (e.g., "tháng" without a number, "ngày" without a number).
    """
    v = value.strip().rstrip(".,;:!?")
    if "day_number" in slot_name:
        # "ngày 20" → "20", "Ngày 20" → "20"
        v = re.sub(r"^[Nn]gày\s*", "", v)
        # Reject if no actual number remains (NLU tagged just "ngày")
        if not v or not any(c.isdigit() for c in v):
            return None
    elif "month_name" in slot_name:
        # Reject bare "tháng" without a number
        if not any(c.isdigit() for c in v):
            return None
    return v.strip()


_NEGATION_WORDS = {"không", "khong", "chưa", "chua", "đừng", "dung", "hủy", "huy"}


def _keyword_slot_fill(state: DialogueState, raw_text: str) -> None:
    """Fill missing slots using keyword detection on the raw transcript."""
    text_lower = raw_text.lower()

    for slot_name, keyword, canonical in _KEYWORD_SLOT_RULES:
        if not state.slots.get(slot_name) and keyword in text_lower:
            # Check for negation: look for negation word within 3 words before keyword
            kw_pos = text_lower.find(keyword)
            prefix = text_lower[:kw_pos].split()
            negated = any(w in _NEGATION_WORDS for w in prefix[-3:]) if prefix else False
            if not negated:
                state.slots[slot_name] = canonical

    # Time extraction: "6 giờ sáng", "lúc 8 giờ"
    if not state.slots.get("depart_time.time"):
        m = _TIME_PATTERN.search(raw_text)
        if m:
            hour = m.group(1)
            period = m.group(2) or ""
            time_str = f"{hour} giờ"
            if period:
                time_str += f" {period}"
            state.slots["depart_time.time"] = time_str

    # Date extraction from raw text.
    # Handles patterns like "từ 20 tháng 1 đến ngày 21 tháng 3",
    # "ngày 20 tháng 1", "20 tháng 3".
    # The JointBERT model often extracts just label words ("tháng", "ngày")
    # instead of the full date phrase, so we re-extract from raw text.
    _extract_dates_from_text(state, text_lower)


def _extract_dates_from_text(state: DialogueState, text: str) -> None:
    """Extract date info from raw Vietnamese text using regex.

    Fills depart_date and return_date sub-slots when the NLU model
    fails to extract the actual numbers (e.g., tags "tháng" instead
    of "tháng 1").

    Handles:
      - "từ 20 tháng 1 đến ngày 21 tháng 3" → depart=20/1, return=21/3
      - "ngày 20 tháng 1" → depart day=20, month=tháng 1
      - "20 tháng 3" → depart day=20, month=tháng 3
    """
    # Look for range pattern: "từ <date> đến <date>"
    range_pat = re.compile(
        r"từ\s+(?:ngày\s+)?(\d{1,2})(?:\s+tháng\s*(\d{1,2}))?"
        r"\s+đến\s+(?:ngày\s+)?(\d{1,2})(?:\s+tháng\s*(\d{1,2}))?",
        re.IGNORECASE,
    )
    m = range_pat.search(text)
    if m:
        d1, m1, d2, m2 = m.group(1), m.group(2), m.group(3), m.group(4)

        # Determine which is depart and which is return
        has_depart_day = state.slots.get("depart_date.day_number")
        has_return = state._has_date_info("return_date")
        is_round_trip = (state.slots.get("round_trip") or "").lower() in (
            "khứ hồi", "khu hoi", "round trip",
        )

        # First date → depart (if not already filled with a real value)
        if not has_depart_day or has_depart_day in ("ngày", ""):
            state.slots["depart_date.day_number"] = d1
            if m1:
                state.slots["depart_date.month_name"] = f"tháng {m1}"

        # Second date → return (if round-trip) or overwrite depart month
        if is_round_trip or state.slots.get("round_trip") is None:
            # Assume second date is return date
            state.slots["return_date.day_number"] = d2
            if m2:
                state.slots["return_date.month_name"] = f"tháng {m2}"
            elif m1:
                # Same month as departure
                state.slots["return_date.month_name"] = f"tháng {m1}"
        return

    # Single date pattern: "ngày 20 tháng 1" or "20 tháng 3"
    single_pat = re.compile(
        r"(?:ngày\s+)?(\d{1,2})\s+tháng\s*(\d{1,2})",
        re.IGNORECASE,
    )
    for sm in single_pat.finditer(text):
        day_val = sm.group(1)
        month_val = sm.group(2)

        # Fill depart date if empty or has bad NLU value
        cur_day = state.slots.get("depart_date.day_number")
        if not cur_day or cur_day in ("ngày", ""):
            state.slots["depart_date.day_number"] = day_val
            state.slots["depart_date.month_name"] = f"tháng {month_val}"
        else:
            # Already have depart, this might be return date
            cur_ret_day = state.slots.get("return_date.day_number")
            if not cur_ret_day or cur_ret_day in ("ngày", ""):
                state.slots["return_date.day_number"] = day_val
                state.slots["return_date.month_name"] = f"tháng {month_val}"
        break  # Only take the first match for single date


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
        - Normalizes date values to strip redundant prefixes.
        - Fills missing slots via keyword detection on raw transcript.
        - Increments turn counter.
        - Stores intent from latest turn.
        """
        state.intent = nlu_result.intent
        state.intent_confidence = nlu_result.intent_confidence
        state.turn_count += 1

        # When the system asked for return_date but NLU extracts depart_date.*,
        # remap those slots to return_date.* (NLU doesn't know about return dates)
        awaiting_return = (
            state.slots.get("round_trip")
            and not state._has_date_info("return_date")
            and state._has_date_info("depart_date")
        )

        for slot in nlu_result.slots:
            if slot.confidence >= _SLOT_CONFIDENCE_THRESHOLD:
                name = slot.name
                if awaiting_return and name.startswith("depart_date."):
                    name = name.replace("depart_date.", "return_date.", 1)
                if name in state.slots:
                    cleaned = _clean_slot_value(slot.value)
                    normalized = _normalize_date_value(name, cleaned)
                    # Only store if normalization produced a real value
                    # (rejects bare "tháng"/"ngày" without numbers)
                    if normalized is not None:
                        state.slots[name] = normalized

        # Keyword-based slot fill for slots NLU may miss
        raw_text = nlu_result.raw_text or ""
        _keyword_slot_fill(state, raw_text)

        # Record turn in history
        state.history.append({
            "turn": state.turn_count,
            "intent": nlu_result.intent,
            "slots": {s.name: s.value for s in nlu_result.slots},
        })

        return state
