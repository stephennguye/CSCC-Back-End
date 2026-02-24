from enum import StrEnum


class SessionState(StrEnum):
    active = "active"
    ended = "ended"
    error = "error"


class UrgencyLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"
