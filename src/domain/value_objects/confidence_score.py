from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceScore:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"ConfidenceScore.value must be between 0.0 and 1.0, got {self.value!r}"
            )

    def is_below_threshold(self, threshold: float) -> bool:
        return self.value < threshold
