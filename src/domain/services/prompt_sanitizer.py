"""PromptSanitizer — domain service for input sanitization.

Guards against prompt injection attacks by:
1. Enforcing a maximum input length.
2. Detecting instruction-override patterns (role-switching, jailbreak attempts).
3. Enforcing role-separation (system-prompt markers in user content).

Zero framework imports — pure Python only.
"""

from __future__ import annotations

import re

from src.domain.errors import PromptInjectionDetectedError

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_MAX_LENGTH = 4096  # characters

# Patterns that signal an instruction-override attempt.
# Each pattern is compiled with IGNORECASE | DOTALL.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?)",
        re.I,
    ),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+", re.I),
    re.compile(r"(system|assistant|<\|im_start\|>|<\|im_end\|>)\s*:", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(
        r"forget\s+(your\s+)?(previous\s+)?(training|instructions?|rules?|constraints?)",
        re.I,
    ),
]

# Role-separation markers that must not appear in user-supplied content.
_ROLE_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"^(system|user|assistant)\s*:", re.I | re.MULTILINE),
]


class PromptSanitizer:
    """Stateless domain service.  Call ``sanitize()`` before forwarding user
    input to the LLM pipeline.
    """

    def __init__(self, max_length: int = _DEFAULT_MAX_LENGTH) -> None:
        self._max_length = max_length

    def sanitize(self, text: str) -> str:
        """Return *text* unchanged if it passes all guards.

        Raises:
            PromptInjectionDetectedError: if any guard fires.
        """
        self._check_length(text)
        self._check_injection_patterns(text)
        self._check_role_separation(text)
        return text

    def _check_length(self, text: str) -> None:
        if len(text) > self._max_length:
            raise PromptInjectionDetectedError(
                f"Input exceeds maximum allowed length of {self._max_length} characters "
                f"(got {len(text)})"
            )

    def _check_injection_patterns(self, text: str) -> None:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                raise PromptInjectionDetectedError(
                    f"Instruction-override pattern detected: {pattern.pattern!r}"
                )

    def _check_role_separation(self, text: str) -> None:
        for pattern in _ROLE_MARKERS:
            if pattern.search(text):
                raise PromptInjectionDetectedError(
                    f"Role-separation violation detected: {pattern.pattern!r}"
                )
