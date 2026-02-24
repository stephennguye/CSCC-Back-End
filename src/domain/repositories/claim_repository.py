"""ClaimRepository — domain interface (Protocol).

Zero framework imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.claim import Claim


@runtime_checkable
class ClaimRepository(Protocol):
    """Persistence interface for *Claim* records."""

    async def create(self, claim: Claim) -> Claim:
        """Persist a new *Claim* and return it.

        Raises:
            PersistenceError: if a Claim for *claim.session_id* already exists
                              and upsert semantics are not supported by the
                              implementation.
        """
        ...  # pragma: no cover

    async def upsert(self, claim: Claim) -> Claim:
        """Insert or update the *Claim* for ``claim.session_id``.

        Idempotent — safe to call multiple times (FR-014 determinism guarantee).
        """
        ...  # pragma: no cover

    async def get_by_session_id(self, session_id: uuid.UUID) -> Claim | None:
        """Return the *Claim* for *session_id*, or *None* when not yet extracted."""
        ...  # pragma: no cover
