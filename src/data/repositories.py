"""
Data Layer Repositories

Repository Pattern implementation for claim storage and retrieval.
Provides clean separation between data access and business logic.

See CHOICES.md A-0001 (4-Layer Architecture) and A-0002 (Repository Pattern).
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from src.core.models import Claim, ClaimState, ClaimType, DirtyReason

logger = logging.getLogger(__name__)


class ClaimRepository:
    """
    Repository for claim CRUD operations.

    Provides abstraction over the underlying storage mechanism,
    allowing the Process Layer to work with claims without
    knowing the storage details.
    """

    def __init__(self, storage: Optional[Any] = None):
        """
        Initialize the claim repository.

        Args:
            storage: Optional storage backend. If None, uses in-memory storage.
        """
        self._storage = storage
        self._claims: Dict[str, Claim] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the repository and underlying storage."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("ClaimRepository initialized")

    async def get_by_id(self, claim_id: str) -> Optional[Claim]:
        """
        Retrieve a claim by its ID.

        Args:
            claim_id: Unique identifier of the claim

        Returns:
            Claim if found, None otherwise
        """
        return self._claims.get(claim_id)

    async def create(self, claim: Claim) -> Claim:
        """
        Create a new claim.

        Args:
            claim: The claim to create

        Returns:
            The created claim

        Raises:
            ValueError: If claim with same ID already exists
        """
        if claim.id in self._claims:
            raise ValueError(f"Claim already exists: {claim.id}")

        self._claims[claim.id] = claim
        logger.debug(f"Created claim: {claim.id}")
        return claim

    async def update(self, claim: Claim) -> Claim:
        """
        Update an existing claim.

        Args:
            claim: The claim with updated data

        Returns:
            The updated claim

        Raises:
            ValueError: If claim does not exist
        """
        if claim.id not in self._claims:
            raise ValueError(f"Claim not found: {claim.id}")

        claim.updated = datetime.now(timezone.utc)
        self._claims[claim.id] = claim
        logger.debug(f"Updated claim: {claim.id}")
        return claim

    async def delete(self, claim_id: str) -> bool:
        """
        Delete a claim by ID.

        Args:
            claim_id: ID of claim to delete

        Returns:
            True if deleted, False if not found
        """
        if claim_id in self._claims:
            del self._claims[claim_id]
            logger.debug(f"Deleted claim: {claim_id}")
            return True
        return False

    async def get_dirty_claims(
        self,
        priority_min: int = 0,
        limit: int = 100
    ) -> List[Claim]:
        """
        Get claims that need re-evaluation.

        Args:
            priority_min: Minimum dirty priority
            limit: Maximum number of claims to return

        Returns:
            List of dirty claims sorted by priority
        """
        dirty = [
            c for c in self._claims.values()
            if c.is_dirty and c.dirty_priority >= priority_min
        ]
        # Sort by priority (descending) then timestamp
        dirty.sort(
            key=lambda x: (-x.dirty_priority, x.dirty_timestamp or datetime.min)
        )
        return dirty[:limit]

    async def query_by_state(
        self,
        state: ClaimState,
        limit: int = 100
    ) -> List[Claim]:
        """
        Query claims by state.

        Args:
            state: The claim state to filter by
            limit: Maximum number of claims to return

        Returns:
            List of claims with the given state
        """
        results = [c for c in self._claims.values() if c.state == state]
        return results[:limit]

    async def query_by_tag(
        self,
        tag: str,
        limit: int = 100
    ) -> List[Claim]:
        """
        Query claims by tag.

        Args:
            tag: The tag to filter by
            limit: Maximum number of claims to return

        Returns:
            List of claims with the given tag
        """
        tag_lower = tag.lower()
        results = [
            c for c in self._claims.values()
            if tag_lower in [t.lower() for t in c.tags]
        ]
        return results[:limit]

    async def get_related(self, claim_id: str) -> Dict[str, List[Claim]]:
        """
        Get claims related to the given claim.

        Args:
            claim_id: ID of the claim to find relations for

        Returns:
            Dict with 'supers' and 'subs' claim lists
        """
        claim = await self.get_by_id(claim_id)
        if not claim:
            return {"supers": [], "subs": []}

        supers = []
        for super_id in claim.supers:
            super_claim = await self.get_by_id(super_id)
            if super_claim:
                supers.append(super_claim)

        subs = []
        for sub_id in claim.subs:
            sub_claim = await self.get_by_id(sub_id)
            if sub_claim:
                subs.append(sub_claim)

        return {"supers": supers, "subs": subs}

    async def mark_dirty(
        self,
        claim_id: str,
        reason: DirtyReason,
        priority: int = 0
    ) -> Optional[Claim]:
        """
        Mark a claim as dirty (needs re-evaluation).

        Args:
            claim_id: ID of claim to mark dirty
            reason: Reason for marking dirty
            priority: Evaluation priority

        Returns:
            Updated claim or None if not found
        """
        claim = await self.get_by_id(claim_id)
        if not claim:
            return None

        claim.is_dirty = True
        claim.dirty = True
        claim.dirty_reason = reason
        claim.dirty_priority = priority
        claim.dirty_timestamp = datetime.now(timezone.utc)

        return await self.update(claim)

    async def mark_clean(self, claim_id: str) -> Optional[Claim]:
        """
        Mark a claim as clean (evaluated).

        Args:
            claim_id: ID of claim to mark clean

        Returns:
            Updated claim or None if not found
        """
        claim = await self.get_by_id(claim_id)
        if not claim:
            return None

        claim.is_dirty = False
        claim.dirty = False
        claim.dirty_reason = None
        claim.dirty_priority = 0
        claim.dirty_timestamp = None

        return await self.update(claim)

    async def count(self) -> int:
        """Return the total number of claims."""
        return len(self._claims)

    async def list_all(self, limit: int = 1000) -> List[Claim]:
        """
        List all claims.

        Args:
            limit: Maximum number of claims to return

        Returns:
            List of all claims
        """
        return list(self._claims.values())[:limit]


class RepositoryFactory:
    """
    Factory for creating repository instances.

    Provides centralized creation and caching of repositories.
    """

    _claim_repo: Optional[ClaimRepository] = None
    _data_manager: Optional[Any] = None

    @classmethod
    def get_claim_repository(cls) -> ClaimRepository:
        """Get or create the claim repository singleton."""
        if cls._claim_repo is None:
            cls._claim_repo = ClaimRepository()
        return cls._claim_repo

    @classmethod
    def reset(cls) -> None:
        """Reset all cached repositories (useful for testing)."""
        cls._claim_repo = None
        cls._data_manager = None


def get_data_manager(use_cache: bool = True) -> Any:
    """
    Get the data manager instance.

    Args:
        use_cache: Whether to use cached instance

    Returns:
        Data manager instance
    """
    if use_cache and RepositoryFactory._data_manager is not None:
        return RepositoryFactory._data_manager

    # Return repository factory as data manager for now
    repo = RepositoryFactory.get_claim_repository()
    if use_cache:
        RepositoryFactory._data_manager = repo
    return repo
