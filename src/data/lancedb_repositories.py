"""
LanceDB-based Repository Implementation
Refactored to use unified LanceDB adapter instead of SQLite + ChromaDB
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

# Local imports
from src.data.models import (
    Claim,
    ClaimState,
    ClaimType,
    ClaimFilter,
    Relationship,
    DataLayerError,
    ClaimNotFoundError,
    InvalidClaimError,
    RelationshipError,
    ProcessingResult,
    ProcessingStats,
)
from src.data.lancedb_adapter import LanceDBAdapter, create_lancedb_adapter

logger = logging.getLogger(__name__)


class LanceDBClaimRepository:
    """
    LanceDB-based repository for claim operations.

    This repository provides a clean interface for claim CRUD operations,
    search functionality, and relationship management using LanceDB.
    """

    def __init__(self, adapter: LanceDBAdapter):
        """
        Initialize the repository with a LanceDB adapter.

        Args:
            adapter: Initialized LanceDBAdapter instance
        """
        self.adapter = adapter
        self._stats = ProcessingStats(operation="claim_repository")
        self._stats.start()

    async def create_claim(self, claim: Claim) -> ProcessingResult:
        """
        Create a new claim.

        Args:
            claim: Claim object to create

        Returns:
            ProcessingResult with operation status and metadata
        """
        try:
            # Validate claim
            if not claim.content or len(claim.content.strip()) < 10:
                raise InvalidClaimError("Claim content must be at least 10 characters")

            if not (0.0 <= claim.confidence <= 1.0):
                raise InvalidClaimError("Confidence must be between 0.0 and 1.0")

            # Create claim
            created_claim = await self.adapter.create_claim(claim)

            self._stats.add_success()

            return ProcessingResult(
                claim_id=created_claim.id,
                success=True,
                message="Claim created successfully",
                updated_confidence=created_claim.confidence,
                processing_time=self._stats.duration,
                metadata={"operation": "create", "state": created_claim.state.value},
            )

        except Exception as e:
            self._stats.add_failure()
            logger.error(f"Failed to create claim: {e}")

            return ProcessingResult(
                claim_id=claim.id,
                success=False,
                message=f"Failed to create claim: {str(e)}",
                processing_time=self._stats.duration,
            )

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """
        Get a claim by ID.

        Args:
            claim_id: Unique identifier for the claim

        Returns:
            Claim object or None if not found
        """
        try:
            return await self.adapter.get_claim(claim_id)
        except Exception as e:
            logger.error(f"Failed to get claim {claim_id}: {e}")
            return None

    async def update_claim(
        self, claim_id: str, updates: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Update an existing claim.

        Args:
            claim_id: ID of claim to update
            updates: Dictionary of fields to update

        Returns:
            ProcessingResult with operation status
        """
        try:
            # Validate updates
            valid_fields = {
                "content",
                "confidence",
                "state",
                "type",
                "tags",
                "scope",
                "is_dirty",
                "dirty_reason",
                "dirty_priority",
                "dirty_timestamp",
                "supported_by",
                "supports",
                "embedding",
            }

            invalid_fields = set(updates.keys()) - valid_fields
            if invalid_fields:
                raise InvalidClaimError(f"Invalid fields for update: {invalid_fields}")

            # Validate confidence if provided
            if "confidence" in updates and not (0.0 <= updates["confidence"] <= 1.0):
                raise InvalidClaimError("Confidence must be between 0.0 and 1.0")

            # Perform update
            updated_claim = await self.adapter.update_claim(claim_id, updates)

            self._stats.add_success()

            return ProcessingResult(
                claim_id=updated_claim.id,
                success=True,
                message="Claim updated successfully",
                updated_confidence=updated_claim.confidence,
                processing_time=self._stats.duration,
                metadata={
                    "operation": "update",
                    "updated_fields": list(updates.keys()),
                },
            )

        except ClaimNotFoundError:
            self._stats.add_failure()
            return ProcessingResult(
                claim_id=claim_id,
                success=False,
                message=f"Claim {claim_id} not found",
                processing_time=self._stats.duration,
            )
        except Exception as e:
            self._stats.add_failure()
            logger.error(f"Failed to update claim {claim_id}: {e}")

            return ProcessingResult(
                claim_id=claim_id,
                success=False,
                message=f"Failed to update claim: {str(e)}",
                processing_time=self._stats.duration,
            )

    async def delete_claim(self, claim_id: str) -> ProcessingResult:
        """
        Delete a claim.

        Args:
            claim_id: ID of claim to delete

        Returns:
            ProcessingResult with operation status
        """
        try:
            # Check if claim exists
            existing = await self.get_claim(claim_id)
            if not existing:
                raise ClaimNotFoundError(f"Claim {claim_id} not found")

            # Delete claim
            success = await self.adapter.delete_claim(claim_id)

            if success:
                self._stats.add_success()
                return ProcessingResult(
                    claim_id=claim_id,
                    success=True,
                    message="Claim deleted successfully",
                    processing_time=self._stats.duration,
                )
            else:
                self._stats.add_failure()
                return ProcessingResult(
                    claim_id=claim_id,
                    success=False,
                    message="Delete operation not supported",
                    processing_time=self._stats.duration,
                )

        except Exception as e:
            self._stats.add_failure()
            logger.error(f"Failed to delete claim {claim_id}: {e}")

            return ProcessingResult(
                claim_id=claim_id,
                success=False,
                message=f"Failed to delete claim: {str(e)}",
                processing_time=self._stats.duration,
            )

    async def search_claims(self, query: str, limit: int = 10) -> List[Claim]:
        """
        Search claims by content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching claims
        """
        try:
            return await self.adapter.search_claims(query, limit)
        except Exception as e:
            logger.error(f"Failed to search claims: {e}")
            return []

    async def list_claims(
        self, filter_obj: Optional[ClaimFilter] = None
    ) -> List[Claim]:
        """
        List claims with optional filtering.

        Args:
            filter_obj: Optional filter criteria

        Returns:
            List of claims matching criteria
        """
        try:
            return await self.adapter.list_claims(filter_obj)
        except Exception as e:
            logger.error(f"Failed to list claims: {e}")
            return []

    async def vector_search(
        self, query_vector: List[float], limit: int = 10, confidence_min: float = 0.0
    ) -> List[Tuple[Claim, float]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            confidence_min: Minimum confidence threshold

        Returns:
            List of (claim, similarity_score) tuples
        """
        try:
            return await self.adapter.vector_search(query_vector, limit, confidence_min)
        except Exception as e:
            logger.error(f"Failed to perform vector search: {e}")
            return []

    async def get_claims_by_state(
        self, state: ClaimState, limit: int = 100
    ) -> List[Claim]:
        """
        Get claims by state.

        Args:
            state: Claim state to filter by
            limit: Maximum number of results

        Returns:
            List of claims with the specified state
        """
        try:
            filter_obj = ClaimFilter(states=[state], limit=limit)
            return await self.adapter.list_claims(filter_obj)
        except Exception as e:
            logger.error(f"Failed to get claims by state {state}: {e}")
            return []

    async def get_dirty_claims(
        self, priority_min: int = 0, limit: int = 100
    ) -> List[Claim]:
        """
        Get claims that need re-evaluation.

        Args:
            priority_min: Minimum dirty priority
            limit: Maximum number of results

        Returns:
            List of dirty claims
        """
        try:
            # Get all dirty claims
            filter_obj = ClaimFilter(dirty_only=True, limit=limit)
            all_dirty = await self.adapter.list_claims(filter_obj)

            # Filter by priority
            high_priority = [
                claim for claim in all_dirty if claim.dirty_priority >= priority_min
            ]

            # Sort by priority (descending) and timestamp
            high_priority.sort(
                key=lambda x: (-x.dirty_priority, x.dirty_timestamp or datetime.min)
            )

            return high_priority

        except Exception as e:
            logger.error(f"Failed to get dirty claims: {e}")
            return []

    async def mark_claim_dirty(
        self, claim_id: str, reason: str, priority: int = 0
    ) -> ProcessingResult:
        """
        Mark a claim as needing re-evaluation.

        Args:
            claim_id: ID of claim to mark dirty
            reason: Reason for marking dirty
            priority: Priority for re-evaluation (higher = more urgent)

        Returns:
            ProcessingResult with operation status
        """
        try:
            updates = {
                "is_dirty": True,
                "dirty_reason": reason,
                "dirty_priority": priority,
                "dirty_timestamp": datetime.utcnow(),
            }

            return await self.update_claim(claim_id, updates)

        except Exception as e:
            logger.error(f"Failed to mark claim {claim_id} dirty: {e}")

            return ProcessingResult(
                claim_id=claim_id,
                success=False,
                message=f"Failed to mark claim dirty: {str(e)}",
            )

    async def mark_claim_clean(self, claim_id: str) -> ProcessingResult:
        """
        Mark a claim as clean (no longer needing re-evaluation).

        Args:
            claim_id: ID of claim to mark clean

        Returns:
            ProcessingResult with operation status
        """
        try:
            updates = {
                "is_dirty": False,
                "dirty_reason": None,
                "dirty_priority": 0,
                "dirty_timestamp": None,
            }

            result = await self.update_claim(claim_id, updates)
            if result.success:
                result.message = "Claim marked clean"
            return result

        except Exception as e:
            logger.error(f"Failed to mark claim {claim_id} clean: {e}")

            return ProcessingResult(
                claim_id=claim_id,
                success=False,
                message=f"Failed to mark claim clean: {str(e)}",
            )

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get repository statistics.

        Returns:
            Dictionary with repository statistics
        """
        try:
            stats = await self.adapter.get_stats()
            stats.update(
                {
                    "repository_operation": "claim_repository",
                    "success_rate": self._stats.success_rate,
                    "items_processed": self._stats.items_processed,
                    "items_succeeded": self._stats.items_succeeded,
                    "items_failed": self._stats.items_failed,
                }
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            return {"error": str(e)}


class LanceDBRelationshipRepository:
    """
    LanceDB-based repository for relationship operations.
    """

    def __init__(self, adapter: LanceDBAdapter):
        """
        Initialize the repository with a LanceDB adapter.

        Args:
            adapter: Initialized LanceDBAdapter instance
        """
        self.adapter = adapter
        self._stats = ProcessingStats(operation="relationship_repository")
        self._stats.start()

    async def create_relationship(
        self, supporter_id: str, supported_id: str
    ) -> ProcessingResult:
        """
        Create a relationship between two claims.

        Args:
            supporter_id: ID of supporting claim
            supported_id: ID of supported claim

        Returns:
            ProcessingResult with operation status
        """
        try:
            # Validate claims exist
            supporter = await self.adapter.get_claim(supporter_id)
            if not supporter:
                raise ClaimNotFoundError(f"Supporter claim {supporter_id} not found")

            supported = await self.adapter.get_claim(supported_id)
            if not supported:
                raise ClaimNotFoundError(f"Supported claim {supported_id} not found")

            # Create relationship
            relationship = Relationship(
                supporter_id=supporter_id, supported_id=supported_id
            )

            created_rel = await self.adapter.create_relationship(relationship)

            self._stats.add_success()

            return ProcessingResult(
                claim_id=f"rel_{supporter_id}_{supported_id}",
                success=True,
                message="Relationship created successfully",
                processing_time=self._stats.duration,
                metadata={"operation": "create_relationship"},
            )

        except Exception as e:
            self._stats.add_failure()
            logger.error(f"Failed to create relationship: {e}")

            return ProcessingResult(
                claim_id=f"rel_{supporter_id}_{supported_id}",
                success=False,
                message=f"Failed to create relationship: {str(e)}",
                processing_time=self._stats.duration,
            )

    async def get_relationships(self, claim_id: str) -> List[Relationship]:
        """
        Get all relationships for a claim.

        Args:
            claim_id: ID of claim

        Returns:
            List of relationships
        """
        try:
            return await self.adapter.get_relationships(claim_id)
        except Exception as e:
            logger.error(f"Failed to get relationships for {claim_id}: {e}")
            return []

    async def get_supporting_claims(self, claim_id: str) -> List[Claim]:
        """
        Get all claims that support the given claim.

        Args:
            claim_id: ID of claim to get supporters for

        Returns:
            List of supporting claims
        """
        try:
            relationships = await self.get_relationships(claim_id)
            supporting_ids = [
                rel.supporter_id
                for rel in relationships
                if rel.supported_id == claim_id
            ]

            supporting_claims = []
            for sid in supporting_ids:
                claim = await self.adapter.get_claim(sid)
                if claim:
                    supporting_claims.append(claim)

            return supporting_claims
        except Exception as e:
            logger.error(f"Failed to get supporting claims for {claim_id}: {e}")
            return []

    async def get_supported_claims(self, claim_id: str) -> List[Claim]:
        """
        Get all claims that are supported by the given claim.

        Args:
            claim_id: ID of claim to get supported claims for

        Returns:
            List of supported claims
        """
        try:
            relationships = await self.get_relationships(claim_id)
            supported_ids = [
                rel.supported_id
                for rel in relationships
                if rel.supporter_id == claim_id
            ]

            supported_claims = []
            for sid in supported_ids:
                claim = await self.adapter.get_claim(sid)
                if claim:
                    supported_claims.append(claim)

            return supported_claims
        except Exception as e:
            logger.error(f"Failed to get supported claims for {claim_id}: {e}")
            return []


# Factory function
async def create_lancedb_repositories(
    db_path: str = "data/conjecture.lance", dimension: int = 384
) -> Tuple[LanceDBClaimRepository, LanceDBRelationshipRepository]:
    """
    Create and initialize LanceDB repositories.

    Args:
        db_path: Path to LanceDB database
        dimension: Embedding dimension

    Returns:
        Tuple of (claim_repository, relationship_repository)
    """
    # Create adapter
    adapter = await create_lancedb_adapter(db_path, dimension)

    # Create repositories
    claim_repo = LanceDBClaimRepository(adapter)
    relationship_repo = LanceDBRelationshipRepository(adapter)

    return claim_repo, relationship_repo
