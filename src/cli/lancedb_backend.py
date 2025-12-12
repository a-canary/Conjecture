"""
LanceDB Backend for CLI
Provides CLI operations using the new LanceDB-based data layer
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Local imports
from src.cli.base_cli import BaseCLI, DatabaseError, BackendNotAvailableError
from src.core.models import Claim, ClaimFilter, ClaimState, ClaimType, ClaimScope
from src.data.lancedb_integration import LanceDBDataManager, create_lancedb_data_manager
from src.data.models import DataConfig

logger = logging.getLogger(__name__)

class LanceDBBackend(BaseCLI):
    """
    LanceDB-based CLI backend.

    This backend uses the new LanceDB-based data layer to provide
    CLI operations. It replaces the old SQLite + ChromaDB implementation
    with a unified LanceDB solution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LanceDB backend.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.data_manager: Optional[LanceDBDataManager] = None
        self._initialized = False

        # Extract configuration
        db_path = config.get('db_path') if config else None
        self.db_path = db_path or "data/conjecture.lance"
        self.dimension = config.get('embedding_dimension', 384) if config else 384

    async def initialize(self) -> None:
        """Initialize the backend and data manager."""
        if self._initialized:
            return

        try:
            # Create data manager
            self.data_manager = await create_lancedb_data_manager(
                db_path=self.db_path,
                dimension=self.dimension
            )
            self._initialized = True
            logger.info("LanceDB backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB backend: {e}")
            raise BackendNotAvailableError(f"LanceDB backend unavailable: {e}")

    def is_available(self) -> bool:
        """Check if the backend is available."""
        try:
            import lancedb
            return True
        except ImportError:
            return False

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.initialize()

    # Claim CRUD operations
    async def create_claim(
        self,
        content: str,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Claim:
        """
        Create a new claim.

        Args:
            content: Claim content
            confidence: Confidence score (0.0-1.0)
            tags: Optional list of tags
            **kwargs: Additional claim parameters

        Returns:
            Created claim object

        Raises:
            DatabaseError: If claim creation fails
        """
        await self._ensure_initialized()

        try:
            # Validate parameters
            if not content or len(content.strip()) < 10:
                raise DatabaseError("Claim content must be at least 10 characters")

            if not (0.0 <= confidence <= 1.0):
                raise DatabaseError("Confidence must be between 0.0 and 1.0")

            # Extract additional parameters
            state = kwargs.get('state')
            claim_type = kwargs.get('type', ['concept'])
            scope = kwargs.get('scope', 'user_workspace')

            # Convert to proper enums
            claim_state = ClaimState.EXPLORE
            if state:
                try:
                    claim_state = ClaimState(state)
                except ValueError:
                    logger.warning(f"Invalid state '{state}', using EXPLORE")

            claim_types = []
            if isinstance(claim_type, str):
                claim_types = [ClaimType(claim_type)]
            elif isinstance(claim_type, list):
                claim_types = [ClaimType(t) for t in claim_type]
            else:
                claim_types = [ClaimType.CONCEPT]

            claim_scope = ClaimScope.USER_WORKSPACE
            try:
                claim_scope = ClaimScope(scope)
            except ValueError:
                logger.warning(f"Invalid scope '{scope}', using USER_WORKSPACE")

            # Create claim using data manager
            claim = await self.data_manager.create_claim(
                content=content,
                confidence=confidence,
                tags=tags or [],
                state=claim_state.value,
                type=[t.value for t in claim_types],
                scope=claim_scope.value,
                **kwargs
            )

            return claim

        except Exception as e:
            logger.error(f"Failed to create claim: {e}")
            raise DatabaseError(f"Failed to create claim: {e}")

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """
        Retrieve a claim by ID.

        Args:
            claim_id: Unique identifier for the claim

        Returns:
            Claim object or None if not found

        Raises:
            DatabaseError: If retrieval fails
        """
        await self._ensure_initialized()

        try:
            return await self.data_manager.get_claim(claim_id)
        except Exception as e:
            logger.error(f"Failed to get claim {claim_id}: {e}")
            raise DatabaseError(f"Failed to get claim: {e}")

    async def update_claim(
        self,
        claim_id: str,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        state: Optional[str] = None,
        **kwargs
    ) -> Claim:
        """
        Update an existing claim.

        Args:
            claim_id: ID of claim to update
            content: New content (optional)
            confidence: New confidence (optional)
            tags: New tags (optional)
            state: New state (optional)
            **kwargs: Additional update parameters

        Returns:
            Updated claim object

        Raises:
            DatabaseError: If update fails
        """
        await self._ensure_initialized()

        try:
            # Build updates dictionary
            updates = {}

            if content is not None:
                if len(content.strip()) < 10:
                    raise DatabaseError("Claim content must be at least 10 characters")
                updates['content'] = content

            if confidence is not None:
                if not (0.0 <= confidence <= 1.0):
                    raise DatabaseError("Confidence must be between 0.0 and 1.0")
                updates['confidence'] = confidence

            if tags is not None:
                updates['tags'] = tags

            if state is not None:
                try:
                    updates['state'] = ClaimState(state)
                except ValueError:
                    raise DatabaseError(f"Invalid state: {state}")

            # Add additional kwargs
            for key, value in kwargs.items():
                if value is not None:
                    updates[key] = value

            if not updates:
                raise DatabaseError("No updates specified")

            # Perform update
            return await self.data_manager.update_claim(claim_id, updates)

        except Exception as e:
            logger.error(f"Failed to update claim {claim_id}: {e}")
            raise DatabaseError(f"Failed to update claim: {e}")

    async def delete_claim(self, claim_id: str) -> bool:
        """
        Delete a claim.

        Args:
            claim_id: ID of claim to delete

        Returns:
            True if successful, False otherwise

        Raises:
            DatabaseError: If deletion fails
        """
        await self._ensure_initialized()

        try:
            return await self.data_manager.delete_claim(claim_id)
        except Exception as e:
            logger.error(f"Failed to delete claim {claim_id}: {e}")
            raise DatabaseError(f"Failed to delete claim: {e}")

    # Search and listing operations
    async def search_claims(self, query: str, limit: int = 10) -> List[Claim]:
        """
        Search claims by content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching claims

        Raises:
            DatabaseError: If search fails
        """
        await self._ensure_initialized()

        try:
            return await self.data_manager.search_claims(query, limit)
        except Exception as e:
            logger.error(f"Failed to search claims: {e}")
            raise DatabaseError(f"Failed to search claims: {e}")

    async def list_claims(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None
    ) -> List[Claim]:
        """
        List claims with optional filtering.

        Args:
            limit: Maximum number of results
            offset: Result offset
            state: Filter by state
            tags: Filter by tags (any match)
            confidence_min: Minimum confidence
            confidence_max: Maximum confidence

        Returns:
            List of claims

        Raises:
            DatabaseError: If listing fails
        """
        await self._ensure_initialized()

        try:
            # Build filter
            filter_obj = None

            if any([state, tags, confidence_min is not None, confidence_max is not None]):
                filter_states = [ClaimState(state)] if state else None
                filter_obj = ClaimFilter(
                    limit=limit,
                    offset=offset,
                    states=filter_states,
                    tags=tags,
                    confidence_min=confidence_min,
                    confidence_max=confidence_max
                )

            return await self.data_manager.list_claims(filter_obj)
        except Exception as e:
            logger.error(f"Failed to list claims: {e}")
            raise DatabaseError(f"Failed to list claims: {e}")

    # Relationship operations
    async def create_relationship(self, supporter_id: str, supported_id: str) -> bool:
        """
        Create a relationship between two claims.

        Args:
            supporter_id: ID of supporting claim
            supported_id: ID of supported claim

        Returns:
            True if successful

        Raises:
            DatabaseError: If relationship creation fails
        """
        await self._ensure_initialized()

        try:
            # Verify claims exist
            supporter = await self.get_claim(supporter_id)
            if not supporter:
                raise DatabaseError(f"Supporter claim not found: {supporter_id}")

            supported = await self.get_claim(supported_id)
            if not supported:
                raise DatabaseError(f"Supported claim not found: {supported_id}")

            return await self.data_manager.create_relationship(supporter_id, supported_id)
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise DatabaseError(f"Failed to create relationship: {e}")

    async def get_relationships(self, claim_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a claim.

        Args:
            claim_id: ID of claim

        Returns:
            List of relationship dictionaries

        Raises:
            DatabaseError: If retrieval fails
        """
        await self._ensure_initialized()

        try:
            relationships = await self.data_manager.get_relationships(claim_id)
            return [
                {
                    "supporter_id": rel.supporter_id,
                    "supported_id": rel.supported_id,
                    "created": rel.created.isoformat()
                }
                for rel in relationships
            ]
        except Exception as e:
            logger.error(f"Failed to get relationships for {claim_id}: {e}")
            raise DatabaseError(f"Failed to get relationships: {e}")

    # Statistics and health
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics

        Raises:
            DatabaseError: If stats retrieval fails
        """
        await self._ensure_initialized()

        try:
            stats = await self.data_manager.get_stats()

            # Add backend-specific info
            stats.update({
                "backend": "lancedb",
                "db_path": self.db_path,
                "embedding_dimension": self.dimension
            })

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise DatabaseError(f"Failed to get stats: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the backend.

        Returns:
            Dictionary with health status

        Raises:
            DatabaseError: If health check fails
        """
        await self._ensure_initialized()

        try:
            health = await self.data_manager.health_check()

            # Add backend info
            health.update({
                "backend": "lancedb",
                "initialized": self._initialized,
                "db_path": self.db_path
            })

            return health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise DatabaseError(f"Health check failed: {e}")

    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        if self.data_manager:
            await self.data_manager.close()

        self.data_manager = None
        self._initialized = False
        logger.info("LanceDB backend closed")

# Factory function
async def create_lancedb_backend(config: Optional[Dict[str, Any]] = None) -> LanceDBBackend:
    """
    Create and initialize a LanceDB backend.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized LanceDBBackend
    """
    backend = LanceDBBackend(config)
    await backend.initialize()
    return backend