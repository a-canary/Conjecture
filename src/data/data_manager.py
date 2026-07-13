# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Data Manager - Bridge to OptimizedSQLiteManager

Provides claim persistence and session management for the Conjecture system.
Bridges the high-level DataManager API to the underlying SQLite storage.

See CHOICES.md D-0001 (Universal Claim Storage) and I-0003 (SQLite Primary).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from src.core.models import Claim, ClaimType, ClaimScope, generate_claim_id
from src.data.optimized_sqlite_manager import OptimizedSQLiteManager

logger = logging.getLogger(__name__)


class DataManager:
    """
    Data manager providing claim persistence and session management.

    Bridges high-level operations to OptimizedSQLiteManager for SQLite storage.
    """

    def __init__(self, db_path: str = "data/conjecture.db"):
        """
        Initialize the data manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._storage = OptimizedSQLiteManager(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        if not self._initialized:
            await self._storage.initialize()
            self._initialized = True
            logger.info(f"DataManager initialized: {self.db_path}")

    async def close(self) -> None:
        """Close the storage backend."""
        if self._initialized:
            await self._storage.close()
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure storage is initialized before operations."""
        if not self._initialized:
            await self.initialize()

    async def create_claim(
        self,
        content: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        claim_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Create a new claim.

        Args:
            content: Claim content
            confidence: Confidence score (0.0 to 1.0)
            tags: Optional list of tags
            claim_id: Optional claim ID (generated if not provided)
            **kwargs: Additional claim fields

        Returns:
            Claim ID
        """
        await self._ensure_initialized()

        # Generate ID if not provided (use canonical format for cross-module consistency)
        if claim_id is None:
            claim_id = generate_claim_id()

        # Build claim object
        claim = Claim(
            id=claim_id,
            content=content,
            confidence=confidence,
            tags=tags or [],
            type=kwargs.get("type", [ClaimType.CONCEPT]),
            scope=kwargs.get("scope", ClaimScope.USER_WORKSPACE),
            supers=kwargs.get("supers", []),
            subs=kwargs.get("subs", []),
            is_dirty=kwargs.get("is_dirty", True),
        )

        return await self._storage.create_claim(claim)

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """
        Retrieve a claim by ID.

        Args:
            claim_id: The claim ID

        Returns:
            Claim if found, None otherwise
        """
        await self._ensure_initialized()
        return await self._storage.get_claim(claim_id)

    async def update_claim(self, claim_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing claim.

        Args:
            claim_id: The claim ID to update
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        await self._ensure_initialized()
        return await self._storage.update_claim(claim_id, updates)

    async def delete_claim(self, claim_id: str) -> bool:
        """
        Delete a claim.

        Args:
            claim_id: The claim ID to delete

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()
        return await self._storage.delete_claim(claim_id)

    async def search_claims(self, query: str, limit: int = 10) -> List[Claim]:
        """
        Search claims by content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching claims
        """
        await self._ensure_initialized()
        return await self._storage.search_claims(query, limit)

    async def list_claims(
        self, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Claim]:
        """
        List claims with optional filtering.

        Args:
            filter_dict: Optional filters (state, is_dirty, etc.)

        Returns:
            List of claims
        """
        await self._ensure_initialized()
        return await self._storage.list_claims(filter_dict)

    async def batch_create_claims(self, claims: List[Claim]) -> List[str]:
        """
        Create multiple claims in a batch.

        Args:
            claims: List of claims to create

        Returns:
            List of created claim IDs
        """
        await self._ensure_initialized()
        return await self._storage.batch_create_claims(claims)

    async def batch_update_claims(
        self, updates: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Update multiple claims in a batch.

        Args:
            updates: Dict mapping claim_id to updates dict

        Returns:
            Number of claims updated
        """
        await self._ensure_initialized()
        return await self._storage.batch_update_claims(updates)

    async def get_dirty_claims(self) -> List[Claim]:
        """
        Get claims that need re-evaluation.

        Returns:
            List of dirty claims sorted by priority
        """
        await self._ensure_initialized()
        return await self._storage.get_dirty_claims()

    async def count(self) -> int:
        """
        Return the total number of claims.

        Returns:
            Claim count
        """
        await self._ensure_initialized()
        return await self._storage.count()


class RepositoryFactory:
    """
    Factory for creating repository instances.

    Provides centralized creation of repositories with shared configuration.
    """

    _instance: Optional["RepositoryFactory"] = None
    _data_manager: Optional[DataManager] = None

    def __init__(self, db_path: str = "data/conjecture.db"):
        """
        Initialize the factory.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path

    @classmethod
    def get_instance(cls, db_path: str = "data/conjecture.db") -> "RepositoryFactory":
        """Get or create singleton factory instance."""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    @classmethod
    def get_data_manager(cls, db_path: str = "data/conjecture.db") -> DataManager:
        """Get or create shared DataManager instance."""
        if cls._data_manager is None:
            cls._data_manager = DataManager(db_path)
        return cls._data_manager

    @staticmethod
    def get_claim_repository(db_path: str = "data/conjecture.db") -> DataManager:
        """
        Get claim repository instance.

        Returns:
            DataManager configured for claim operations
        """
        return RepositoryFactory.get_data_manager(db_path)

    @staticmethod
    def get_session_repository(db_path: str = "data/conjecture.db") -> DataManager:
        """
        Get session repository instance.

        Note: Session management uses the same DataManager for now.
        Future: Separate SessionRepository with scope-aware operations.

        Returns:
            DataManager for session operations
        """
        return RepositoryFactory.get_data_manager(db_path)
