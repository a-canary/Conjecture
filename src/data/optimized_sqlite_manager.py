# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Optimized SQLite Manager

SQLite-based claim storage with async support for the Data Layer.
Provides persistence, CRUD operations, and query capabilities.

See CHOICES.md D-0001 (Universal Claim Storage) and I-0003 (SQLite Primary).
"""

import aiosqlite
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope, DirtyReason

logger = logging.getLogger(__name__)


class OptimizedSQLiteManager:
    """
    SQLite-based manager for claim persistence.

    Provides async CRUD operations, batch processing, and query capabilities
    for the Conjecture claim system.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS claims (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        confidence REAL NOT NULL,
        state TEXT NOT NULL DEFAULT 'Explore',
        type TEXT NOT NULL DEFAULT '["concept"]',
        tags TEXT NOT NULL DEFAULT '[]',
        scope TEXT NOT NULL DEFAULT 'user-{workspace}',
        supers TEXT NOT NULL DEFAULT '[]',
        subs TEXT NOT NULL DEFAULT '[]',
        created TEXT NOT NULL,
        updated TEXT NOT NULL,
        embedding TEXT,
        is_dirty INTEGER NOT NULL DEFAULT 1,
        dirty_reason TEXT,
        dirty_timestamp TEXT,
        dirty_priority INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_claims_state ON claims(state);
    CREATE INDEX IF NOT EXISTS idx_claims_is_dirty ON claims(is_dirty);
    CREATE INDEX IF NOT EXISTS idx_claims_dirty_priority ON claims(dirty_priority);
    """

    def __init__(self, db_path: str = "data/conjecture.db"):
        """
        Initialize the SQLite manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and schema."""
        if self._initialized:
            return

        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Connect and create schema
        self._connection = await aiosqlite.connect(self.db_path)
        await self._connection.executescript(self.SCHEMA)
        await self._connection.commit()

        self._initialized = True
        logger.info(f"SQLite database initialized: {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False
            logger.info("SQLite connection closed")

    async def _ensure_connected(self) -> None:
        """Ensure database is connected."""
        if not self._initialized or not self._connection:
            await self.initialize()

    def _claim_to_row(self, claim: Claim) -> Dict[str, Any]:
        """Convert a Claim to a database row."""
        return {
            "id": claim.id,
            "content": claim.content,
            "confidence": claim.confidence,
            "state": claim.state.value,
            "type": json.dumps([t.value for t in claim.type]),
            "tags": json.dumps(claim.tags),
            "scope": claim.scope.value,
            "supers": json.dumps(claim.supers),
            "subs": json.dumps(claim.subs),
            "created": claim.created.isoformat(),
            "updated": claim.updated.isoformat(),
            "embedding": json.dumps(claim.embedding) if claim.embedding else None,
            "is_dirty": 1 if claim.is_dirty else 0,
            "dirty_reason": claim.dirty_reason.value if claim.dirty_reason else None,
            "dirty_timestamp": claim.dirty_timestamp.isoformat() if claim.dirty_timestamp else None,
            "dirty_priority": claim.dirty_priority,
        }

    def _row_to_claim(self, row: aiosqlite.Row) -> Claim:
        """Convert a database row to a Claim."""
        row_dict = dict(row)
        return Claim(
            id=row_dict["id"],
            content=row_dict["content"],
            confidence=row_dict["confidence"],
            state=ClaimState(row_dict["state"]),
            type=[ClaimType(t) for t in json.loads(row_dict["type"])],
            tags=json.loads(row_dict["tags"]),
            scope=ClaimScope(row_dict["scope"]),
            supers=json.loads(row_dict["supers"]),
            subs=json.loads(row_dict["subs"]),
            created=datetime.fromisoformat(row_dict["created"]),
            updated=datetime.fromisoformat(row_dict["updated"]),
            embedding=json.loads(row_dict["embedding"]) if row_dict["embedding"] else None,
            is_dirty=bool(row_dict["is_dirty"]),
            dirty=bool(row_dict["is_dirty"]),
            dirty_reason=DirtyReason(row_dict["dirty_reason"]) if row_dict["dirty_reason"] else None,
            dirty_timestamp=datetime.fromisoformat(row_dict["dirty_timestamp"]) if row_dict["dirty_timestamp"] else None,
            dirty_priority=row_dict["dirty_priority"],
        )

    async def create_claim(self, claim: Claim) -> str:
        """
        Create a new claim.

        Args:
            claim: The Claim object to create

        Returns:
            The claim ID

        Raises:
            ValueError: If claim with same ID already exists
        """
        await self._ensure_connected()

        row = self._claim_to_row(claim)
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))

        try:
            await self._connection.execute(
                f"INSERT INTO claims ({columns}) VALUES ({placeholders})",
                list(row.values())
            )
            await self._connection.commit()
            logger.debug(f"Created claim: {claim.id}")
            return claim.id
        except aiosqlite.IntegrityError:
            raise ValueError(f"Claim already exists: {claim.id}")

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """
        Retrieve a claim by ID.

        Args:
            claim_id: The claim ID

        Returns:
            Claim if found, None otherwise
        """
        await self._ensure_connected()

        async with self._connection.execute(
            "SELECT * FROM claims WHERE id = ?",
            (claim_id,)
        ) as cursor:
            cursor.row_factory = aiosqlite.Row
            row = await cursor.fetchone()
            if row:
                return self._row_to_claim(row)
            return None

    async def update_claim(self, claim_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing claim.

        Args:
            claim_id: The claim ID to update
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        await self._ensure_connected()

        # Get current claim
        existing = await self.get_claim(claim_id)
        if not existing:
            return False

        # Apply updates
        updates["updated"] = datetime.now(timezone.utc).isoformat()

        # Build SET clause
        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ["type", "tags", "supers", "subs", "embedding"]:
                value = json.dumps(value)
            elif key == "state" and hasattr(value, "value"):
                value = value.value
            elif key == "dirty_reason" and hasattr(value, "value"):
                value = value.value
            elif key == "is_dirty":
                value = 1 if value else 0
            elif isinstance(value, datetime):
                value = value.isoformat()
            set_parts.append(f"{key} = ?")
            values.append(value)

        values.append(claim_id)

        await self._connection.execute(
            f"UPDATE claims SET {', '.join(set_parts)} WHERE id = ?",
            values
        )
        await self._connection.commit()
        logger.debug(f"Updated claim: {claim_id}")
        return True

    async def delete_claim(self, claim_id: str) -> bool:
        """
        Delete a claim.

        Args:
            claim_id: The claim ID to delete

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_connected()

        cursor = await self._connection.execute(
            "DELETE FROM claims WHERE id = ?",
            (claim_id,)
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted claim: {claim_id}")
        return deleted

    async def search_claims(self, query: str, limit: int = 10) -> List[Claim]:
        """
        Search claims by content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching claims
        """
        await self._ensure_connected()

        async with self._connection.execute(
            "SELECT * FROM claims WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        ) as cursor:
            cursor.row_factory = aiosqlite.Row
            rows = await cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    async def list_claims(
        self,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Claim]:
        """
        List claims with optional filtering.

        Args:
            filter_dict: Optional filters (state, tags, etc.)

        Returns:
            List of claims
        """
        await self._ensure_connected()

        query = "SELECT * FROM claims"
        params = []

        if filter_dict:
            conditions = []
            if "state" in filter_dict:
                conditions.append("state = ?")
                state = filter_dict["state"]
                params.append(state.value if hasattr(state, "value") else state)
            if "is_dirty" in filter_dict:
                conditions.append("is_dirty = ?")
                params.append(1 if filter_dict["is_dirty"] else 0)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        limit = filter_dict.get("limit", 100) if filter_dict else 100
        query += f" LIMIT {limit}"

        async with self._connection.execute(query, params) as cursor:
            cursor.row_factory = aiosqlite.Row
            rows = await cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    async def batch_create_claims(self, claims: List[Claim]) -> List[str]:
        """
        Create multiple claims in a batch.

        Args:
            claims: List of claims to create

        Returns:
            List of created claim IDs
        """
        await self._ensure_connected()

        created_ids = []
        for claim in claims:
            try:
                claim_id = await self.create_claim(claim)
                created_ids.append(claim_id)
            except ValueError as e:
                logger.warning(f"Batch create skipped: {e}")

        return created_ids

    async def batch_update_claims(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Update multiple claims in a batch.

        Args:
            updates: Dict mapping claim_id to updates dict

        Returns:
            Number of claims updated
        """
        await self._ensure_connected()

        count = 0
        for claim_id, claim_updates in updates.items():
            if await self.update_claim(claim_id, claim_updates):
                count += 1

        return count

    async def get_dirty_claims(self) -> List[Claim]:
        """
        Get claims that need re-evaluation.

        Returns:
            List of dirty claims sorted by priority
        """
        await self._ensure_connected()

        async with self._connection.execute(
            "SELECT * FROM claims WHERE is_dirty = 1 ORDER BY dirty_priority DESC, dirty_timestamp ASC"
        ) as cursor:
            cursor.row_factory = aiosqlite.Row
            rows = await cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    async def count(self) -> int:
        """Return the total number of claims."""
        await self._ensure_connected()

        async with self._connection.execute("SELECT COUNT(*) FROM claims") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
