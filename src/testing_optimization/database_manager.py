"""
Optimized Database State Management for Testing
Provides database isolation, connection pooling, and efficient test data management.
"""
import asyncio
import sqlite3
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
import json
import uuid
import time
from dataclasses import dataclass
import aiosqlite
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class DatabaseConfig:
    """Configuration for test database."""
    type: str = "sqlite"
    isolation_level: str = "EXCLUSIVE"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = 2000
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456  # 256MB


class DatabaseIsolationManager:
    """Manages database isolation for test execution."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="test_db_"))
        self.active_connections: Dict[str, aiosqlite.Connection] = {}
        self.db_configs: Dict[str, DatabaseConfig] = {}
        self._lock = asyncio.Lock()
        self._cleanup_registered = False

    def _ensure_cleanup_registered(self):
        """Ensure cleanup function is registered on exit."""
        if not self._cleanup_registered:
            import atexit
            atexit.register(self.cleanup_all)
            self._cleanup_registered = True

    async def create_isolated_database(self, test_id: str, config: Optional[DatabaseConfig] = None) -> str:
        """Create an isolated database for a specific test."""
        config = config or DatabaseConfig()
        self._ensure_cleanup_registered()

        db_path = self.temp_dir / f"test_{test_id}_{uuid.uuid4().hex[:8]}.db"
        connection_string = f"file:{db_path}?mode=rwc"

        async with self._lock:
            # Create and configure database
            conn = await aiosqlite.connect(connection_string)

            # Apply performance optimizations
            await conn.execute(f"PRAGMA isolation_level = {config.isolation_level}")
            await conn.execute(f"PRAGMA journal_mode = {config.journal_mode}")
            await conn.execute(f"PRAGMA synchronous = {config.synchronous}")
            await conn.execute(f"PRAGMA cache_size = {config.cache_size}")
            await conn.execute(f"PRAGMA temp_store = {config.temp_store}")
            await conn.execute(f"PRAGMA mmap_size = {config.mmap_size}")

            # Enable foreign keys
            await conn.execute("PRAGMA foreign_keys = ON")

            await conn.commit()

            self.active_connections[test_id] = conn
            self.db_configs[test_id] = config

        return str(db_path)

    @asynccontextmanager
    async def get_connection(self, test_id: str) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get database connection for a test with automatic cleanup."""
        if test_id not in self.active_connections:
            raise ValueError(f"No database found for test: {test_id}")

        conn = self.active_connections[test_id]

        # Start transaction
        await conn.execute("BEGIN IMMEDIATE")

        try:
            yield conn
        except Exception:
            await conn.rollback()
            raise
        else:
            await conn.commit()

    async def cleanup_database(self, test_id: str):
        """Clean up database for a specific test."""
        async with self._lock:
            if test_id in self.active_connections:
                await self.active_connections[test_id].close()
                del self.active_connections[test_id]

            if test_id in self.db_configs:
                del self.db_configs[test_id]

    def cleanup_database_sync(self, test_id: str):
        """Synchronous cleanup for database."""
        if test_id in self.active_connections:
            # Use run_in_executor to make async cleanup synchronous
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.cleanup_database(test_id))
            finally:
                loop.close()

    async def cleanup_all(self):
        """Clean up all databases and connections."""
        async with self._lock:
            for conn in self.active_connections.values():
                await conn.close()
            self.active_connections.clear()
            self.db_configs.clear()

        # Clean up temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def cleanup_all_sync(self):
        """Synchronous cleanup for all databases."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.cleanup_all())
        finally:
            loop.close()


class TestDataManager:
    """Manages test data generation and cleanup for efficiency."""

    def __init__(self):
        self._data_cache: Dict[str, Any] = {}
        self._utf8_test_strings = [
            "ASCII only test string",
            "Café Résumé - UTF-8 accents",
            "北京测试 - Chinese characters",
            "Тест Москва - Cyrillic characters",
            "العربية اختبار - Arabic characters",
            "🚀 Test Emoji 🧪 Scientific",
            "Mixed: café 北京 🌟 Тест العربية",
            "Mathematical: ∑∏∫∆∇∂",
            "Special: \"quotes\" 'apostrophes' &symbols",
            "Long text with UTF-8: " + " café " * 100
        ]

    def get_utf8_test_strings(self) -> List[str]:
        """Get standardized UTF-8 test strings."""
        return self._utf8_test_strings.copy()

    def generate_test_claim(self, claim_id: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """Generate a test claim with UTF-8 compliance."""
        if claim_id is None:
            claim_id = f"test_claim_{uuid.uuid4().hex[:8]}"

        base_claim = {
            "id": claim_id,
            "content": f"Test claim {claim_id} with UTF-8: café 北京 🌟",
            "source": f"test_source_{claim_id}",
            "confidence": 0.7,
            "tags": ["test", "utf8", "optimization"],
            "state": "pending",
            "claim_type": "hypothesis",
            "metadata": {
                "test_id": claim_id,
                "created_at": time.time(),
                "utf8_valid": True
            }
        }

        base_claim.update(overrides)
        return base_claim

    def generate_test_claims(self, count: int, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate multiple test claims efficiently."""
        batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        claims = []

        for i in range(count):
            claim = self.generate_test_claim(
                claim_id=f"{batch_id}_{i:04d}",
                content=f"Test claim {i+1}/{count} with UTF-8: {self._utf8_test_strings[i % len(self._utf8_test_strings)]}",
                confidence=0.5 + (i % 5) * 0.1
            )
            claims.append(claim)

        return claims

    def validate_utf8_compliance(self, data: Any) -> bool:
        """Validate UTF-8 encoding compliance."""
        try:
            if isinstance(data, str):
                data.encode('utf-8')
            elif isinstance(data, dict):
                json.dumps(data, ensure_ascii=False).encode('utf-8')
            elif isinstance(data, list):
                for item in data:
                    self.validate_utf8_compliance(item)
            return True
        except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
            return False

    async def create_database_schema(self, conn: aiosqlite.Connection) -> None:
        """Create optimized database schema for testing."""
        schema_sql = """
        -- Claims table with UTF-8 support
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            tags TEXT, -- JSON array
            state TEXT NOT NULL DEFAULT 'pending',
            claim_type TEXT NOT NULL DEFAULT 'hypothesis',
            metadata TEXT, -- JSON object
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            hash TEXT, -- Content hash for quick comparison
            dirty_flag INTEGER DEFAULT 1
        );

        -- Relationships table
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_claim_id TEXT NOT NULL,
            target_claim_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            metadata TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (source_claim_id) REFERENCES claims(id) ON DELETE CASCADE,
            FOREIGN KEY (target_claim_id) REFERENCES claims(id) ON DELETE CASCADE
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_claims_state ON claims(state);
        CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
        CREATE INDEX IF NOT EXISTS idx_claims_confidence ON claims(confidence);
        CREATE INDEX IF NOT EXISTS idx_claims_created ON claims(created_at);
        CREATE INDEX IF NOT EXISTS idx_claims_dirty ON claims(dirty_flag);
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_claim_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_claim_id);

        -- Triggers for automatic timestamp updates
        CREATE TRIGGER IF NOT EXISTS update_claim_timestamp
        AFTER UPDATE ON claims
        FOR EACH ROW
        BEGIN
            UPDATE claims SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
        END;
        """

        # Execute schema creation
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                await conn.execute(statement)
        await conn.commit()


class DatabaseConnectionPool:
    """Connection pool for efficient database access in testing."""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._pool: List[aiosqlite.Connection] = []
        self._available = asyncio.Event()
        self._lock = asyncio.Lock()
        self._created_connections = 0

    async def get_connection(self, database_path: str) -> aiosqlite.Connection:
        """Get a connection from the pool."""
        async with self._lock:
            while self._pool:
                conn = self._pool.pop()
                try:
                    # Test if connection is still alive
                    await conn.execute("SELECT 1")
                    return conn
                except Exception:
                    # Connection is dead, close it
                    await conn.close()

            # Create new connection if under limit
            if self._created_connections < self.max_connections:
                conn = await aiosqlite.connect(database_path)
                self._created_connections += 1
                return conn

        # If no connections available, wait
        await self._available.wait()
        return await self.get_connection(database_path)

    async def return_connection(self, conn: aiosqlite.Connection):
        """Return a connection to the pool."""
        async with self._lock:
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
                self._available.set()
            else:
                await conn.close()
                self._created_connections -= 1

    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
            self._created_connections = 0


class PerformanceOptimizedTestDatabase:
    """High-performance test database with optimizations."""

    def __init__(self, test_id: str, isolation_manager: DatabaseIsolationManager):
        self.test_id = test_id
        self.isolation_manager = isolation_manager
        self.db_path: Optional[str] = None
        self.data_manager = TestDataManager()
        self.connection_pool = DatabaseConnectionPool()

    async def initialize(self, schema_config: Optional[Dict[str, Any]] = None) -> str:
        """Initialize the test database."""
        self.db_path = await self.isolation_manager.create_isolated_database(self.test_id)

        async with self.isolation_manager.get_connection(self.test_id) as conn:
            await self.data_manager.create_database_schema(conn)

        return self.db_path

    async def insert_claims_batch(self, claims: List[Dict[str, Any]]) -> List[str]:
        """Efficiently insert multiple claims in a batch."""
        if not self.db_path:
            raise RuntimeError("Database not initialized")

        inserted_ids = []

        async with self.isolation_manager.get_connection(self.test_id) as conn:
            # Begin transaction
            await conn.execute("BEGIN IMMEDIATE")

            try:
                for claim in claims:
                    # Validate UTF-8 compliance
                    if not self.data_manager.validate_utf8_compliance(claim):
                        raise ValueError(f"UTF-8 validation failed for claim: {claim.get('id')}")

                    # Insert claim
                    await conn.execute("""
                        INSERT INTO claims (id, content, source, confidence, tags, state,
                                          claim_type, metadata, created_at, updated_at, hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        claim["id"],
                        claim["content"],
                        claim["source"],
                        claim["confidence"],
                        json.dumps(claim.get("tags", []), ensure_ascii=False),
                        claim["state"],
                        claim["claim_type"],
                        json.dumps(claim.get("metadata", {}), ensure_ascii=False),
                        claim.get("created_at", time.time()),
                        claim.get("updated_at", time.time()),
                        self._calculate_content_hash(claim["content"])
                    ))
                    inserted_ids.append(claim["id"])

                await conn.commit()

            except Exception:
                await conn.rollback()
                raise

        return inserted_ids

    async def query_claims(self, filter_criteria: Optional[Dict[str, Any]] = None,
                          limit: Optional[int] = None,
                          offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query claims with optional filtering and pagination."""
        if not self.db_path:
            raise RuntimeError("Database not initialized")

        query = "SELECT * FROM claims"
        params = []
        conditions = []

        if filter_criteria:
            if "state" in filter_criteria:
                conditions.append("state = ?")
                params.append(filter_criteria["state"])
            if "claim_type" in filter_criteria:
                conditions.append("claim_type = ?")
                params.append(filter_criteria["claim_type"])
            if "min_confidence" in filter_criteria:
                conditions.append("confidence >= ?")
                params.append(filter_criteria["min_confidence"])
            if "max_confidence" in filter_criteria:
                conditions.append("confidence <= ?")
                params.append(filter_criteria["max_confidence"])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        async with self.isolation_manager.get_connection(self.test_id) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        if not self.db_path:
            raise RuntimeError("Database not initialized")

        async with self.isolation_manager.get_connection(self.test_id) as conn:
            # Get database statistics
            cursor = await conn.execute("SELECT COUNT(*) as claim_count FROM claims")
            claim_count = (await cursor.fetchone())["claim_count"]

            cursor = await conn.execute("SELECT COUNT(*) as relationship_count FROM relationships")
            relationship_count = (await cursor.fetchone())["relationship_count"]

            # Get database file size
            db_size = Path(self.db_path).stat().st_size if self.db_path else 0

            return {
                "claim_count": claim_count,
                "relationship_count": relationship_count,
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "test_id": self.test_id,
                "utf8_compliant": True
            }

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content to enable quick comparison."""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def cleanup(self):
        """Clean up database resources."""
        await self.isolation_manager.cleanup_database(self.test_id)
        await self.connection_pool.close_all()


# Global instances for test usage
_global_isolation_manager: Optional[DatabaseIsolationManager] = None


def get_isolation_manager() -> DatabaseIsolationManager:
    """Get global database isolation manager."""
    global _global_isolation_manager
        if _global_isolation_manager is None:
