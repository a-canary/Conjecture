"""
Enhanced SQLite Manager with Advanced Performance Optimizations
Implements comprehensive database performance improvements designed in Phase 2
"""

import aiosqlite
import json
import sqlite3
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import RLock
import weakref

from .connection_pool import ConnectionPool
from .models import (
    Claim,
    Relationship,
    ClaimFilter,
    DataConfig,
    ClaimNotFoundError,
    DataLayerError,
)

logger = logging.getLogger(__name__)

@dataclass
class QueryPlan:
    """Query execution plan with performance metrics"""
    query: str
    params: tuple
    execution_time: float
    rows_affected: int
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations"""
    total_queries: int = 0
    total_time: float = 0.0
    avg_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    connection_waits: float = 0.0

class QueryCache:
    """Thread-safe query result cache with TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def _generate_key(self, query: str, params: tuple) -> str:
        """Generate cache key from query and parameters"""
        return f"{query}:{hash(params)}"

    def get(self, query: str, params: tuple) -> Optional[Any]:
        """Get cached result if valid"""
        key = self._generate_key(query, params)

        with self._lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                    self._hits += 1
                    return result
                else:
                    # Expired entry
                    del self.cache[key]

            self._misses += 1
            return None

    def put(self, query: str, params: tuple, result: Any):
        """Cache result with automatic eviction"""
        key = self._generate_key(query, params)

        with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(),
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (result, datetime.now())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_size
            }

class EnhancedSQLiteManager:
    """
    Enhanced SQLite manager with advanced performance optimizations:
    - Query plan caching
    - Result caching with TTL
    - Enhanced indexing
    - Performance monitoring
    - Connection optimization
    """

    def __init__(self, db_path: str, pool_size: int = 10, enable_caching: bool = True):
        self.db_path = db_path
        self.enable_caching = enable_caching

        # Enhanced connection pool with load monitoring
        self.connection_pool = ConnectionPool(
            db_path=db_path,
            min_connections=max(2, pool_size // 4),
            max_connections=pool_size,
            max_idle_time=300,
            connection_timeout=30
        )

        # Performance components
        self.query_cache = QueryCache() if enable_caching else None
        self.prepared_statements: Dict[str, aiosqlite.Cursor] = {}
        self.performance_metrics = PerformanceMetrics()
        self.query_plans: List[QueryPlan] = []
        self._lock = RLock()

        # Monitoring
        self._initialized = False
        self._slow_query_threshold = 1.0  # seconds

        # Enhanced schema with optimized indexes
        self.enhanced_schema_sql = self._get_enhanced_schema()

    def _get_enhanced_schema(self) -> str:
        """Get enhanced database schema with performance optimizations"""
        return """
        -- Enhanced claims table with performance optimizations
        CREATE TABLE IF NOT EXISTS claims (
            id VARCHAR(20) PRIMARY KEY,
            content TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            state VARCHAR(20) NOT NULL DEFAULT 'Explore' CHECK (state IN ('Explore', 'Validated', 'Orphaned', 'Queued')),
            supported_by TEXT NOT NULL DEFAULT '[]',
            supports TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            type TEXT NOT NULL DEFAULT '["concept"]', -- JSON array for claim types
            scope VARCHAR(50) NOT NULL DEFAULT 'user-workspace',
            created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            embedding TEXT,

            -- Dirty flag fields with enhanced indexing
            is_dirty BOOLEAN NOT NULL DEFAULT true,
            dirty BOOLEAN NOT NULL DEFAULT true,
            dirty_reason VARCHAR(50),
            dirty_timestamp TIMESTAMP,
            dirty_priority INTEGER NOT NULL DEFAULT 0,

            -- Legacy fields
            created_by VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Performance optimization fields
            search_vector TEXT,  -- Precomputed search vector
            access_count INTEGER DEFAULT 0,  -- Access frequency
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Last access time
        );

        -- Claim relationships with enhanced indexing
        CREATE TABLE IF NOT EXISTS claim_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            supporter_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
            supported_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
            relationship_type VARCHAR(20) NOT NULL DEFAULT 'supports',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(50),
            strength REAL DEFAULT 1.0 CHECK (strength >= 0.0 AND strength <= 1.0),
            UNIQUE(supporter_id, supported_id, relationship_type)
        );

        -- Enhanced indexes for optimal query performance
        CREATE INDEX IF NOT EXISTS idx_claims_confidence ON claims(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_state ON claims(state);
        CREATE INDEX IF NOT EXISTS idx_claims_dirty ON claims(is_dirty);
        CREATE INDEX IF NOT EXISTS idx_claims_scope ON claims(scope);
        CREATE INDEX IF NOT EXISTS idx_claims_created ON claims(created DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_updated ON claims(updated DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_access_count ON claims(access_count DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_last_accessed ON claims(last_accessed DESC);

        -- Composite indexes for common query patterns
        CREATE INDEX IF NOT EXISTS idx_claims_state_confidence ON claims(state, confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_scope_dirty_priority ON claims(scope, is_dirty, dirty_priority DESC);
        CREATE INDEX IF NOT EXISTS idx_claims_created_state ON claims(created DESC, state);
        CREATE INDEX IF NOT EXISTS idx_claims_confidence_state ON claims(confidence DESC, state);

        -- Partial indexes for filtered queries
        CREATE INDEX IF NOT EXISTS idx_claims_dirty_only ON claims(id, dirty_priority DESC) WHERE is_dirty = true;
        CREATE INDEX IF NOT EXISTS idx_claims_high_confidence ON claims(id, created DESC) WHERE confidence >= 0.8;
        CREATE INDEX IF NOT EXISTS idx_claims_recent ON claims(id, confidence DESC) WHERE created >= date('now', '-7 days');
        CREATE INDEX IF NOT EXISTS idx_claims_frequent ON claims(id, last_accessed DESC) WHERE access_count > 10;

        -- Relationship indexes
        CREATE INDEX IF NOT EXISTS idx_relationships_supporter ON claim_relationships(supporter_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_supported ON claim_relationships(supported_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON claim_relationships(relationship_type);
        CREATE INDEX IF NOT EXISTS idx_relationships_strength ON claim_relationships(strength DESC);
        CREATE INDEX IF NOT EXISTS idx_relationships_composite ON claim_relationships(relationship_type, strength DESC);

        -- Full-text search with enhanced configuration
        CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
            content,
            tags,
            content=claims,
            content_rowid=rowid,
            tokenize='porter unicode61'
        );

        -- Enhanced triggers for FTS and performance tracking
        CREATE TRIGGER IF NOT EXISTS claims_fts_insert AFTER INSERT ON claims
        BEGIN
            INSERT INTO claims_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS claims_fts_update AFTER UPDATE ON claims
        BEGIN
            UPDATE claims_fts SET content = new.content, tags = new.tags WHERE rowid = new.rowid;
        END;

        CREATE TRIGGER IF NOT EXISTS claims_fts_delete AFTER DELETE ON claims
        BEGIN
            DELETE FROM claims_fts WHERE rowid = old.rowid;
        END;

        -- Performance tracking triggers
        CREATE TRIGGER IF NOT EXISTS claims_access_trigger AFTER SELECT ON claims
        BEGIN
            UPDATE claims SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """

    async def initialize(self) -> None:
        """Initialize enhanced database with all optimizations"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Initialize connection pool
            await self.connection_pool.initialize()

            # Create enhanced schema
            async with self.connection_pool.get_connection() as conn:
                await conn.executescript(self.enhanced_schema_sql)

                # Optimize database settings for performance
                optimizations = [
                    "PRAGMA journal_mode=WAL",
                    "PRAGMA synchronous=NORMAL",
                    "PRAGMA cache_size=64000",  # Increased cache
                    "PRAGMA temp_store=MEMORY",
                    "PRAGMA mmap_size=268435456",  # 256MB memory mapping
                    "PRAGMA optimize",
                    "PRAGMA analysis_limit=1000",
                    "PRAGMA threads=4"  # Multi-threading
                ]

                for pragma in optimizations:
                    await conn.execute(pragma)

                # Update statistics for query optimizer
                await conn.execute("ANALYZE")
                await conn.commit()

            self._initialized = True
            logger.info(f"Enhanced SQLite manager initialized with {self.connection_pool.max_connections} connections")

        except Exception as e:
            raise DataLayerError(f"Failed to initialize enhanced SQLite: {e}")

    async def close(self) -> None:
        """Close database and cleanup resources"""
        if self.connection_pool:
            await self.connection_pool.close()

        if self.query_cache:
            logger.info(f"Final cache stats: {self.query_cache.get_stats()}")

        self._initialized = False

    @asynccontextmanager
    async def _get_connection(self):
        """Get connection with performance monitoring"""
        start_time = time.time()

        async with self.connection_pool.get_connection() as conn:
            # Record connection wait time
            wait_time = time.time() - start_time
            self.performance_metrics.connection_waits += wait_time

            # Configure connection for optimal performance
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA query_only=OFF")

            yield conn

    async def _execute_with_monitoring(
        self,
        query: str,
        params: tuple = (),
        use_cache: bool = True
    ) -> aiosqlite.Cursor:
        """Execute query with performance monitoring and caching"""
        start_time = time.time()

        # Check cache if enabled and appropriate
        if (use_cache and self.query_cache and
            query.strip().upper().startswith('SELECT')):
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                return cached_result

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params)
                execution_time = time.time() - start_time

                # Update metrics
                self._update_performance_metrics(query, execution_time, cursor.rowcount)

                # Cache result if appropriate
                if (use_cache and self.query_cache and
                    query.strip().upper().startswith('SELECT') and
                    execution_time < self._slow_query_threshold):
                    # For SELECT queries, fetch and cache results
                    if cursor.rowcount < 1000:  # Only cache reasonably sized results
                        results = await cursor.fetchall()
                        self.query_cache.put(query, params, results)
                        return results

                return cursor

        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
            raise DataLayerError(f"Query execution failed: {e}")

    def _update_performance_metrics(self, query: str, execution_time: float, rows_affected: int):
        """Update performance metrics"""
        with self._lock:
            self.performance_metrics.total_queries += 1
            self.performance_metrics.total_time += execution_time
            self.performance_metrics.avg_query_time = (
                self.performance_metrics.total_time / self.performance_metrics.total_queries
            )

            # Track slow queries
            if execution_time > self._slow_query_threshold:
                logger.warning(f"Slow query detected ({execution_time:.2f}s): {query[:100]}...")

            # Store query plan for analysis
            plan = QueryPlan(
                query=query,
                params=(),
                execution_time=execution_time,
                rows_affected=rows_affected,
                timestamp=datetime.now()
            )
            self.query_plans.append(plan)

            # Keep only recent plans
            if len(self.query_plans) > 1000:
                self.query_plans = self.query_plans[-1000:]

    async def create_claim(self, claim: Claim) -> str:
        """Create claim with performance optimizations"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        try:
            async with self._get_connection() as conn:
                # Use transaction for atomicity
                await conn.execute("BEGIN IMMEDIATE")

                try:
                    await conn.execute(
                        """
                        INSERT INTO claims (
                            id, content, confidence, state, supported_by, supports,
                            tags, scope, embedding, is_dirty, dirty, dirty_reason,
                            dirty_timestamp, dirty_priority, created_by, search_vector,
                            access_count, last_accessed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            claim.id,
                            claim.content,
                            claim.confidence,
                            claim.state.value,
                            json.dumps(claim.supported_by),
                            json.dumps(claim.supports),
                            json.dumps(claim.tags),
                            claim.scope.value,
                            json.dumps(claim.embedding) if claim.embedding else None,
                            claim.is_dirty,
                            claim.dirty,
                            claim.dirty_reason.value if claim.dirty_reason else None,
                            claim.dirty_timestamp.isoformat() if claim.dirty_timestamp else None,
                            claim.dirty_priority,
                            getattr(claim, "created_by", "system"),
                            self._generate_search_vector(claim.content, claim.tags),
                            0,  # Initial access count
                            datetime.now().isoformat()
                        )
                    )

                    # Invalidate relevant cache entries
                    if self.query_cache:
                        self._invalidate_cache_for_pattern("SELECT * FROM claims")

                    await conn.commit()
                    return claim.id

                except Exception:
                    await conn.rollback()
                    raise

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise DataLayerError(f"Claim with ID {claim.id} already exists")
            raise DataLayerError(f"Failed to create claim: {e}")

    def _generate_search_vector(self, content: str, tags: List[str]) -> str:
        """Generate optimized search vector for FTS"""
        # Combine content and tags for better search
        tag_text = " ".join(tags) if tags else ""
        return f"{content} {tag_text}"

    def _invalidate_cache_for_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if not self.query_cache:
            return

        with self.query_cache._lock:
            keys_to_remove = [
                key for key in self.query_cache.cache.keys()
                if pattern in key
            ]
            for key in keys_to_remove:
                del self.query_cache.cache[key]

    async def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get claim with performance optimizations"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        async with self._get_connection() as conn:
            conn.connection.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT * FROM claims WHERE id = ?
                """,
                (claim_id,)
            )
            row = await cursor.fetchone()

            if row:
                claim_dict = dict(row)
                # Parse JSON fields
                claim_dict["tags"] = json.loads(claim_dict["tags"] or "[]")
                claim_dict["supported_by"] = json.loads(claim_dict["supported_by"] or "[]")
                claim_dict["supports"] = json.loads(claim_dict["supports"] or "[]")
                claim_dict["embedding"] = (
                    json.loads(claim_dict["embedding"]) if claim_dict["embedding"] else None
                )

                # Update access statistics
                await conn.execute(
                    "UPDATE claims SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                    (claim_id,)
                )
                await conn.commit()

                return claim_dict
            return None

    async def filter_claims(self, filters: ClaimFilter) -> List[Dict[str, Any]]:
        """Enhanced claim filtering with optimized queries"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        # Build optimized query based on filters
        query_parts = ["SELECT * FROM claims WHERE 1=1"]
        params = []

        # Use appropriate indexes based on filters
        if filters.states:
            state_placeholders = ",".join("?" for _ in filters.states)
            query_parts.append(f"AND state IN ({state_placeholders})")
            params.extend([s.value for s in filters.states])

        if filters.confidence_min is not None:
            query_parts.append("AND confidence >= ?")
            params.append(filters.confidence_min)

        if filters.confidence_max is not None:
            query_parts.append("AND confidence <= ?")
            params.append(filters.confidence_max)

        if filters.dirty_only is not None:
            query_parts.append("AND dirty = ?")
            params.append(filters.dirty_only)

        if filters.tags:
            # Enhanced tag filtering using JSON functions
            tag_conditions = []
            for tag in filters.tags:
                tag_conditions.append("json_extract(tags, '$') LIKE ?")
                params.append(f'%"{tag}"%')
            query_parts.append(f"AND ({' OR '.join(tag_conditions)})")

        if filters.content_contains:
            # Use FTS for content search
            query_parts[0] = "SELECT claims.* FROM claims_fts JOIN claims ON claims.rowid = claims_fts.rowid WHERE claims_fts MATCH ?"
            params.insert(0, filters.content_contains)
            # Add back the WHERE 1=1 after the JOIN
            query_parts.insert(1, "AND 1=1")

        # Optimize ordering based on filters
        if filters.confidence_min is not None and filters.confidence_min > 0.8:
            query_parts.append("ORDER BY confidence DESC, access_count DESC")
        elif filters.created_after:
            query_parts.append("ORDER BY created DESC")
        else:
            query_parts.append("ORDER BY last_accessed DESC")

        # Add pagination
        query_parts.append("LIMIT ? OFFSET ?")
        params.extend([filters.limit, filters.offset])

        query = " ".join(query_parts)

        async with self._get_connection() as conn:
            conn.connection.row_factory = aiosqlite.Row
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            claims = []
            for row in rows:
                claim_dict = dict(row)
                claim_dict["tags"] = json.loads(claim_dict["tags"] or "[]")
                claim_dict["supported_by"] = json.loads(claim_dict["supported_by"] or "[]")
                claim_dict["supports"] = json.loads(claim_dict["supports"] or "[]")
                claim_dict["embedding"] = (
                    json.loads(claim_dict["embedding"]) if claim_dict["embedding"] else None
                )
                claims.append(claim_dict)

            return claims

    async def batch_create_claims(self, claims: List[Claim]) -> List[str]:
        """Enhanced batch creation with optimized performance"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        if not claims:
            return []

        # Prepare batch data
        claim_data = []
        for claim in claims:
            claim_data.append((
                claim.id,
                claim.content,
                claim.confidence,
                claim.state.value,
                json.dumps(claim.supported_by),
                json.dumps(claim.supports),
                json.dumps(claim.tags),
                claim.scope.value,
                json.dumps(claim.embedding) if claim.embedding else None,
                claim.is_dirty,
                claim.dirty,
                claim.dirty_reason.value if claim.dirty_reason else None,
                claim.dirty_timestamp.isoformat() if claim.dirty_timestamp else None,
                claim.dirty_priority,
                getattr(claim, "created_by", "system"),
                self._generate_search_vector(claim.content, claim.tags),
                0,  # Initial access count
                datetime.now().isoformat()
            ))

        try:
            async with self._get_connection() as conn:
                # Use transaction for batch atomicity
                await conn.execute("BEGIN IMMEDIATE")

                try:
                    await conn.executemany(
                        """
                        INSERT INTO claims (
                            id, content, confidence, state, supported_by, supports,
                            tags, scope, embedding, is_dirty, dirty, dirty_reason,
                            dirty_timestamp, dirty_priority, created_by, search_vector,
                            access_count, last_accessed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        claim_data
                    )

                    # Invalidate cache
                    if self.query_cache:
                        self._invalidate_cache_for_pattern("SELECT * FROM claims")

                    await conn.commit()
                    return [claim.id for claim in claims]

                except Exception:
                    await conn.rollback()
                    raise

        except sqlite3.Error as e:
            raise DataLayerError(f"Batch create failed: {e}")

    async def get_dirty_claims(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Optimized dirty claims query using enhanced indexes"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        # Use optimized index for dirty claims
        async with self._get_connection() as conn:
            conn.connection.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT * FROM claims
                WHERE is_dirty = true
                ORDER BY dirty_priority DESC, confidence ASC, last_accessed ASC
                LIMIT ?
                """,
                (limit,)
            )
            rows = await cursor.fetchall()

            claims = []
            for row in rows:
                claim_dict = dict(row)
                claim_dict["tags"] = json.loads(claim_dict["tags"] or "[]")
                claim_dict["supported_by"] = json.loads(claim_dict["supported_by"] or "[]")
                claim_dict["supports"] = json.loads(claim_dict["supports"] or "[]")
                claim_dict["embedding"] = (
                    json.loads(claim_dict["embedding"]) if claim_dict["embedding"] else None
                )
                claims.append(claim_dict)

            return claims

    async def update_claim(self, claim_id: str, updates: Dict[str, Any]) -> bool:
        """Enhanced claim update with performance optimizations"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        if not updates:
            return False

        # Handle special fields
        json_fields = ["tags", "supported_by", "supports", "embedding"]
        for field in json_fields:
            if field in updates:
                if updates[field] is not None:
                    updates[field] = json.dumps(updates[field])
                else:
                    updates[field] = None

        # Handle enum fields
        if "dirty_reason" in updates and updates["dirty_reason"] is not None:
            if hasattr(updates["dirty_reason"], "value"):
                updates["dirty_reason"] = updates["dirty_reason"].value

        # Handle timestamps
        if "dirty_timestamp" in updates and updates["dirty_timestamp"] is not None:
            if hasattr(updates["dirty_timestamp"], "isoformat"):
                updates["dirty_timestamp"] = updates["dirty_timestamp"].isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [claim_id]

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    f"""
                    UPDATE claims SET {set_clause}, updated = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    values
                )

                # Invalidate cache for this claim
                if self.query_cache:
                    self._invalidate_cache_for_pattern(f"SELECT * FROM claims WHERE id = {claim_id}")
                    self._invalidate_cache_for_pattern("SELECT * FROM claims WHERE")

                await conn.commit()
                return cursor.rowcount > 0

        except sqlite3.Error as e:
            raise DataLayerError(f"Failed to update claim: {e}")

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        # Get database statistics
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM claims")
            total_claims = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM claims WHERE is_dirty = true")
            dirty_claims = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM claim_relationships")
            total_relationships = (await cursor.fetchone())[0]

        # Get connection pool stats
        pool_stats = self.connection_pool.get_stats()

        # Get cache stats
        cache_stats = self.query_cache.get_stats() if self.query_cache else None

        # Analyze slow queries
        slow_queries = [
            plan for plan in self.query_plans[-100:]  # Last 100 queries
            if plan.execution_time > self._slow_query_threshold
        ]

        return {
            "database_stats": {
                "total_claims": total_claims,
                "dirty_claims": dirty_claims,
                "total_relationships": total_relationships,
                "clean_claims": total_claims - dirty_claims
            },
            "performance_metrics": {
                "total_queries": self.performance_metrics.total_queries,
                "avg_query_time": self.performance_metrics.avg_query_time,
                "total_execution_time": self.performance_metrics.total_time,
                "connection_waits": self.performance_metrics.connection_waits
            },
            "connection_pool": pool_stats,
            "cache_stats": cache_stats,
            "slow_queries": {
                "count": len(slow_queries),
                "threshold": self._slow_query_threshold,
                "recent_slow_queries": [
                    {
                        "query": plan.query[:100] + "..." if len(plan.query) > 100 else plan.query,
                        "execution_time": plan.execution_time,
                        "timestamp": plan.timestamp.isoformat()
                    }
                    for plan in slow_queries[-10:]  # Last 10 slow queries
                ]
            },
            "optimizations": {
                "caching_enabled": self.enable_caching,
                "prepared_statements": len(self.prepared_statements),
                "query_plans_tracked": len(self.query_plans)
            }
        }

    async def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization routines"""
        if not self._initialized:
            raise DataLayerError("Database not initialized")

        optimization_results = {}

        try:
            async with self._get_connection() as conn:
                # Run ANALYZE to update statistics
                start_time = time.time()
                await conn.execute("ANALYZE")
                optimization_results["analyze_time"] = time.time() - start_time

                # Run VACUUM if needed (check fragmentation first)
                cursor = await conn.execute("PRAGMA page_count")
                page_count = (await cursor.fetchone())[0]

                cursor = await conn.execute("PRAGMA freelist_count")
                freelist_count = (await cursor.fetchone())[0]

                fragmentation_ratio = freelist_count / page_count if page_count > 0 else 0

                if fragmentation_ratio > 0.1:  # 10% fragmentation threshold
                    start_time = time.time()
                    await conn.execute("VACUUM")
                    optimization_results["vacuum_time"] = time.time() - start_time
                    optimization_results["vacuum_run"] = True
                else:
                    optimization_results["vacuum_run"] = False
                    optimization_results["fragmentation_ratio"] = fragmentation_ratio

                # Rebuild indexes if needed
                start_time = time.time()
                await conn.execute("REINDEX")
                optimization_results["reindex_time"] = time.time() - start_time

                await conn.commit()

                # Clear cache after optimization
                if self.query_cache:
                    with self.query_cache._lock:
                        self.query_cache.cache.clear()

                optimization_results["success"] = True

        except Exception as e:
            optimization_results["success"] = False
            optimization_results["error"] = str(e)
            logger.error(f"Database optimization failed: {e}")

        return optimization_results