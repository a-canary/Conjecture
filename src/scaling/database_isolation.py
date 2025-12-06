"""
Enhanced Database Isolation for Concurrent Operations - Phase 2
Provides transaction isolation, connection pooling, and concurrent operation management
"""

import asyncio
import time
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from pathlib import Path
import json
from contextlib import asynccontextmanager
import uuid
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """Database isolation levels"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionMetrics:
    """Metrics for database transactions"""
    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TransactionStatus = TransactionStatus.ACTIVE
    operations_count: int = 0
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    locks_held: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class ConnectionMetrics:
    """Metrics for database connections"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    active_transaction: Optional[str] = None
    operations_count: int = 0
    is_available: bool = True
    pool_name: str = "default"


class DatabaseConnectionPool:
    """
    Connection pool for managing database connections efficiently
    """

    def __init__(
        self,
        database_path: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0
    ):
        self.database_path = database_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout

        # Connection management
        self._available_connections: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active_connections: Dict[str, aiosqlite.Connection] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}

        # Pool state
        self._pool_initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(f"DatabaseConnectionPool initialized: {min_connections}-{max_connections} connections")

    async def initialize(self):
        """Initialize the connection pool"""
        if self._pool_initialized:
            return

        async with self._lock:
            if self._pool_initialized:
                return

            # Ensure database exists
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

            # Create minimum connections
            for i in range(self.min_connections):
                await self._create_connection()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())

            self._pool_initialized = True
            logger.info(f"Connection pool initialized with {self.min_connections} connections")

    async def _create_connection(self) -> str:
        """Create a new database connection"""
        connection_id = str(uuid.uuid4())

        try:
            # Create connection with optimized settings
            connection = await aiosqlite.connect(
                self.database_path,
                timeout=self.connection_timeout
            )

            # Configure connection for concurrent access
            await connection.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA cache_size=10000")
            await connection.execute("PRAGMA temp_store=MEMORY")
            await connection.execute("PRAGMA busy_timeout=30000")

            # Store connection
            self._active_connections[connection_id] = connection
            self._connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                is_available=True
            )

            # Add to available queue
            await self._available_connections.put(connection_id)

            logger.debug(f"Created connection {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self._pool_initialized:
            await self.initialize()

        connection_id = None
        try:
            # Get connection from pool or create new one
            try:
                connection_id = await asyncio.wait_for(
                    self._available_connections.get(),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                # Try to create new connection if under limit
                if len(self._active_connections) < self.max_connections:
                    connection_id = await self._create_connection()
                else:
                    raise RuntimeError("Connection pool exhausted")

            # Get connection object
            connection = self._active_connections[connection_id]
            metrics = self._connection_metrics[connection_id]

            # Update metrics
            metrics.is_available = False
            metrics.last_used = datetime.utcnow()
            metrics.operations_count += 1

            yield connection, connection_id

        except Exception as e:
            logger.error(f"Error getting connection: {e}")
            raise
        finally:
            # Return connection to pool
            if connection_id and connection_id in self._connection_metrics:
                metrics = self._connection_metrics[connection_id]
                metrics.is_available = True

                # Don't put back connection if it's in a bad state
                if connection_id in self._active_connections:
                    await self._available_connections.put(connection_id)

    async def _cleanup_idle_connections(self):
        """Periodically cleanup idle connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                current_time = datetime.utcnow()
                connections_to_close = []

                for connection_id, metrics in self._connection_metrics.items():
                    # Check if connection is idle and we have more than minimum
                    if (metrics.is_available and
                        len(self._active_connections) > self.min_connections and
                        (current_time - metrics.last_used).total_seconds() > self.idle_timeout):
                        connections_to_close.append(connection_id)

                # Close idle connections
                for connection_id in connections_to_close:
                    await self._close_connection(connection_id)

                if connections_to_close:
                    logger.info(f"Cleaned up {len(connections_to_close)} idle connections")

            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def _close_connection(self, connection_id: str):
        """Close a specific connection"""
        if connection_id in self._active_connections:
            try:
                connection = self._active_connections[connection_id]
                await connection.close()
                del self._active_connections[connection_id]
                del self._connection_metrics[connection_id]
                logger.debug(f"Closed connection {connection_id}")
            except Exception as e:
                logger.error(f"Error closing connection {connection_id}: {e}")

    async def close_all(self):
        """Close all connections and cleanup pool"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for connection_id in list(self._active_connections.keys()):
            await self._close_connection(connection_id)

        self._pool_initialized = False
        logger.info("Connection pool closed")

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        total_connections = len(self._active_connections)
        available_connections = self._available_connections.qsize()
        active_connections = total_connections - available_connections

        return {
            "total_connections": total_connections,
            "available_connections": available_connections,
            "active_connections": active_connections,
            "pool_initialized": self._pool_initialized,
            "connection_metrics": {
                conn_id: {
                    "created_at": metrics.created_at.isoformat(),
                    "last_used": metrics.last_used.isoformat(),
                    "operations_count": metrics.operations_count,
                    "is_available": metrics.is_available
                }
                for conn_id, metrics in self._connection_metrics.items()
            }
        }


class ConcurrentTransactionManager:
    """
    Manager for concurrent database transactions with proper isolation
    """

    def __init__(self, connection_pool: DatabaseConnectionPool):
        self.connection_pool = connection_pool

        # Transaction management
        self._active_transactions: Dict[str, TransactionMetrics] = {}
        self._transaction_locks: Dict[str, List[str]] = defaultdict(list)  # resource -> transactions
        self._lock_waiters: Dict[str, List[asyncio.Event]] = defaultdict(list)

        # Deadlock detection
        self._deadlock_check_interval = 5.0
        self._deadlock_timeout = 30.0
        self._deadlock_task: Optional[asyncio.Task] = None

        # Performance tracking
        self._transaction_history: deque = deque(maxlen=1000)
        self._stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "deadlocks_detected": 0,
            "average_duration": 0.0
        }

        logger.info("ConcurrentTransactionManager initialized")

    async def start(self):
        """Start the transaction manager"""
        self._deadlock_task = asyncio.create_task(self._deadlock_detection_loop())
        logger.info("Transaction manager started")

    async def stop(self):
        """Stop the transaction manager"""
        if self._deadlock_task:
            self._deadlock_task.cancel()
            try:
                await self._deadlock_task
            except asyncio.CancelledError:
                pass

        # Rollback all active transactions
        for transaction_id in list(self._active_transactions.keys()):
            try:
                await self.rollback_transaction(transaction_id)
            except Exception as e:
                logger.error(f"Error rolling back transaction {transaction_id}: {e}")

        logger.info("Transaction manager stopped")

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        resources: Optional[List[str]] = None,
        timeout: float = 30.0
    ):
        """
        Execute operations within a transaction with proper isolation

        Args:
            isolation_level: Database isolation level
            resources: List of resources this transaction will access
            timeout: Transaction timeout in seconds
        """
        transaction_id = str(uuid.uuid4())

        try:
            # Acquire locks for resources
            await self._acquire_locks(transaction_id, resources or [])

            # Get connection and start transaction
            async with self.connection_pool.get_connection() as (connection, conn_id):
                # Set isolation level
                await connection.execute(f"PRAGMA read_uncommitted = {isolation_level == IsolationLevel.READ_UNCOMMITTED}")

                # Start transaction
                await connection.execute("BEGIN")

                # Record transaction
                transaction_metrics = TransactionMetrics(
                    transaction_id=transaction_id,
                    start_time=datetime.utcnow(),
                    isolation_level=isolation_level,
                    locks_held=resources or []
                )
                self._active_transactions[transaction_id] = transaction_metrics

                # Update connection metrics
                if conn_id in self.connection_pool._connection_metrics:
                    self.connection_pool._connection_metrics[conn_id].active_transaction = transaction_id

                logger.debug(f"Started transaction {transaction_id} with isolation {isolation_level.value}")

                # Yield connection for operations
                yield connection, transaction_id

                # Commit if no exception
                await connection.execute("COMMIT")
                transaction_metrics.status = TransactionStatus.COMMITTED
                transaction_metrics.end_time = datetime.utcnow()

                self._stats["successful_transactions"] += 1

                logger.debug(f"Committed transaction {transaction_id}")

        except Exception as e:
            # Rollback on error
            try:
                if 'connection' in locals():
                    await connection.execute("ROLLBACK")
                if transaction_id in self._active_transactions:
                    self._active_transactions[transaction_id].status = TransactionStatus.FAILED
                    self._active_transactions[transaction_id].end_time = datetime.utcnow()
                self._stats["failed_transactions"] += 1
                logger.error(f"Rolled back transaction {transaction_id} due to error: {e}")
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction {transaction_id}: {rollback_error}")
            finally:
                raise
        finally:
            # Release locks
            await self._release_locks(transaction_id)

            # Update connection metrics
            if 'conn_id' in locals() and conn_id in self.connection_pool._connection_metrics:
                self.connection_pool._connection_metrics[conn_id].active_transaction = None

            # Record transaction completion
            if transaction_id in self._active_transactions:
                transaction_metrics = self._active_transactions[transaction_id]
                end_time = transaction_metrics.end_time or datetime.utcnow()
                duration = (end_time - transaction_metrics.start_time).total_seconds()

                # Update statistics
                self._update_stats(duration)

                # Move to history
                self._transaction_history.append(transaction_metrics)

                # Remove from active
                del self._active_transactions[transaction_id]

    async def _acquire_locks(self, transaction_id: str, resources: List[str]):
        """Acquire locks for transaction resources"""
        for resource in resources:
            # Check if resource is locked
            if resource in self._transaction_locks:
                locking_transactions = self._transaction_locks[resource]

                # Check for deadlock
                if self._would_deadlock(transaction_id, resource):
                    await self._resolve_deadlock(transaction_id, resource)
                    continue

                # Wait for locks to be released
                if locking_transactions:
                    event = asyncio.Event()
                    self._lock_waiters[resource].append(event)

                    try:
                        await asyncio.wait_for(event.wait(), timeout=self._deadlock_timeout)
                    except asyncio.TimeoutError:
                        # Timeout - could indicate deadlock
                        raise TimeoutError(f"Timeout acquiring lock for resource {resource}")

            # Acquire lock
            self._transaction_locks[resource].append(transaction_id)

    async def _release_locks(self, transaction_id: str):
        """Release all locks held by transaction"""
        for resource, locking_transactions in list(self._transaction_locks.items()):
            if transaction_id in locking_transactions:
                locking_transactions.remove(transaction_id)

                # Remove empty resource locks
                if not locking_transactions:
                    del self._transaction_locks[resource]

                # Notify waiters
                if resource in self._lock_waiters:
                    for event in self._lock_waiters[resource]:
                        event.set()
                    del self._lock_waiters[resource]

    def _would_deadlock(self, transaction_id: str, resource: str) -> bool:
        """Check if acquiring lock would cause deadlock"""
        # Simplified deadlock detection
        # In production, would use more sophisticated cycle detection
        locking_transactions = self._transaction_locks.get(resource, [])

        for locking_tx in locking_transactions:
            if locking_tx != transaction_id:
                # Check if locking transaction is waiting for resources held by this transaction
                for waited_resource, waiters in self._lock_waiters.items():
                    if waited_resource in self._active_transactions.get(transaction_id, {}).get("locks_held", []):
                        if waiters:  # Someone is waiting for this transaction's resources
                            return True

        return False

    async def _resolve_deadlock(self, transaction_id: str, resource: str):
        """Resolve deadlock by choosing victim transaction"""
        # Simple strategy: choose the transaction with fewer locks
        locking_transactions = self._transaction_locks.get(resource, [])

        if not locking_transactions:
            return

        # Find victim (transaction with fewest locks)
        victim_tx = min(locking_transactions,
                       key=lambda tx: len(self._active_transactions.get(tx, {}).get("locks_held", [])))

        if victim_tx != transaction_id:
            # Cancel the victim transaction
            await self._cancel_transaction(victim_tx)
            self._stats["deadlocks_detected"] += 1
            logger.warning(f"Resolved deadlock: cancelled transaction {victim_tx}")

    async def _cancel_transaction(self, transaction_id: str):
        """Cancel a transaction"""
        if transaction_id in self._active_transactions:
            metrics = self._active_transactions[transaction_id]
            metrics.status = TransactionStatus.ROLLED_BACK
            metrics.end_time = datetime.utcnow()

            # Release locks
            await self._release_locks(transaction_id)

            # Remove from active
            del self._active_transactions[transaction_id]

            logger.info(f"Cancelled transaction {transaction_id}")

    async def _deadlock_detection_loop(self):
        """Background task for deadlock detection"""
        while True:
            try:
                await asyncio.sleep(self._deadlock_check_interval)

                # Check for potential deadlocks
                current_time = datetime.utcnow()

                for transaction_id, metrics in list(self._active_transactions.items()):
                    # Check if transaction has been active too long
                    duration = (current_time - metrics.start_time).total_seconds()

                    if duration > self._deadlock_timeout:
                        logger.warning(f"Long-running transaction detected: {transaction_id} ({duration:.1f}s)")

                        # Could implement automatic rollback for very long transactions
                        if duration > self._deadlock_timeout * 2:
                            await self._cancel_transaction(transaction_id)
                            self._stats["deadlocks_detected"] += 1

            except Exception as e:
                logger.error(f"Error in deadlock detection: {e}")

    def _update_stats(self, duration: float):
        """Update transaction statistics"""
        self._stats["total_transactions"] += 1

        # Update average duration
        if self._stats["average_duration"] == 0:
            self._stats["average_duration"] = duration
        else:
            self._stats["average_duration"] = (
                self._stats["average_duration"] * 0.9 + duration * 0.1
            )

    def get_transaction_statistics(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        return {
            **self._stats,
            "active_transactions": len(self._active_transactions),
            "locked_resources": len(self._transaction_locks),
            "waiting_transactions": sum(len(waiters) for waiters in self._lock_waiters.values()),
            "recent_transactions": [
                {
                    "transaction_id": tx.transaction_id,
                    "duration": (tx.end_time or datetime.utcnow() - tx.start_time).total_seconds(),
                    "status": tx.status.value,
                    "operations_count": tx.operations_count
                }
                for tx in list(self._transaction_history)[-10:]
            ]
        }

    async def rollback_transaction(self, transaction_id: str):
        """Explicitly rollback a transaction"""
        await self._cancel_transaction(transaction_id)


class ConcurrentDataManager:
    """
    High-level data manager with concurrent operation support and isolation
    """

    def __init__(self, database_path: str = "conjecture_concurrent.db"):
        self.database_path = database_path

        # Initialize connection pool
        self.connection_pool = DatabaseConnectionPool(database_path)

        # Initialize transaction manager
        self.transaction_manager = ConcurrentTransactionManager(self.connection_pool)

        # Performance metrics
        self._operation_stats = defaultdict(int)
        self._lock = threading.Lock()

    async def initialize(self):
        """Initialize the concurrent data manager"""
        await self.connection_pool.initialize()
        await self.transaction_manager.start()

        # Initialize database schema
        await self._initialize_schema()

        logger.info("ConcurrentDataManager initialized")

    async def _initialize_schema(self):
        """Initialize database schema with concurrent access considerations"""
        async with self.transaction_manager.transaction() as (conn, tx_id):
            # Create claims table with proper indexing
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    state TEXT DEFAULT 'EXPLORE',
                    tags TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dirty BOOLEAN DEFAULT FALSE,
                    dirty_reasons TEXT,  -- JSON array
                    version INTEGER DEFAULT 1
                )
            """)

            # Create indexes for concurrent access
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_state ON claims(state)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_dirty ON claims(dirty)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_created ON claims(created_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_updated ON claims(updated_at)")

            # Create evaluation results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    result TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (claim_id) REFERENCES claims(id)
                )
            """)

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_claim_id ON evaluation_results(claim_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_provider ON evaluation_results(provider)")

    async def create_claim_concurrent(
        self,
        claim_id: str,
        content: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        state: str = "EXPLORE"
    ) -> bool:
        """Create a claim with concurrent access support"""
        resources = [f"claim:{claim_id}"]

        async with self.transaction_manager.transaction(
            isolation_level=IsolationLevel.SERIALIZABLE,
            resources=resources
        ) as (conn, tx_id):

            # Check if claim already exists
            cursor = await conn.execute(
                "SELECT id FROM claims WHERE id = ?",
                (claim_id,)
            )
            if await cursor.fetchone():
                return False  # Claim already exists

            # Insert claim
            await conn.execute("""
                INSERT INTO claims (id, content, confidence, state, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (claim_id, content, confidence, state, json.dumps(tags or [])))

            with self._lock:
                self._operation_stats["claims_created"] += 1

            logger.debug(f"Created claim {claim_id} in transaction {tx_id}")
            return True

    async def get_claim_concurrent(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get a claim with concurrent access support"""
        resources = [f"claim:{claim_id}"]

        async with self.transaction_manager.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED,
            resources=resources
        ) as (conn, tx_id):

            cursor = await conn.execute(
                "SELECT * FROM claims WHERE id = ?",
                (claim_id,)
            )
            row = await cursor.fetchone()

            if row:
                claim_data = {
                    "id": row[0],
                    "content": row[1],
                    "confidence": row[2],
                    "state": row[3],
                    "tags": json.loads(row[4]) if row[4] else [],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "dirty": bool(row[7]),
                    "dirty_reasons": json.loads(row[8]) if row[8] else [],
                    "version": row[9]
                }

                with self._lock:
                    self._operation_stats["claims_read"] += 1

                logger.debug(f"Read claim {claim_id} in transaction {tx_id}")
                return claim_data

            return None

    async def update_claim_concurrent(
        self,
        claim_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a claim with optimistic concurrency control"""
        resources = [f"claim:{claim_id}"]

        async with self.transaction_manager.transaction(
            isolation_level=IsolationLevel.SERIALIZABLE,
            resources=resources
        ) as (conn, tx_id):

            # Get current version
            cursor = await conn.execute(
                "SELECT version FROM claims WHERE id = ?",
                (claim_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return False  # Claim doesn't exist

            current_version = row[0]
            expected_version = updates.get("version", current_version)

            if current_version != expected_version:
                # Optimistic concurrency failure
                logger.warning(f"Version conflict for claim {claim_id}: expected {expected_version}, found {current_version}")
                return False

            # Build update query
            set_clauses = []
            values = []

            if "content" in updates:
                set_clauses.append("content = ?")
                values.append(updates["content"])

            if "confidence" in updates:
                set_clauses.append("confidence = ?")
                values.append(updates["confidence"])

            if "state" in updates:
                set_clauses.append("state = ?")
                values.append(updates["state"])

            if "tags" in updates:
                set_clauses.append("tags = ?")
                values.append(json.dumps(updates["tags"]))

            if "dirty" in updates:
                set_clauses.append("dirty = ?")
                values.append(updates["dirty"])

            if "dirty_reasons" in updates:
                set_clauses.append("dirty_reasons = ?")
                values.append(json.dumps(updates["dirty_reasons"]))

            # Always update version and timestamp
            set_clauses.extend(["version = ?", "updated_at = CURRENT_TIMESTAMP"])
            values.extend([current_version + 1])

            # Execute update
            if set_clauses:
                values.append(claim_id)
                await conn.execute(f"""
                    UPDATE claims SET {', '.join(set_clauses)}
                    WHERE id = ?
                """, values)

                with self._lock:
                    self._operation_stats["claims_updated"] += 1

                logger.debug(f"Updated claim {claim_id} in transaction {tx_id}")
                return True

            return False

    async def batch_create_claims_concurrent(
        self,
        claims_data: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Batch create claims with optimized concurrent access"""
        results = {}

        # Group claims by potential conflicts (simplified - just batch process)
        async with self.transaction_manager.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED
        ) as (conn, tx_id):

            for claim_data in claims_data:
                claim_id = claim_data["id"]

                # Check if claim exists
                cursor = await conn.execute(
                    "SELECT id FROM claims WHERE id = ?",
                    (claim_id,)
                )

                if await cursor.fetchone():
                    results[claim_id] = False  # Already exists
                else:
                    # Insert claim
                    await conn.execute("""
                        INSERT INTO claims (id, content, confidence, state, tags)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        claim_id,
                        claim_data["content"],
                        claim_data.get("confidence", 0.5),
                        claim_data.get("state", "EXPLORE"),
                        json.dumps(claim_data.get("tags", []))
                    ))

                    results[claim_id] = True

            with self._lock:
                self._operation_stats["batch_claims_created"] += len(claims_data)

            logger.debug(f"Batch created {len(claims_data)} claims in transaction {tx_id}")

        return results

    async def get_dirty_claims_concurrent(self) -> List[Dict[str, Any]]:
        """Get all dirty claims with concurrent access support"""
        async with self.transaction_manager.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED
        ) as (conn, tx_id):

            cursor = await conn.execute("""
                SELECT * FROM claims WHERE dirty = TRUE
                ORDER BY updated_at ASC
            """)

            rows = await cursor.fetchall()

            claims = []
            for row in rows:
                claim_data = {
                    "id": row[0],
                    "content": row[1],
                    "confidence": row[2],
                    "state": row[3],
                    "tags": json.loads(row[4]) if row[4] else [],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "dirty": bool(row[7]),
                    "dirty_reasons": json.loads(row[8]) if row[8] else [],
                    "version": row[9]
                }
                claims.append(claim_data)

            with self._lock:
                self._operation_stats["dirty_claims_read"] += 1

            logger.debug(f"Read {len(claims)} dirty claims in transaction {tx_id}")
            return claims

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._lock:
            operation_stats = dict(self._operation_stats)

        return {
            "operation_statistics": operation_stats,
            "connection_pool": self.connection_pool.get_pool_statistics(),
            "transaction_manager": self.transaction_manager.get_transaction_statistics()
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.transaction_manager.stop()
        await self.connection_pool.close_all()
        logger.info("ConcurrentDataManager cleanup completed")